import pygame
import time
import math
from threading import Thread
from enum import Enum
import typing as T
import platform
import threading
from copy import deepcopy
import queue
import socket
import json

if "linux" in platform.platform().lower():
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(20, GPIO.OUT)
    GPIO.setup(21, GPIO.OUT)

class JoyStickKey(Enum):
    StartKey = 6
    SelectKey = 4
    ModeKey = 5
    RLeftKey = 2
    RRightKey = 1
    RTopKey = 3
    RBottomKey = 0
    R1 = 10
    L1 = 9
    ArrowUp = 11
    ArrowDown = 12
    ArrowLeft = 13
    ArrowRight = 14

class JoyStickContinous(Enum):
    LeftXAxis = 0
    LeftYAxis = 1
    L2 = 4
    RightXAxis = 2
    RightYAxis = 3
    R2 = 5

# Keep existing mapping dictionaries unchanged
joystick_key_map = {
    0: JoyStickKey.RBottomKey,
    1: JoyStickKey.RRightKey,
    2: JoyStickKey.RLeftKey,
    3: JoyStickKey.RTopKey,
    9: JoyStickKey.L1,
    10: JoyStickKey.R1,
    4: JoyStickKey.SelectKey,
    6: JoyStickKey.StartKey,
    5: JoyStickKey.ModeKey,
    11: JoyStickKey.ArrowUp,
    12: JoyStickKey.ArrowDown,
    14: JoyStickKey.ArrowRight,
    13: JoyStickKey.ArrowLeft,
}

joystick_continous_map = {
    0: JoyStickContinous.LeftXAxis,
    1: JoyStickContinous.LeftYAxis,
    4: JoyStickContinous.L2,
    2: JoyStickContinous.RightXAxis,
    3: JoyStickContinous.RightYAxis,
    5: JoyStickContinous.R2,
}

def pump_on():
    if "linux" in platform.platform().lower():
        GPIO.output(20, 0)

def pump_off():
    if "linux" in platform.platform().lower():
        GPIO.output(20, 1)
        time.sleep(0.05)
        GPIO.output(21, 0)
        time.sleep(1)
        GPIO.output(21, 1)
        time.sleep(0.05)

class JoyStick:
    def __init__(self, robot=None):
        """Initialize joystick controller
        Args:
            robot: Robot instance to control (optional, for standalone use will create its own)
        """
        self.context = {"running": True}
        self.arm_speed = 50
        self.ratio = 0.5
        self.arm_angle_table = {"init": [0, 0, -90, 0, 0, 0]}
        
        # Initialize robot connection
        if robot is None:
            #from pymycobot import MyCobot
            #self.mc = MyCobot("/dev/ttyAMA0", 1000000)
            from pymycobot import MyCobot280Socket
            self.mc = MyCobot280Socket("192.168.31.84",9000)
            self.mc.set_fresh_mode(0)
        else:
            self.mc = robot

        self.global_states = {
            "enable": True,
            "initialized": True,
            "origin": None,
            "last": None,
            "gripper_val": 20,
            "last_gripper_val": 20,
            "pump": False,
            "move_key_states": {
                JoyStickContinous.LeftXAxis: 0,
                JoyStickContinous.LeftYAxis: 0,
                JoyStickContinous.RightYAxis: 0,
                JoyStickKey.ArrowUp: 0,
                JoyStickKey.ArrowDown: 0,
                JoyStickKey.ArrowLeft: 0,
                JoyStickKey.ArrowRight: 0,
                JoyStickContinous.RightXAxis: 0,
            },
        }

        self.key_hold_timestamp = {
            JoyStickKey.L1: -1,
            JoyStickKey.R1: -1,
            JoyStickContinous.L2: -1,
            JoyStickContinous.R2: -1,
        }

        pygame.init()
        pygame.joystick.init()
        try:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        except:
            print("Please connect the handle first.")
            raise RuntimeError("Joystick not found")

        self._event_queue = queue.Queue()
        self._input_thread = None
        self._move_thread = None
        self._lock = threading.Lock()

        # Add server socket initialization
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(('0.0.0.0', 6789))  # Listen on all interfaces
        self._server_socket.listen(1)
        self._server_thread = None

    def _get_coords(self):
        coords = self.mc.get_coords()
        while coords == None or coords == -1 or len(coords) != 6:
            print("get_coords", coords)
            time.sleep(0.01)
            coords = self.mc.get_coords()
        return coords

    def _get_angles(self):
        angles = self.mc.get_angles()
        while angles == None or angles == -1 or len(angles) != 6:
            print("get_angles", angles)
            time.sleep(0.01)
            angles = self.mc.get_angles()
        return angles

    def _get_gripper_value(self):
        value = self.mc.get_gripper_value()
        while value == None or value == -1:
            print("get_gripper_value", value)
            time.sleep(0.01)
            value = self.mc.get_gripper_value()
        return value

    def start(self):
        """Start joystick control threads"""
        self.global_states["origin"] = self._get_coords()
        self.global_states["last"] = deepcopy(self.global_states["origin"])
        self.global_states["gripper_val"] = self._get_gripper_value()
        self.global_states["last_gripper_val"] = self.global_states["gripper_val"]
        
        print(self.global_states)

        self._move_thread = Thread(target=self._continous_move)
        self._move_thread.start()
        
        # Start server thread
        self._server_thread = Thread(target=self._handle_clients)
        self._server_thread.daemon = True
        self._server_thread.start()
        
        # Start event processing in the main thread
        self._process_events()

    def stop(self):
        """Stop joystick control"""
        self.context["running"] = False
        if self._move_thread:
            self._move_thread.join()
        if self._server_thread:
            self._server_socket.close()
        pygame.quit()

    def _process_events(self):
        """Process joystick events in the main thread"""
        while self.context["running"]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.context["running"] = False
                elif event.type == pygame.JOYBUTTONDOWN:
                    for key_id in range(self.joystick.get_numbuttons()):
                        if self.joystick.get_button(key_id):
                            print("joystick_key_map[key_id]", key_id, joystick_key_map[key_id])
                            self._dispatch_key_action(joystick_key_map[key_id], 1.0)
                elif event.type == pygame.JOYBUTTONUP:
                    self._dispatch_key_action(joystick_key_map[event.button], 0)
                elif event.type == pygame.JOYAXISMOTION:
                    for key_id in range(self.joystick.get_numaxes()):
                        axis = self.joystick.get_axis(key_id)
                        if joystick_continous_map[key_id] in (JoyStickContinous.L2, JoyStickContinous.R2):
                            axis = math.ceil(axis)
                            if int(axis) == -1:
                                continue
                        #print("key_id", key_id, "axis", axis, "id", joystick_continous_map[key_id])
                        self._dispatch_key_action(joystick_continous_map[key_id], axis)
                elif event.type == pygame.JOYHATMOTION:
                    print("joystick_key_map[self.joystick.get_hat(0)]", joystick_key_map[self.joystick.get_hat(0)])
                    self._dispatch_key_action(joystick_key_map[self.joystick.get_hat(0)], 1.0)
            
            # Add a small sleep to prevent high CPU usage
            time.sleep(0.01)

    def _dispatch_key_action(self, key: T.Union[JoyStickKey, JoyStickContinous], value: float):
        """Handle joystick key actions"""
        not_zero = lambda x: x < -0.1 or x > 0.1
        is_zero = lambda x: -0.1 < x < 0.1

        if key == JoyStickKey.StartKey:
            if self.mc.is_all_servo_enable() != 1:
                self._blink_color([(255,0,0)]*3)
            else:
                self._blink_color([(0,255,0)]*3)
                self.global_states["initialized"] = True

        elif key == JoyStickKey.R1:
            self.mc.send_angles(self.arm_angle_table["init"], self.arm_speed)
            self.global_states["enable"] = True
            time.sleep(3)
            self.global_states["origin"] = self._get_coords()

        if not self.global_states["enable"] or not self.global_states["initialized"]:
            return

        # Handle movement keys
        if key in self.global_states["move_key_states"]:
            if self.global_states["origin"] is None:
                coords = self._get_coords()
                with self._lock:
                    self.global_states["origin"] = coords
            
            if is_zero(value):
                self.global_states["move_key_states"][key] = 0
            else:
                self.global_states["move_key_states"][key] = value
            
        #print("key", key, "states", self.global_states["move_key_states"], "value", self.global_states["origin"])

        # Handle function keys
        if key == JoyStickContinous.L2 and not_zero(value):
            self.mc.release_all_servos()
        elif key == JoyStickContinous.R2 and not_zero(value):
            self.mc.power_on()
        elif key == JoyStickKey.L1 and not_zero(value):
            self.mc.send_angles([0, 0, 0, 0, 0, 0], 50)
            time.sleep(3)
            self.global_states["origin"] = None

        # Handle tool controls
        if key == JoyStickKey.RLeftKey:
            self.global_states["last_gripper_val"] = self.global_states["gripper_val"]
            self.global_states["gripper_val"] = min(100, self.global_states["gripper_val"] + 2)
            self.mc.set_gripper_value(self.global_states["gripper_val"], 50)
        elif key == JoyStickKey.RTopKey:
            self.global_states["last_gripper_val"] = self.global_states["gripper_val"]
            self.global_states["gripper_val"] = max(20, self.global_states["gripper_val"] - 2)
            self.mc.set_gripper_value(self.global_states["gripper_val"], 50)
        elif key == JoyStickKey.RBottomKey:
            pump_on()
            self.ratio -= 0.2
            if self.ratio < 0.2:
                self.ratio = 0.2
            print("ratio:", self.ratio)
            val = int(255 * self.ratio / 2)
            self.mc.set_color(0, val, 0)
        elif key == JoyStickKey.RRightKey:
            pump_off()
            self.ratio += 0.2
            if self.ratio > 2:
                self.ratio = 2
            print("ratio:", self.ratio)
            val = int(255 * self.ratio / 2)
            self.mc.set_color(0, val, 0)


    def _blink_color(self, colors, delay=0.5):
        """Helper to blink robot LED colors"""
        for r,g,b in colors:
            self.mc.set_color(0, 0, 0)
            time.sleep(delay)
            self.mc.set_color(r, g, b)
            time.sleep(delay)

    def _continous_move(self):
        """Handle continuous movement updates"""
        move_speed = 100
        ratio = 0.5
        not_zero = lambda x: x < -0.1 or x > 0.1
        
        while self.context["running"]:
            if not self.global_states["enable"] or not self.global_states["initialized"]:
                time.sleep(0.05)
                continue

            if self.global_states["origin"] is None:
                time.sleep(0.05)
                continue

            moving = False
            for key, value in self.global_states["move_key_states"].items():
                if not_zero(value):
                    moving = True
                    self._update_coordinates(key, value, ratio)

            if moving:
                start_time = time.perf_counter()
                self.mc.send_coords(self.global_states["origin"], move_speed, 1)
                dt = time.perf_counter() - start_time
                print(f"moving {self.global_states['origin']}, send_coords took {dt*1000:.2f}ms")

            time.sleep(0.02)

            # looks like _get_coords() is high cost, while _get_angles() is ok.
            # if moving:
            #     angles = self._get_angles()
            #     print("angles", angles)

    def _update_coordinates(self, key, value, ratio):
        """Update target coordinates based on joystick input"""
        ratio = self.ratio
        with self._lock:
            if self.global_states["origin"] is None:
                return

            self.global_states["last"] = deepcopy(self.global_states["origin"])

            if key == JoyStickContinous.LeftXAxis:
                self.global_states["origin"][0] += value * ratio * 2
            elif key == JoyStickContinous.LeftYAxis:
                self.global_states["origin"][1] += value * ratio * 2
            elif key == JoyStickContinous.RightYAxis:
                self.global_states["origin"][2] += value * ratio
            elif key == JoyStickContinous.RightXAxis:
                self.global_states["origin"][5] -= value * ratio
                if self.global_states["origin"][5] > 180:
                    self.global_states["origin"][5] -= 360
                elif self.global_states["origin"][5] < -180:
                    self.global_states["origin"][5] += 360
            elif key == JoyStickKey.ArrowRight:
                self.global_states["origin"][3] += 1 * ratio
                if self.global_states["origin"][3] > 180:
                    self.global_states["origin"][3] -= 360
            elif key == JoyStickKey.ArrowLeft:
                self.global_states["origin"][3] -= 1 * ratio
                if self.global_states["origin"][3] < -180:
                    self.global_states["origin"][3] += 360
            elif key == JoyStickKey.ArrowDown:
                self.global_states["origin"][4] -= 1 * ratio
                if self.global_states["origin"][4] < -180:
                    self.global_states["origin"][4] += 360
            elif key == JoyStickKey.ArrowUp:
                self.global_states["origin"][4] += 1 * ratio
                if self.global_states["origin"][4] > 180:
                    self.global_states["origin"][4] -= 360

    def get_current_coords(self) -> list[float] | None:
        """Get current target coordinates in a thread-safe way
        
        Returns:
            list[float] | None: Current coordinates [x,y,z,rx,ry,rz] or None if not initialized
        """
        with self._lock:
            if self.global_states["last"] is None:
                return None
            return deepcopy(self.global_states["last"])

    def get_action(self) -> list[float] | None:
        with self._lock:
            if self.global_states["origin"] is None:
                return None
            ret = deepcopy(self.global_states["origin"])
            ret.append(self.global_states["gripper_val"])
            return ret
    
    def _handle_clients(self):
        """Handle client connections and requests"""
        while self.context["running"]:
            try:
                self._server_socket.settimeout(1.0)
                client_socket, address = self._server_socket.accept()
                print(f"Client connected from {address}")
                
                try:
                    while self.context["running"]:
                        # Receive request
                        data = client_socket.recv(1024).decode('utf-8')
                        if not data:
                            break
                            
                        # Parse request
                        request = json.loads(data)
                        command = request.get('command')
                        params = request.get('params', {})
                        
                        # Handle commands
                        response = {'status': 'error', 'data': None}
                        if command == 'get_gripper_value':
                            use_robot_data = params.get('use_robot_data', False)
                            if use_robot_data:
                                value = self._get_gripper_value()
                            else:
                                with self._lock:
                                    value = self.global_states["last_gripper_val"]
                            response = {
                                'status': 'ok',
                                'data': value
                            }
                        elif command == 'get_coords':
                            use_robot_data = params.get('use_robot_data', False)
                            if use_robot_data:
                                coords = self._get_angles() # hack!!! _get_coords()
                            else:
                                with self._lock:
                                    coords = deepcopy(self.global_states["last"])
                            response = {
                                'status': 'ok',
                                'data': coords if coords is None else list(coords)
                            }
                        elif command == 'get_action':
                            action = self.get_action()
                            response = {
                                'status': 'ok',
                                'data': action if action is None else list(action)
                            }
                        elif command == 'set_gripper_value':
                            value = params.get('value')
                            speed = params.get('speed', 50)
                            with self._lock:
                                self.global_states["last_gripper_val"] = self.global_states["gripper_val"]
                                self.global_states["gripper_val"] = value
                                self.mc.set_gripper_value(value, speed)
                            response = {
                                'status': 'ok',
                                'data': None
                            }
                        elif command == 'send_angles':
                            angles = params.get('angles')
                            speed = params.get('speed', 50)
                            if not angles or len(angles) != 6:
                                response = {
                                    'status': 'error',
                                    'error': 'Invalid angles parameter'
                                }
                            else:
                                self.mc.send_angles(angles, speed)
                                response = {
                                    'status': 'ok',
                                    'data': None
                                }
                            
                        # Send response
                        response_data = json.dumps(response).encode('utf-8')
                        client_socket.sendall(response_data + b'\n')  # Add newline for message boundary
                        
                except Exception as e:
                    print(f"Error handling client request: {e}")
                finally:
                    client_socket.close()
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.context["running"]:
                    print(f"Error accepting client connection: {e}")

def main():
    """Standalone operation"""
    joystick = JoyStick()
    try:
        joystick.start()  # This will now block in the main thread
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        joystick.stop()

if __name__ == "__main__":
    main()
