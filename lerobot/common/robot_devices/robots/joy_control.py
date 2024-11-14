import pygame
import time
import math
from threading import Thread
from enum import Enum
import typing as T
import platform

if "linux" in platform.platform().lower():
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(20, GPIO.OUT)
    GPIO.setup(21, GPIO.OUT)

class JoyStickKey(Enum):
    StartKey = 7
    SelectKey = 6
    ModeKey = 8
    RLeftKey = 2
    RRightKey = 1
    RTopKey = 3
    RBottomKey = 0
    R1 = 5
    L1 = 4
    LJoyStickKey = 9
    RJoyStickKey = 10
    ArrowUp = (0, 1)
    ArrowDown = (0, -1)
    ArrowLeft = (-1, 0)
    ArrowRight = (1, 0)
    ArrowReleased = (0, 0)

class JoyStickContinous(Enum):
    LeftXAxis = 0
    LeftYAxis = 1
    L2 = 2
    RightXAxis = 3
    RightYAxis = 4
    R2 = 5

# Keep existing mapping dictionaries unchanged
joystick_key_map = {
    0: JoyStickKey.RBottomKey,
    1: JoyStickKey.RRightKey,
    2: JoyStickKey.RLeftKey,
    3: JoyStickKey.RTopKey,
    4: JoyStickKey.L1,
    5: JoyStickKey.R1,
    6: JoyStickKey.SelectKey,
    7: JoyStickKey.StartKey,
    8: JoyStickKey.ModeKey,
    9: JoyStickKey.LJoyStickKey,
    10: JoyStickKey.RJoyStickKey,
    (0, 1): JoyStickKey.ArrowUp,
    (0, -1): JoyStickKey.ArrowDown,
    (1, 0): JoyStickKey.ArrowRight,
    (-1, 0): JoyStickKey.ArrowLeft,
    (0, 0): JoyStickKey.ArrowReleased,
}

joystick_continous_map = {
    0: JoyStickContinous.LeftXAxis,
    1: JoyStickContinous.LeftYAxis,
    2: JoyStickContinous.L2,
    3: JoyStickContinous.RightXAxis,
    4: JoyStickContinous.RightYAxis,
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
        self.sampling_rate = 10
        self.arm_angle_table = {"init": [0, 0, -90, 0, 0, 0]}
        
        # Initialize robot connection
        if robot is None:
            from pymycobot import MyCobot
            self.mc = MyCobot("/dev/ttyAMA0", 1000000)
            self.mc.set_fresh_mode(0)
        else:
            self.mc = robot

        self.global_states = {
            "enable": True,
            "initialized": True,
            "origin": None,
            "gripper_val": 20,
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
                JoyStickKey.ArrowReleased: 0,
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

        self._input_thread = None
        self._move_thread = None

    def start(self):
        """Start joystick control threads"""
        self._input_thread = Thread(target=self._retreive_joystick_input)
        self._move_thread = Thread(target=self._continous_move)
        self._input_thread.start()
        self._move_thread.start()

    def stop(self):
        """Stop joystick control"""
        self.context["running"] = False
        if self._input_thread:
            self._input_thread.join()
        if self._move_thread:
            self._move_thread.join()
        pygame.quit()

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
            self.global_states["origin"] = None

        if not self.global_states["enable"] or not self.global_states["initialized"]:
            return

        # Handle movement keys
        if key in self.global_states["move_key_states"]:
            if self.global_states["origin"] is None:
                coords = []
                for _ in range(5):
                    coords = self.mc.get_coords()
                    # print(f"dispatch coords:{coords}")
                    if len(coords) != 0:
                        break

                if len(coords) != 0:
                    self.global_states["origin"] = coords
                else:
                    print("Can't get coords.")
                    return   
            
            if is_zero(value):
                self.global_states["move_key_states"][key] = 0
            else:
                self.global_states["move_key_states"][key] = value

            if key == JoyStickKey.ArrowReleased:
                for arrow_key in [JoyStickKey.ArrowUp, JoyStickKey.ArrowDown, 
                                JoyStickKey.ArrowLeft, JoyStickKey.ArrowRight]:
                    self.global_states["move_key_states"][arrow_key] = 0

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
            self.global_states["gripper_val"] = min(100, self.global_states["gripper_val"] + 5)
            self.mc.set_gripper_value(self.global_states["gripper_val"], 50)
        elif key == JoyStickKey.RTopKey:
            self.global_states["gripper_val"] = max(20, self.global_states["gripper_val"] - 5)
            self.mc.set_gripper_value(self.global_states["gripper_val"], 50)
        elif key == JoyStickKey.RBottomKey:
            pump_on()
        elif key == JoyStickKey.RRightKey:
            pump_off()

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
                self.mc.send_coords(self.global_states["origin"], move_speed, 1)

            time.sleep(0.01)

    def _update_coordinates(self, key, value, ratio):
        """Update target coordinates based on joystick input"""
        if self.global_states["origin"] is None:
            return
            
        if key == JoyStickContinous.LeftXAxis:
            self.global_states["origin"][1] -= value * ratio * 2
        elif key == JoyStickContinous.LeftYAxis:
            self.global_states["origin"][0] -= value * ratio * 2
        elif key == JoyStickContinous.RightYAxis:
            self.global_states["origin"][2] -= value * ratio
        elif key == JoyStickContinous.RightXAxis:
            self.global_states["origin"][5] -= value * ratio
        elif key == JoyStickKey.ArrowUp:
            self.global_states["origin"][3] += 1
        elif key == JoyStickKey.ArrowDown:
            self.global_states["origin"][3] -= 1
        elif key == JoyStickKey.ArrowRight:
            self.global_states["origin"][4] -= 1
        elif key == JoyStickKey.ArrowLeft:
            self.global_states["origin"][4] += 1

    def _retreive_joystick_input(self):
        """Main joystick input processing loop"""
        while self.context["running"]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.context["running"] = False
                elif event.type == pygame.JOYBUTTONDOWN or event.type == pygame.JOYBUTTONUP:
                    for key_id in range(self.joystick.get_numbuttons()):
                        if self.joystick.get_button(key_id):
                            self._dispatch_key_action(joystick_key_map[key_id], 1.0)
                elif event.type == pygame.JOYAXISMOTION:
                    for key_id in range(self.joystick.get_numaxes()):
                        axis = self.joystick.get_axis(key_id)
                        if joystick_continous_map[key_id] in (JoyStickContinous.L2, JoyStickContinous.R2):
                            axis = math.ceil(axis)
                            if int(axis) == -1:
                                continue
                        self._dispatch_key_action(joystick_continous_map[key_id], axis)
                elif event.type == pygame.JOYHATMOTION:
                    self._dispatch_key_action(joystick_key_map[self.joystick.get_hat(0)], 1.0)

def main():
    """Standalone operation"""
    joystick = JoyStick()
    try:
        joystick.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        joystick.stop()

if __name__ == "__main__":
    main()
