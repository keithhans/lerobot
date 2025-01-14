import time
import socket
import pickle
import struct
import torch
from pathlib import Path

from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config

def recv_msg(sock):
    # First receive the message length as a 4-byte integer
    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    
    # Now receive the message data
    data = bytearray()
    while len(data) < msglen:
        packet = sock.recv(min(msglen - len(data), 4096))
        if not packet:
            return None
        data.extend(packet)
    return data

def send_msg(sock, msg):
    # Prefix message with its length as a 4-byte integer
    msg_length = struct.pack('>I', len(msg))
    sock.sendall(msg_length + msg)

def main():
    # Initialize robot
    robot_path = Path("lerobot/configs/robot/mycobot.yaml")
    robot_overrides = {}
    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)
    if not robot.is_connected:
        robot.connect()

    # Set up client socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 6666))

    inference_time_s = 60
    fps = 30

    try:
        for _ in range(inference_time_s * fps):
            start_time = time.perf_counter()

            # Read the follower state and access the frames from the cameras
            observation = robot.capture_observation()

            # Convert to pytorch format: channel first and float32 in [0,1]
            # with batch dimension
            for name in observation:
                if "image" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()
                observation[name] = observation[name].unsqueeze(0)

            # Send observation to server
            observation_data = pickle.dumps(observation)
            send_msg(client_socket, observation_data)

            # Receive action from server
            action_data = recv_msg(client_socket)
            if action_data is None:
                raise ConnectionError("Connection lost while receiving action")
            
            action = pickle.loads(action_data)

            # Order the robot to move
            robot.send_action(action)
            print(action)

            dt_s = time.perf_counter() - start_time
            busy_wait(1 / fps - dt_s)

    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        client_socket.close()

if __name__ == "__main__":
    main() 
