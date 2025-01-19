import time
from pathlib import Path
import torch
import argparse

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config
    
# Add argument parsing
parser = argparse.ArgumentParser(description='Run inference with ACT policy')
parser.add_argument('--ckpt_path', type=str, required=True,
                   help='Path to the checkpoint directory or file')
parser.add_argument('--device', type=str, default='cpu',
                   choices=['cuda', 'cpu', 'mps'],
                   help='Device to run inference on (default: cpu)')
parser.add_argument('--inference_time', type=int, default=60,
                   help='Duration of inference in seconds (default: 60)')
parser.add_argument('--fps', type=int, default=30,
                   help='Frames per second (default: 30)')
args = parser.parse_args()
    
robot_path = Path("lerobot/configs/robot/mycobot.yaml")
robot_overrides = {}
robot_cfg = init_hydra_config(robot_path, robot_overrides)
robot = make_robot(robot_cfg)
if not robot.is_connected:
    robot.connect()

# Use command line arguments
inference_time_s = args.inference_time
fps = args.fps
device = args.device
policy = ACTPolicy.from_pretrained(args.ckpt_path)
policy.to(device)

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    t1 = time.perf_counter()

    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    # action = action.to("cpu")
    t2 = time.perf_counter()
    print(f"{(t2-t1):.3f}s", action)

    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)