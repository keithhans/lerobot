#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from dataclasses import dataclass, field, replace

import torch

from pymycobot import MyCobot
from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.robots.joy_control import JoyStick

@dataclass
class MyCobotConfig:
    robot_type: str | None = "mycobot"
    cameras: dict[str, Camera] = field(default_factory=lambda: {})
    # TODO(aliberts): add feature with max_relative target
    # TODO(aliberts): add comment on max_relative target
    max_relative_target: list[float] | float | None = None


class MyCobot280:
    """Wrapper of stretch_body.robot.Robot"""

    def __init__(self, config: MyCobotConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = MyCobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.robot_type = self.config.robot_type
        self.cameras = self.config.cameras
        self.is_connected = False
        self.joystick = None
        self.logs = {}

        self.mc = None


        self.state_keys = None
        self.action_keys = None

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self) -> None:
        self.mc = MyCobot("/dev/ttyAMA0", 1000000, debug=False)
        self.mc.set_fresh_mode(0)
        self.is_connected = self.mc is not None
        if not self.is_connected:
            print("Can't connect to mycobot! ")
            raise ConnectionError()

        print("mycobot connected")

        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.run_calibration()

    def run_calibration(self) -> None:
        pass
        #    self.home()

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        if self.joystick is None:
            self.joystick = JoyStick(self.mc)
            self.joystick.start()

        before_read_t = time.perf_counter()
        state = self.get_state()
        action = [0, 0, 0, 0, 0, 0]
        #action = self.mc.get_angles()
        #count = 0
        #while action == None:
        #    time.sleep(0.1)
        #    count += 1
        #    action = self.mc.get_angles()
        #print(f"get_angles retry count:{count}")
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        print(state, action)

        # robot move is done outside so we just read the data
        #before_write_t = time.perf_counter()
        #self.joystick.do_motion(action)
        #self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        if self.state_keys is None:
            self.state_keys = list(state)

        if not record_data:
            return

        state = torch.as_tensor(list(state.values()))
        action = torch.as_tensor(list(action))

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def get_state(self) -> dict:
        coords = self.joystick.get_current_coords()
        while coords == None:
            print("Can't get coords, sleep for 1ms...")
            time.sleep(0.001)
            coords = self.mc.get_coords()
        return {
            "x": coords[0],
            "y": coords[1],
            "z": coords[2],
            "rx": coords[3],
            "ry": coords[4],
            "rz": coords[5],
        }

    def capture_observation(self) -> dict:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        before_read_t = time.perf_counter()
        state = self.get_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        if self.state_keys is None:
            self.state_keys = list(state)

        state = torch.as_tensor(list(state.values()))

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        # convert action to angles
        angles = [action[0], action[1], action[2], action[3], action[4], action[5]]

        before_write_t = time.perf_counter()
        self.mc.send_angles(angles, 50)
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        # TODO(aliberts): return action_sent when motion is limited
        return action

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self) -> None:
        #self.stop()
        if self.joystick is not None:
            self.joystick.stop()

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False

    def __del__(self):
        self.disconnect()
