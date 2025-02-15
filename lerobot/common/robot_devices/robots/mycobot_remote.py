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
from dataclasses import replace

import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.mycobot_client import MyCobotClient
from lerobot.common.robot_devices.robots.configs import MyCobotRobotConfig

class MyCobot280:
    """Wrapper of stretch_body.robot.Robot"""

    def __init__(self, config: MyCobotRobotConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            self.config = MyCobotRobotConfig()
        else:
            # Overwrite config arguments using kwargs
            self.config = replace(config, **kwargs)

        self.robot_type = self.config.type
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.is_connected = False

        self.logs = {}

        self.mc = None


        self.state_keys = None
        self.action_keys = None
        
        self.model = None
        self.data = None

        self._is_shutting_down = False



    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        #action_names = self.get_motor_names(self.leader_arms)
        #state_names = self.get_motor_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ['J1', 'J2','J3','J4','J5','J6','gripper'],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ['J1', 'J2','J3','J4','J5','J6','gripper'],
            },
        }

    def connect(self) -> None:
        self.mc = MyCobotClient()
        # todo: add error checking later
        self.mc.connect()
        self.is_connected = True
        
        print("mycobot connected")

        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()


    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        before_read_t = time.perf_counter()
        state = self.get_state()
        action = self.mc.get_action()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        #print(self.logs["read_pos_dt_s"], state, action)

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
        state = self.mc.get_state()

        return {
            "j0": state[0],
            "j1": state[1],
            "j2": state[2],
            "j3": state[3],
            "j4": state[4],
            "j5": state[5],
            "gripper": state[6],
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

        before_write_t = time.perf_counter()
        self.mc.send_action(action.tolist(), 50)
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        # get the actual action since user might override the action
        # action = self.mc.get_action()
        # action = torch.as_tensor(list(action))

        return action

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self) -> None:
        """Disconnect from robot and cameras"""
        if self._is_shutting_down:
            return

        self.mc.disconnect()

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False

    def __del__(self):
        """Cleanup when object is deleted"""
        try:
            # Set flag to prevent disconnect during shutdown
            self._is_shutting_down = True
            self.disconnect()
        except:
            # Ignore any errors during shutdown
            pass
