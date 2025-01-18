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

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.robots.mycobot_client import MyCobotClient

import pinocchio as pin
import numpy as np






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

        urdf_filename = "lerobot/common/robot_devices/robots/mycobot_280_pi.urdf"
        self.model = pin.buildModelFromUrdf(urdf_filename)
        self.data = self.model.createData()


    def ik(self, q_init, target_position, target_rpy=None):
        JOINT_ID = 6
        eps = 1e-4
        IT_MAX = 1000
        DT = 1e-1
        damp = 1e-12

        if target_rpy is None:
            target_rpy = [-3.1416, 0, -1.5708]  # Default RPY if not specified
        target_rotation = pin.utils.rpyToMatrix(target_rpy[0], target_rpy[1], target_rpy[2])
        oMdes = pin.SE3(target_rotation, target_position)
        q = q_init.copy()
        i = 0
        while True:
            pin.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[JOINT_ID].actInv(oMdes)
            err = pin.log(iMd).vector
            if np.linalg.norm(err) < eps:
                return np.array(q), True  # Converged successfully
            if i >= IT_MAX:
                print(f"Warning: max iterations reached without convergence. error norm:{np.linalg.norm(err)}")
                return np.array(q), False  # Did not converge
            J = pin.computeJointJacobian(self.model, self.data, q, JOINT_ID)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * DT)
            i += 1

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        before_read_t = time.perf_counter()
        state = self.get_state()

        # Temporarily use target coord for action, IK will be calculated later
        action = self.mc.get_action()     #[0, 0, 0, 0, 0, 0, state["gripper"]]
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

    def get_state(self, use_robot_data = False) -> dict:
        coords = self.mc.get_coords(use_robot_data)
        gripper_value = self.mc.get_gripper_value(use_robot_data)

        while coords == None:
            print("Can't get coords, sleep for 10ms...")
            time.sleep(0.01)
            coords = self.mc.get_coords(use_robot_data)
            gripper_value = self.mc.get_gripper_value(use_robot_data)

        return {
            "x": coords[0],
            "y": coords[1],
            "z": coords[2],
            "rx": coords[3],
            "ry": coords[4],
            "rz": coords[5],
            "gripper": gripper_value,
        }

    def capture_observation(self) -> dict:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        before_read_t = time.perf_counter()
        state = self.get_state(True)
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
        angles = action[:6].tolist()

        before_write_t = time.perf_counter()
        self.mc.send_angles(angles, 50)
        # print("action:", angles, action[6].item())
        self.mc.set_gripper_value(int(action[6].item()), 50)
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        # TODO(aliberts): return action_sent when motion is limited
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
