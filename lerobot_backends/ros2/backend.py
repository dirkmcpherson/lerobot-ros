"""ROS2 backend — wraps ROS2Interface + cameras as a RobotBackend."""

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs

from .config import ActionType, ROS2BackendConfig
from .ros_interface import ROS2Interface

logger = logging.getLogger(__name__)


class ROS2Backend:
    """RobotBackend implementation for ROS2-controlled robots."""

    def __init__(self, config: ROS2BackendConfig):
        self.config = config
        self._interface = ROS2Interface(config.ros2_interface, config.action_type)
        self._cameras = make_cameras_from_configs(config.cameras)
        self._connected = False

    # --- Feature descriptors ---

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        iface = self.config.ros2_interface
        all_joints = iface.arm_joint_names.copy()
        if iface.gripper_joint_name:
            all_joints.append(iface.gripper_joint_name)
        features: dict[str, type | tuple] = {f"{j}.pos": float for j in all_joints}
        for cam_name, cam_cfg in self.config.cameras.items():
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        iface = self.config.ros2_interface
        if self.config.action_type == ActionType.CARTESIAN_VELOCITY:
            features: dict[str, type] = {
                "linear_x.vel": float,
                "linear_y.vel": float,
                "linear_z.vel": float,
                "angular_x.vel": float,
                "angular_y.vel": float,
                "angular_z.vel": float,
            }
            if iface.gripper_joint_name:
                features["gripper.pos"] = float
            return features

        # JOINT_POSITION / JOINT_TRAJECTORY
        features = {f"{j}.pos": float for j in iface.arm_joint_names}
        if iface.gripper_joint_name:
            features["gripper.pos"] = float
        return features

    # --- Lifecycle ---

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        for cam in self._cameras.values():
            cam.connect()
        self._interface.connect()
        self._connected = True

    def disconnect(self) -> None:
        for cam in self._cameras.values():
            cam.disconnect()
        self._interface.disconnect()
        self._connected = False

    # --- Episode reset ---

    def reset(self) -> dict[str, Any]:
        """Move to home position (if configured), open gripper, return obs."""
        iface = self.config.ros2_interface
        if self.config.home_position is not None:
            logger.info(f"Resetting to home: {self.config.home_position}")
            self._interface.send_joint_position_command(
                self.config.home_position, unnormalize=False, time_from_start_sec=5.0
            )
            if iface.gripper_joint_name:
                self._interface.send_gripper_command(
                    iface.gripper_open_position, unnormalize=False
                )
            time.sleep(self.config.home_settle_sec)
        return self.get_observation()

    # --- Observation ---

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        joint_state = self._interface.joint_state
        if joint_state is None:
            raise ValueError("Joint state is not available yet.")
        obs.update({f"{j}.pos": pos for j, pos in joint_state["position"].items()})

        for cam_key, cam in self._cameras.items():
            start = time.perf_counter()
            try:
                obs[cam_key] = cam.async_read(timeout_ms=300)
            except Exception as e:
                logger.error(f"Failed to read camera {cam_key}: {e}")
                obs[cam_key] = None
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"Read {cam_key}: {dt_ms:.1f}ms")

        return obs

    # --- Action ---

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        iface_cfg = self.config.ros2_interface

        if self.config.action_type == ActionType.CARTESIAN_VELOCITY:
            linear = (
                action["linear_x.vel"],
                action["linear_y.vel"],
                action["linear_z.vel"],
            )
            angular = (
                action["angular_x.vel"],
                action["angular_y.vel"],
                action["angular_z.vel"],
            )
            self._interface.servo(linear=linear, angular=angular)
        elif self.config.action_type in (ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY):
            joint_positions = [
                action[f"{j}.pos"] for j in iface_cfg.arm_joint_names
            ]
            self._interface.send_joint_position_command(joint_positions)

        if "gripper.pos" in action:
            self._interface.send_gripper_command(action["gripper.pos"], unnormalize=False)

        return action

    # --- Sim-only properties (not applicable for real hardware) ---

    @property
    def episode_success(self) -> bool | None:
        return None

    @property
    def episode_reward(self) -> float:
        return 0.0
