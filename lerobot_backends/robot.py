"""BackendRobot — generic lerobot Robot that delegates to a RobotBackend."""

import logging
from typing import Any

from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .backend import RobotBackend
from .config import BackendRobotConfig
from .factory import make_backend

logger = logging.getLogger(__name__)


class BackendRobot(Robot):
    """Generic Robot that delegates to a :class:`RobotBackend`."""

    config_class = BackendRobotConfig
    name = "backend"

    def __init__(self, config: BackendRobotConfig):
        super().__init__(config)
        self.config = config
        self.backend: RobotBackend = make_backend(config)

    # --- Feature descriptors (forwarded from backend) ---

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return self.backend.observation_features

    @property
    def action_features(self) -> dict[str, type]:
        return self.backend.action_features

    # --- Lifecycle ---

    @property
    def is_connected(self) -> bool:
        return self.backend.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.backend.connect()

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.backend.disconnect()
        logger.info(f"{self} disconnected.")

    # --- Observation / Action ---

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self.backend.get_observation()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Optional safety clipping (shared for all backends)
        if self.config.max_relative_target is not None:
            obs = self.backend.get_observation()
            goal_present_pos = {}
            for key, goal in action.items():
                if isinstance(goal, (int, float)):
                    present = obs.get(key, 0.0)
                    goal_present_pos[key] = (goal, present)
            if goal_present_pos:
                action = ensure_safe_goal_position(
                    goal_present_pos, self.config.max_relative_target
                )

        return self.backend.send_action(action)

    # --- Calibration (not applicable at this level) ---

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # --- Backward compatibility ---

    @property
    def ros2_interface(self):
        """Legacy accessor for ROS2 backend's interface. Raises if backend is not ROS2."""
        from .ros2.backend import ROS2Backend
        if isinstance(self.backend, ROS2Backend):
            return self.backend._interface
        raise AttributeError("ros2_interface is only available for ROS2 backends")
