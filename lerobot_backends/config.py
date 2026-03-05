"""Base configuration for BackendRobot."""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@dataclass
class BackendRobotConfig(RobotConfig):
    """Base config for BackendRobot — extended by each backend's own config."""

    # Safety: clip relative action magnitude (shared across all backends).
    max_relative_target: int | None = None

    # Camera configs (backend-specific meaning: ROS2 uses ROS cameras,
    # ManiSkill embeds camera config in env_kwargs).
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
