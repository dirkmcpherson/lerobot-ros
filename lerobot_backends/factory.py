"""Backend factory — create a RobotBackend from a config object."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import RobotBackend


def make_backend(config) -> "RobotBackend":
    """Create a backend instance from a config object.

    The config's type determines which backend is instantiated.
    Imports are deferred so that optional dependencies (e.g. rclpy,
    mani_skill) are only required when actually used.
    """
    from .maniskill.config import ManiSkillBackendConfig
    from .ros2.config import ROS2BackendConfig

    if isinstance(config, ROS2BackendConfig):
        from .ros2.backend import ROS2Backend
        return ROS2Backend(config)

    if isinstance(config, ManiSkillBackendConfig):
        from .maniskill.backend import ManiSkillBackend
        return ManiSkillBackend(config)

    raise ValueError(f"Unknown backend config type: {type(config).__name__}")
