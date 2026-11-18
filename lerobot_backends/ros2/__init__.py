from .config import (
    ActionType,
    AnninAR4Config,
    GripperActionType,
    KinovaGen3Config,
    KinovaGen3LiteConfig,
    KinovaGen3SweepSimConfig,
    ROS2BackendConfig,
    ROS2InterfaceConfig,
    SO101ROSConfig,
)


def __getattr__(name):
    # ROS2Backend imports rclpy. Loading it lazily keeps the config classes (and the
    # robots built on a fake backend in tests) importable without a ROS installation.
    if name == "ROS2Backend":
        from .backend import ROS2Backend

        return ROS2Backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
