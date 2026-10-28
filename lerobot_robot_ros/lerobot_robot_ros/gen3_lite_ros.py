"""lerobot Robot for the Kinova Gen3 Lite over ROS 2 (see ``config_gen3_lite_ros``)."""

from lerobot_backends.robot import BackendRobot

from .config_gen3_lite_ros import Gen3LiteRosConfig


class Gen3LiteRos(BackendRobot):
    config_class = Gen3LiteRosConfig
    name = "gen3_lite_ros"
