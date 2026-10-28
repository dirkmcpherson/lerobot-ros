"""lerobot RobotConfig for the Kinova Gen3 Lite over ROS 2 (``--robot.type=gen3_lite_ros``).

Registered so the stock lerobot CLIs (``lerobot-record``, ``lerobot-rollout``, ...)
can build the robot once this package is installed (lerobot imports installed
``lerobot_robot_*`` distributions as plugins).
"""

from dataclasses import dataclass

from lerobot.robots import RobotConfig

from lerobot_backends.ros2.config import KinovaGen3LiteConfig


@RobotConfig.register_subclass("gen3_lite_ros")
@dataclass
class Gen3LiteRosConfig(KinovaGen3LiteConfig):
    """Kinova Gen3 Lite (6 joints + gripper), absolute joint-position actions.

    Actions: ``joint_1.pos`` .. ``joint_6.pos`` (rad) and ``gripper.pos`` (raw joint
    units, 0.0 open .. 0.85 closed). Observations: the same 6 joints plus the measured
    ``right_finger_bottom_joint.pos``, and one entry per camera.
    """

    # Per-step cap on |goal - measured| for each arm joint, in rad (0.1 rad at 10 Hz
    # is 1 rad/s). Applies to policy and teleop actions alike; the gripper is not capped.
    max_relative_target: float | None = 0.1
