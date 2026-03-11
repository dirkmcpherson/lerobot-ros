"""ROS2 backend configuration classes.

Re-uses the existing config structures from ``lerobot_robot_ros`` and extends
them with ``home_position`` / ``home_settle_sec`` so the backend can handle
episode resets automatically.
"""

from dataclasses import dataclass, field
from enum import Enum

from lerobot.cameras import CameraConfig

from ..config import BackendRobotConfig


# ---------------------------------------------------------------------------
# Enums (originally in lerobot_robot_ros.config)
# ---------------------------------------------------------------------------

class ActionType(Enum):
    CARTESIAN_VELOCITY = "cartesian_velocity"
    JOINT_POSITION = "joint_position"
    JOINT_TRAJECTORY = "joint_trajectory"


class GripperActionType(Enum):
    TRAJECTORY = "trajectory"  # Use JointTrajectoryController for gripper
    ACTION = "action"          # Use GripperActionClient


# ---------------------------------------------------------------------------
# Low-level interface config (unchanged from upstream)
# ---------------------------------------------------------------------------

@dataclass
class ROS2InterfaceConfig:
    """Configuration for the low-level ROS 2 interface (publishers / subscribers)."""

    namespace: str = ""

    arm_joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6",
        ]
    )
    gripper_joint_name: str | None = "gripper_joint"

    base_link: str = "base_link"

    # Cartesian velocity limits
    max_linear_velocity: float = 0.10
    max_angular_velocity: float = 0.25  # rad/s

    # Joint position limits (for clamping)
    min_joint_positions: list[float] | None = None
    max_joint_positions: list[float] | None = None

    gripper_open_position: float = 0.0
    gripper_close_position: float = 1.0

    gripper_action_type: GripperActionType = GripperActionType.TRAJECTORY

    # Topic names
    arm_topic: str = "/arm_controller/joint_trajectory"
    gripper_topic: str = "/gripper_controller/gripper_cmd"
    position_topic: str = "/position_controller/commands"
    gripper_traj_topic: str = "/gripper_controller/joint_trajectory"


# ---------------------------------------------------------------------------
# Backend-level config
# ---------------------------------------------------------------------------

@dataclass
class ROS2BackendConfig(BackendRobotConfig):
    """Config that produces a :class:`ROS2Backend` via the factory."""

    action_type: ActionType = ActionType.JOINT_POSITION

    ros2_interface: ROS2InterfaceConfig = field(default_factory=ROS2InterfaceConfig)

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Episode-reset parameters
    home_position: list[float] | None = None
    home_settle_sec: float = 6.0

    # Simulation
    use_sim_time: bool = False


# ---------------------------------------------------------------------------
# Robot-specific presets
# ---------------------------------------------------------------------------

@dataclass
class AnninAR4Config(ROS2BackendConfig):
    """Annin Robotics AR4 robot configuration."""

    action_type: ActionType = ActionType.CARTESIAN_VELOCITY

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            gripper_joint_name="gripper_jaw1_joint",
            base_link="base_link",
            min_joint_positions=[-2.9671, -0.7330, -1.5533, -2.8798, -1.8326, -2.7053],
            max_joint_positions=[2.9671, 1.5708, 0.9076, 2.8798, 1.8326, 2.7053],
            gripper_open_position=0.014,
            gripper_close_position=0.0,
            gripper_action_type=GripperActionType.ACTION,
        ),
    )


@dataclass
class SO101ROSConfig(ROS2BackendConfig):
    """Configuration for the ROS 2 version of SO101."""

    action_type: ActionType = ActionType.JOINT_TRAJECTORY

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            arm_joint_names=["1", "2", "3", "4", "5"],
            gripper_joint_name="6",
            base_link="base",
            min_joint_positions=[-1.91986, -1.74533, -1.74533, -1.65806, -2.79253],
            max_joint_positions=[1.91986, 1.74533, 1.5708, 1.65806, 2.79253],
            gripper_open_position=1.74533,
            gripper_close_position=0.0,
        ),
    )


@dataclass
class KinovaGen3Config(ROS2BackendConfig):
    """Kinova Gen3 7-DOF arm (no gripper)."""

    action_type: ActionType = ActionType.JOINT_TRAJECTORY

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            arm_joint_names=[
                "joint_1", "joint_2", "joint_3", "joint_4",
                "joint_5", "joint_6", "joint_7",
            ],
            gripper_joint_name=None,
            namespace="",
            arm_topic="/joint_trajectory_controller/joint_trajectory",
            min_joint_positions=[-6.2832, -2.24, -6.2832, -2.57, -6.2832, -2.09, -6.2832],
            max_joint_positions=[6.2832, 2.24, 6.2832, 2.57, 6.2832, 2.09, 6.2832],
            gripper_open_position=0.0,
            gripper_close_position=0.8,
            gripper_action_type=GripperActionType.ACTION,
        ),
    )

    home_position: list[float] | None = field(
        default_factory=lambda: [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
    )


@dataclass
class KinovaGen3LiteConfig(ROS2BackendConfig):
    """Kinova Gen3 Lite 6-DOF arm + integrated 2-finger gripper."""

    action_type: ActionType = ActionType.JOINT_TRAJECTORY

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            arm_joint_names=[
                "joint_1", "joint_2", "joint_3",
                "joint_4", "joint_5", "joint_6",
            ],
            gripper_joint_name="right_finger_bottom_joint",
            namespace="",
            arm_topic="/joint_trajectory_controller/joint_trajectory",
            gripper_topic="/gen3_lite_2f_gripper_controller/gripper_cmd",
            min_joint_positions=[-2.68, -2.61, -2.61, -2.6, -2.53, -2.6],
            max_joint_positions=[2.68, 2.61, 2.61, 2.6, 2.53, 2.6],
            gripper_open_position=0.0,
            gripper_close_position=0.85,
            gripper_action_type=GripperActionType.ACTION,
        ),
    )

    home_position: list[float] | None = field(
        # default_factory=lambda: [0.0, -1.57, 0.0, 0.0, 0.0, 0.0]
        default_factory=lambda: [0.0, -1.00, 3.14, 1.57, -2.0, 1.57]

    )
