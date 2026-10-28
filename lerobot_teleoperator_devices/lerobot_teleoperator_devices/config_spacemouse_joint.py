from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

GEN3_LITE_JOINTS = [f"joint_{i}" for i in range(1, 7)]


@TeleoperatorConfig.register_subclass("spacemouse_joint")
@dataclass
class SpacemouseJointTeleopConfig(TeleoperatorConfig):
    """SpaceMouse -> differential IK -> absolute joint targets (defaults: Gen3 Lite)."""

    # "spacemouse" reads the HID device; "scripted" replays a fixed slow motion
    # (no device needed, for smoke tests).
    device: str = "spacemouse"

    # URDF for the IK chain. None: generate it from kortex_description's
    # gen3_lite_gen3_lite_2f.xacro (needs the ROS workspace sourced).
    urdf_path: str | None = None
    base_link: str = "base_link"
    ee_link: str = "tool_frame"

    joint_names: list[str] = field(default_factory=lambda: list(GEN3_LITE_JOINTS))
    min_joint_positions: list[float] = field(
        default_factory=lambda: [-2.68, -2.61, -2.61, -2.6, -2.53, -2.6]
    )
    max_joint_positions: list[float] = field(
        default_factory=lambda: [2.68, 2.61, 2.61, 2.6, 2.53, 2.6]
    )
    gripper_joint_name: str | None = "right_finger_bottom_joint"
    gripper_open_position: float = 0.0
    gripper_closed_position: float = 0.85
    # On takeover the gripper toggle starts closed if the measured gripper joint is
    # past this (a grasp stops well short of fully closed, so not the midpoint).
    gripper_closed_threshold: float = 0.1

    joint_states_topic: str = "/joint_states"
    use_sim_time: bool = False

    linear_scale: float = 0.15   # m/s at full deflection
    angular_scale: float = 0.4   # rad/s at full deflection
    dead_zone: float = 0.05
    input_smoothing: float = 0.5
    damping: float = 0.05

    # Cap on |target - measured| per joint per step (rad). Keep it equal to the robot's
    # max_relative_target so recorded actions are exactly what the robot was sent.
    max_joint_step: float | None = 0.1

    # Integrate each IK step from the previous target (kept within max_joint_step of the
    # measured pose) rather than from the measured pose, which lags the command and makes
    # every step restart from behind. False: integrate from the measured pose.
    integrate_from_target: bool = True

    # A gap this long between get_action() calls means control was handed over
    # (e.g. a DAgger correction starting): re-sync targets and gripper to the robot.
    resync_after_sec: float = 0.5

    def __post_init__(self):
        if self.device not in ("spacemouse", "scripted"):
            raise ValueError(f"device must be 'spacemouse' or 'scripted', got '{self.device}'")
