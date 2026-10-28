"""lerobot RobotConfig for the Kinova Gen3 Lite over ROS 2 (``--robot.type=gen3_lite_ros``).

Registered so the stock lerobot CLIs (``lerobot-record``, ``lerobot-rollout``, ...)
can build the robot once this package is installed (lerobot imports installed
``lerobot_robot_*`` distributions as plugins).
"""

from dataclasses import dataclass, field

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

    # Workspace guard (lerobot_backends.ros2.workspace_guard), checked on every command.
    # Table plane: z in the base_link frame, whose z=0 is the bottom face of the arm's
    # base. The arm sits on a mount, so the real table is at or below z=0 and a plane
    # at 0 is conservative. Every moving link must stay plane_margin above it.
    guard_enabled: bool = True
    plane_z: float = 0.0
    plane_margin: float = 0.015
    self_collision_margin: float = 0.005
    # Box the tool frame must stay inside (base_link frame, metres). Default: about
    # +-15 cm in x/y around the cup-grasp pose taught on 2026-10-28 (tool at 0.282,
    # 0.162, 0.065; right fingertip 1.8 cm above the plane), z from 3 to 50 cm, with x/z
    # max raised so Kinova's home pose is inside (tool at 0.439, 0.193, 0.443 on the real
    # arm; 0.442, 0.192, 0.453 in Gazebo, which sags differently).
    workspace_min: list[float] | None = field(default_factory=lambda: [0.13, 0.01, 0.03])
    workspace_max: list[float] | None = field(default_factory=lambda: [0.45, 0.31, 0.50])
    urdf_path: str | None = None  # None: generated from kortex_description

    # Homing: on connect, drive slowly to home_position (every step through the guard),
    # gripper open. Default home is Kinova's own home pose, measured on the real arm after
    # homing it from the web app on 2026-10-28 (tool frame at 0.439, 0.193, 0.443).
    home_on_connect: bool = True
    home_position: list[float] | None = field(
        default_factory=lambda: [0.0, -0.282, 1.315, -0.003, -1.047, 0.0]
    )
    home_speed: float = 0.3  # rad/s, fastest joint
    home_tolerance: float = 0.02  # rad, max joint error to count as home
    home_timeout_sec: float = 30.0
