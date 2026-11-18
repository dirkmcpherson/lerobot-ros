"""lerobot RobotConfig for HIL-SERL on the Kinova Gen3 Lite (``--robot.type=gen3_lite_hil``).

Same ROS 2 path as ``gen3_lite_ros`` (joint targets to joint_trajectory_controller, the
workspace guard, guarded homing), with an end-effector delta action space on top.
"""

from dataclasses import dataclass, field

from lerobot.robots import RobotConfig

from .config_gen3_lite_ros import Gen3LiteRosConfig


@RobotConfig.register_subclass("gen3_lite_hil")
@dataclass
class Gen3LiteHilConfig(Gen3LiteRosConfig):
    """Kinova Gen3 Lite driven by end-effector position deltas.

    Actions: ``delta_x``, ``delta_y``, ``delta_z`` in [-1, 1] (one unit is
    ``ee_step_size_m`` of tool travel in the base frame) and, when ``use_gripper``,
    ``gripper`` (0 close, 1 stay, 2 open: lerobot's gamepad convention). The tool
    orientation is held at its value when control (re)starts.

    Observations: ``joint_1.pos`` .. ``joint_6.pos`` (rad), ``gripper.pos`` (measured
    finger joint, 0.0 open .. 0.85 closed), ``ee.x`` / ``ee.y`` / ``ee.z`` (tool frame in
    the base frame, m), optionally joint velocities, and one entry per camera.
    """

    # --- Action space ---
    use_gripper: bool = False
    # Gripper joint position held when use_gripper is False. None: leave the gripper alone.
    fixed_gripper_position: float | None = None

    # Joint pose (rad) each episode starts from; the environment drives there with the
    # guarded homing move on reset. None: no move on reset (unless the env config gives
    # one). Keep its tool position inside ee_bounds.
    reset_joint_positions: list[float] | None = None

    # --- Limits enforced on every command (see ee_delta.limit_step) ---
    control_period_s: float = 0.1  # 10 Hz
    ee_step_size_m: float = 0.01  # tool travel per unit of action, per axis
    max_ee_step_m: float = 0.01  # cap on the length of one step
    max_ee_speed_m_s: float = 0.1
    max_ee_lead_m: float = 0.03  # target may not lead the measured tool position by more
    # Box the commanded tool position is clipped to (base frame, m). Kept inside the
    # workspace guard's box (workspace_min/max), which stays on as the hard backstop
    # together with the table plane and self-collision checks. Set per task.
    ee_bounds_min: list[float] = field(default_factory=lambda: [0.15, 0.03, 0.04])
    ee_bounds_max: list[float] = field(default_factory=lambda: [0.43, 0.29, 0.40])
    # A command whose IK solution misses the target by more than this is not sent.
    ik_position_tolerance_m: float = 0.004
    ik_orientation_tolerance_rad: float = 0.1

    # --- Watchdog and staleness ---
    # If no action arrives for this many control periods after the last one, the arm is
    # commanded to hold its measured pose (the zero command for a position interface).
    watchdog_periods: float = 3.0
    # Observations older than this raise StaleObservationError instead of being returned,
    # and no command is sent on stale joint state.
    max_obs_age_s: float = 0.2
    # A gap this long between actions restarts the target from the measured pose.
    resync_after_s: float = 0.5

    # --- Observation extras ---
    # The Gen3 Lite driver has reported zero joint velocity (see the streaming controllers
    # yaml), so this is off until the operator confirms real values on /joint_states.
    observe_joint_velocities: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.ee_limits().validate()
        if self.workspace_min is not None and self.workspace_max is not None:
            lo_ok = all(a >= b for a, b in zip(self.ee_bounds_min, self.workspace_min, strict=True))
            hi_ok = all(a <= b for a, b in zip(self.ee_bounds_max, self.workspace_max, strict=True))
            if not (lo_ok and hi_ok):
                raise ValueError(
                    f"ee_bounds {self.ee_bounds_min}..{self.ee_bounds_max} must lie inside the workspace "
                    f"guard box {self.workspace_min}..{self.workspace_max}"
                )
        if self.watchdog_periods < 1:
            raise ValueError("watchdog_periods must be at least 1")

    def ee_limits(self):
        from .ee_delta import EeLimits

        return EeLimits(
            step_size_m=self.ee_step_size_m,
            max_step_m=self.max_ee_step_m,
            max_speed_m_s=self.max_ee_speed_m_s,
            control_period_s=self.control_period_s,
            max_lead_m=self.max_ee_lead_m,
            box_min=list(self.ee_bounds_min),
            box_max=list(self.ee_bounds_max),
        )

    def limits_summary(self) -> str:
        return (
            f"  tool step per action unit : {self.ee_step_size_m * 100:.1f} cm\n"
            f"  max step length           : {self.max_ee_step_m * 100:.1f} cm per command\n"
            f"  max tool speed            : {self.max_ee_speed_m_s:.3f} m/s\n"
            f"  tool box (base frame, m)  : {self.ee_bounds_min} .. {self.ee_bounds_max}\n"
            f"  guard box (hard backstop) : {self.workspace_min} .. {self.workspace_max}\n"
            f"  table plane               : z={self.plane_z} m, margin {self.plane_margin} m"
            f" (guard {'on' if self.guard_enabled else 'OFF'})\n"
            f"  per-joint step cap        : {self.max_relative_target} rad per command\n"
            f"  watchdog                  : hold after {self.watchdog_periods * self.control_period_s:.2f} s"
            " without a command\n"
            f"  stale observation limit   : {self.max_obs_age_s:.2f} s\n"
            f"  home on connect           : {self.home_on_connect} ({self.home_position}, {self.home_speed} rad/s)\n"
            f"  episode reset pose        : {self.reset_joint_positions}"
        )
