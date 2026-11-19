from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("scripted_pick_place")
@dataclass
class ScriptedPickPlaceTeleopConfig(TeleoperatorConfig):
    """A stand-in operator for ``gen3_lite_hil_genesis``: a waypoint motion planner that
    picks the cylinder and places it on the shelf, using the simulator's privileged state
    (object pose, finger positions). Same actions and events as lerobot's gamepad teleop.

    Gaussian noise is added to every step's x, y, z command (``noise_std``, in action
    units, 1 unit = ``ee_step_size_m``) so the demonstrations are not perfectly smooth; the
    planner corrects for it on the next step. Success (the operator's success key) is
    reported when the cylinder rests upright on the shelf with the gripper open.
    """

    takeovers: list[list[int]] = field(default_factory=lambda: [[20, 80]])
    max_takeover_episodes: int | None = None
    always_intervene: bool = False
    episode_steps: int = 149
    ee_step_size_m: float = 0.01
    noise_std: float = 0.3
    seed: int = 0
    # Finger-centre height above the object's centre when grasping (m), the approach and
    # carry clearances, and the tolerances that switch phases.
    grasp_dz: float = 0.0
    approach_height: float = 0.08
    carry_clearance: float = 0.05
    place_drop: float = 0.005
    xy_tolerance: float = 0.01
    z_tolerance: float = 0.008
    # Gripper value (0 open .. 0.85 closed) above which it counts as closed on the object,
    # below which as open; steps to wait for a close before retrying.
    gripper_closed_above: float = 0.15
    gripper_open_below: float = 0.10
    close_wait_steps: int = 20
