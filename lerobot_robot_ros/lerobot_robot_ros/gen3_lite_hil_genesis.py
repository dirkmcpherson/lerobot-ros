"""``--robot.type=gen3_lite_hil_genesis``: Gen3LiteHil (same limits, IK, guard, watchdog)
on a Genesis scene with a cylinder to pick up and a shelf to place it on.

The policy's observation is what the real robot reports: joints, gripper, tool position
and the head-on camera. The cylinder's pose is privileged; only scripted operators read
it, through ``PICK_PLACE_SOURCE`` (``lerobot_teleoperator_devices.scripted_pick_place``).
Each episode reset (``go_home`` to the reset pose) also puts the cylinder back.
"""

from dataclasses import dataclass, field

from lerobot.robots import RobotConfig

from .config_gen3_lite_hil import Gen3LiteHilConfig

# Top-down grasp: tool at (0.30, 0.15, 0.20) m, approach axis pointing straight down.
TOP_DOWN_RESET = [0.323, -0.356, 1.508, 1.571, 1.278, -0.201]


@RobotConfig.register_subclass("gen3_lite_hil_genesis")
@dataclass
class Gen3LiteHilGenesisConfig(Gen3LiteHilConfig):
    use_sim_time: bool = False  # the simulator runs on the wall clock
    home_on_connect: bool = False  # the scene starts at the reset pose
    reset_joint_positions: list[float] | None = field(default_factory=lambda: list(TOP_DOWN_RESET))
    use_gripper: bool = True
    # Seconds over which each arm goal is interpolated (the real driver's goal is due 0.2 s out).
    action_time_s: float = 0.2

    genesis_device: str = "cpu"  # cpu | gpu. One env: cpu holds real time (gpu ran at 0.87x, its per-step reads sync the device)
    genesis_viewer: bool = False
    genesis_substeps: int = 4
    genesis_seed: int = 0
    genesis_render_period_s: float = 0.05  # camera frames at 20 Hz, so a 10 Hz reader never sees one older than ~0.1 s
    genesis_urdf: str | None = None  # None: assets/gen3_lite_2f_genesis.urdf (genesis_pickaplace's)
    genesis_arm_kp: list[float] = field(default_factory=lambda: [200.0, 200.0, 150.0, 100.0, 60.0, 60.0])
    genesis_arm_kv: list[float] = field(default_factory=lambda: [20.0, 20.0, 15.0, 10.0, 6.0, 6.0])
    genesis_arm_force: list[float] = field(default_factory=lambda: [87.0, 87.0, 52.0, 30.0, 14.0, 14.0])
    genesis_finger_kp: float = 40.0
    genesis_finger_kv: float = 10.0
    genesis_finger_force: float = 50.0
    # Camera pose (all cameras in ``cameras`` use it; one head-on camera is the plan).
    camera_pos: list[float] = field(default_factory=lambda: [0.95, 0.16, 0.35])
    camera_lookat: list[float] = field(default_factory=lambda: [0.30, 0.16, 0.10])
    camera_fov: float = 55.0

    # The task (base frame, metres). The base sits on the ground plane, which is the table.
    object_start: list[float] = field(default_factory=lambda: [0.32, 0.08, 0.05])
    object_start_jitter: list[float] = field(default_factory=lambda: [0.01, 0.01, 0.0])
    object_radius: float = 0.03
    object_height: float = 0.10
    object_density: float = 300.0
    object_friction: float = 1.0
    shelf_position: list[float] = field(default_factory=lambda: [0.32, 0.24, 0.04])
    shelf_size: list[float] = field(default_factory=lambda: [0.12, 0.10, 0.08])

    @property
    def genesis_camera_poses(self) -> dict:
        pose = {"pos": self.camera_pos, "lookat": self.camera_lookat, "fov": self.camera_fov}
        return {name: pose for name in self.cameras}


class Gen3LiteHilGenesis:
    """Factory used by lerobot's ``make_robot_from_config`` for ``gen3_lite_hil_genesis``."""

    def __new__(cls, config: Gen3LiteHilGenesisConfig):
        from .genesis_backend import GenesisArmBackend
        from .gen3_lite_hil import Gen3LiteHil

        class _Robot(Gen3LiteHil):
            name = "gen3_lite_hil_genesis"

            def connect(self, calibrate: bool = True) -> None:
                super().connect(calibrate)
                try:
                    from lerobot_teleoperator_devices.scripted_pick_place import PICK_PLACE_SOURCE
                except ImportError:
                    return
                PICK_PLACE_SOURCE["task"] = self.backend.task_state

            def go_home(self, position=None) -> None:
                super().go_home(position)  # opens the gripper on the way
                self.backend.reset_task()

        return _Robot(config, backend=GenesisArmBackend(config))
