"""In-memory stand-in for ROS2Backend: a kinematic arm that reaches each joint target
instantly, plus optional synthetic cameras. For unit tests and for dry runs of the
HIL-SERL pipeline on a machine with no robot (``--robot.type=gen3_lite_hil_fake``).

It implements what BackendRobot / Gen3LiteRos / Gen3LiteHil use from a backend:
connect, disconnect, is_connected, joint_positions, joint_velocities, joint_state_age,
camera_frame_ages, get_observation, send_action.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lerobot.robots import RobotConfig

from .config_gen3_lite_hil import Gen3LiteHilConfig


class FakeArmBackend:
    def __init__(self, config, start_joints: list[float] | None = None, camera_renderer=None):
        self.config = config
        iface = config.ros2_interface
        self._arm = list(iface.arm_joint_names)
        self._gripper = iface.gripper_joint_name
        home = config.home_position or [0.0] * len(self._arm)
        self.joints = np.array(start_joints if start_joints is not None else home, dtype=float)
        self.gripper = float(iface.gripper_open_position)
        self._connected = False
        self._state_time: float | None = None
        self._frame_times: dict[str, float] = {}
        self._camera_renderer = camera_renderer
        self._cameras: dict[str, Any] = {}
        # Test hooks
        self.frozen = False  # True: joint state stops updating (driver stalled)
        self.camera_frozen = False
        self.sent: list[dict[str, float]] = []

    # lifecycle
    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True
        self._state_time = time.monotonic()

    def disconnect(self) -> None:
        self._connected = False

    # state
    def _touch(self) -> None:
        if not self.frozen:
            self._state_time = time.monotonic()

    @property
    def joint_state_age(self) -> float | None:
        self._touch()  # a live driver publishes continuously
        return None if self._state_time is None else time.monotonic() - self._state_time

    def joint_positions(self) -> dict[str, float]:
        out = {f"{j}.pos": float(v) for j, v in zip(self._arm, self.joints, strict=True)}
        if self._gripper:
            out[f"{self._gripper}.pos"] = self.gripper
        return out

    def joint_velocities(self) -> dict[str, float]:
        return {f"{j}.vel": 0.0 for j in self._arm}

    def camera_frame_ages(self) -> dict[str, float]:
        now = time.monotonic()
        return {k: now - t for k, t in self._frame_times.items()}

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = self.joint_positions()
        for cam, cam_cfg in self.config.cameras.items():
            if self._camera_renderer is not None:
                frame = self._camera_renderer(cam, self)
            else:
                frame = np.zeros((cam_cfg.height, cam_cfg.width, 3), dtype=np.uint8)
            if not self.camera_frozen:
                self._frame_times[cam] = time.monotonic()
            obs[cam] = frame
        return obs

    # action
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.sent.append({k: float(v) for k, v in action.items()})
        if not self.frozen:
            self.joints = np.array([float(action[f"{j}.pos"]) for j in self._arm])
            if "gripper.pos" in action:
                self.gripper = float(action["gripper.pos"])
        return action

    def reset(self) -> dict[str, Any]:
        return self.get_observation()

    @property
    def episode_success(self):
        return None

    @property
    def episode_reward(self) -> float:
        return 0.0


@RobotConfig.register_subclass("gen3_lite_hil_fake")
@dataclass
class Gen3LiteHilFakeConfig(Gen3LiteHilConfig):
    """Gen3LiteHil on FakeArmBackend: no ROS, no arm. Needs ``urdf_path``.

    The workspace guard is off by default because it needs the collision meshes;
    the end-effector limits, watchdog and staleness checks all still run.
    """

    guard_enabled: bool = False
    home_on_connect: bool = False
    # Tool position (base frame, m) the synthetic camera draws as the reach target.
    fake_target: list[float] = field(default_factory=lambda: [0.33, 0.12, 0.25])


def render_reach_scene(height: int, width: int, tool_xy, target_xy, box_min, box_max) -> np.ndarray:
    """Top-down picture of the tool box: green target disc, white tool disc."""
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    lo, hi = np.asarray(box_min[:2], float), np.asarray(box_max[:2], float)
    yy, xx = np.mgrid[0:height, 0:width]

    def disc(xy, radius_px, colour):
        u = (np.asarray(xy, float) - lo) / (hi - lo)
        cx, cy = u[0] * (width - 1), u[1] * (height - 1)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2] = colour

    disc(target_xy, max(2, width // 12), (0, 200, 0))
    disc(tool_xy, max(2, width // 20), (255, 255, 255))
    return img


class Gen3LiteHilFake:
    """Factory used by lerobot's ``make_robot_from_config`` for ``gen3_lite_hil_fake``."""

    def __new__(cls, config: Gen3LiteHilFakeConfig):
        from .ee_delta import ArmKinematics
        from .gen3_lite_hil import Gen3LiteHil

        if config.urdf_path is None:
            raise ValueError("gen3_lite_hil_fake needs robot.urdf_path (a Gen3 Lite URDF)")
        iface = config.ros2_interface
        kin = ArmKinematics(config.urdf_path, iface.arm_joint_names, iface.ee_link)

        def renderer(cam: str, backend: FakeArmBackend) -> np.ndarray:
            cam_cfg = config.cameras[cam]
            tool, _ = kin.fk(backend.joints)
            return render_reach_scene(
                cam_cfg.height, cam_cfg.width, tool[:2], config.fake_target[:2],
                config.ee_bounds_min, config.ee_bounds_max,
            )

        backend = FakeArmBackend(config, camera_renderer=renderer)
        robot = Gen3LiteHil(config, backend=backend, kinematics=kin)
        robot.name = "gen3_lite_hil_fake"
        return robot
