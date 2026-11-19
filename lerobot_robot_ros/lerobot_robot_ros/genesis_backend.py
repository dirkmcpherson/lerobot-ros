"""Genesis simulation of the Gen3 Lite with a cylinder and a shelf, as a backend for
``Gen3LiteHil`` (``--robot.type=gen3_lite_hil_genesis``).

The simulator runs in its own thread in real time (dt 0.01 s), like a driver: commands
are non-blocking, the arm tracks them with lag, joint state and camera frames are
snapshots with ages. Each arm command is interpolated from the previous goal over
``action_time_s`` (the real ``joint_trajectory_controller`` gets a one-point goal due
0.2 s out). The arm has gravity compensation and PD position control; the fingers are
four independent joints in Genesis, driven consistently through the URDF's mimic
relations.

The policy only sees what the robot class reports: joints, gripper, tool position and the
camera. The cylinder and shelf state is available to scripted operators through
``task_state()`` (privileged; never part of the observation).
"""

from __future__ import annotations

import logging
import pathlib
import tempfile
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ASSETS = pathlib.Path(__file__).parent / "assets"
FINGERS = ["right_finger_bottom_joint", "left_finger_bottom_joint", "right_finger_tip_joint", "left_finger_tip_joint"]
# Driver joint (right_finger_bottom) angles in the Genesis URDF. Measured in the scene:
# 0.96 leaves 10.5 cm between the fingertips, -0.09 closes them fully.
THETA_OPEN, THETA_CLOSED = 0.96, -0.09


def kortex_description_dir() -> pathlib.Path:
    import os

    candidates = []
    if os.environ.get("KORTEX_DESCRIPTION_PARENT"):
        candidates.append(pathlib.Path(os.environ["KORTEX_DESCRIPTION_PARENT"]) / "kortex_description")
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(pathlib.Path(get_package_share_directory("kortex_description")))
    except Exception:
        pass
    candidates.append(pathlib.Path.home() / "workspace/ros2_kortex_ws/src/ros2_kortex/kortex_description")
    for c in candidates:
        if (c / "arms/gen3_lite/6dof/meshes").is_dir():
            return c
    raise FileNotFoundError("kortex_description meshes not found; set KORTEX_DESCRIPTION_PARENT")


def resolved_urdf(urdf: pathlib.Path) -> pathlib.Path:
    """The URDF with ``package://kortex_description/`` replaced by the installed path."""
    text = urdf.read_text().replace("package://kortex_description/", str(kortex_description_dir()) + "/")
    out = pathlib.Path(tempfile.mkdtemp(prefix="gen3_lite_genesis_")) / urdf.name
    out.write_text(text)
    return out


def gripper_to_theta(g: float, open_pos: float, close_pos: float) -> float:
    """Robot-class gripper value (open_pos .. close_pos) -> Genesis driver joint angle."""
    frac = float(np.clip((g - open_pos) / (close_pos - open_pos), 0.0, 1.0))
    return THETA_OPEN + frac * (THETA_CLOSED - THETA_OPEN)


def theta_to_gripper(theta: float, open_pos: float, close_pos: float) -> float:
    frac = (float(theta) - THETA_OPEN) / (THETA_CLOSED - THETA_OPEN)
    return open_pos + frac * (close_pos - open_pos)


def finger_targets(theta: float) -> list[float]:
    """Targets for FINGERS from the driver angle (the URDF's mimic multipliers/offsets)."""
    tip = -0.676 * theta + 0.149
    return [theta, -theta, tip, tip]


def fingertip_center(finger_verts, band: float = 0.01) -> np.ndarray:
    """Midpoint between the two fingers' tips: the mean of each finger's vertices within
    ``band`` of its lowest point (top-down grasp), averaged over the fingers."""
    tips = []
    for v in finger_verts:
        v = np.asarray(v, float).reshape(-1, 3)
        tips.append(v[v[:, 2] <= v[:, 2].min() + band].mean(axis=0))
    return np.mean(tips, axis=0)


def interpolate(start: np.ndarray, goal: np.ndarray, t0: float, duration: float, now: float) -> np.ndarray:
    if duration <= 0:
        return goal
    s = float(np.clip((now - t0) / duration, 0.0, 1.0))
    return start + (goal - start) * s


def on_shelf(obj_pos, obj_up_z: float, shelf_pos, shelf_size, obj_height: float, z_tol: float = 0.015,
             xy_margin: float = 0.0, min_upright: float = 0.9) -> bool:
    """Object centre over the shelf top (inside it by ``xy_margin``), resting on it, upright."""
    obj_pos, shelf_pos, shelf_size = (np.asarray(v, float) for v in (obj_pos, shelf_pos, shelf_size))
    half = shelf_size[:2] / 2 - xy_margin
    inside = bool(np.all(np.abs(obj_pos[:2] - shelf_pos[:2]) <= half))
    top = shelf_pos[2] + shelf_size[2] / 2
    resting = abs(obj_pos[2] - (top + obj_height / 2)) <= z_tol
    return inside and resting and obj_up_z >= min_upright


class GenesisArmBackend:
    """What Gen3LiteHil uses from a backend (see FakeArmBackend), on a Genesis scene."""

    def __init__(self, config):
        self.config = config
        iface = config.ros2_interface
        self._arm = list(iface.arm_joint_names)
        self._gripper_name = iface.gripper_joint_name
        self._g_open, self._g_close = float(iface.gripper_open_position), float(iface.gripper_close_position)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._connected = False
        self._rng = np.random.default_rng(config.genesis_seed)
        # command state (written by send_action, read by the sim thread)
        self._goal: np.ndarray | None = None
        self._start: np.ndarray | None = None
        self._t0 = 0.0
        self._grip_theta = THETA_OPEN
        self._reset_request = False
        # snapshots (written by the sim thread)
        self._q = np.zeros(len(self._arm))
        self._dq = np.zeros(len(self._arm))
        self._theta = THETA_OPEN
        self._state_time: float | None = None
        self._frames: dict[str, np.ndarray] = {}
        self._frame_times: dict[str, float] = {}
        self._obj_pos = np.zeros(3)
        self._obj_up_z = 1.0
        self._finger_center = np.zeros(3)
        self.sim_steps = 0
        self.realtime_lag_s = 0.0  # how far the simulation runs behind the wall clock (max seen)
        self.late_steps = 0  # physics steps that finished more than 50 ms behind the wall clock

    # ------------------------------------------------------------------ lifecycle
    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="genesis_sim", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=600):
            raise TimeoutError("Genesis scene did not build within 600 s")
        if self._error is not None:
            raise RuntimeError(f"Genesis scene failed to build: {self._error}") from self._error
        self._connected = True

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._connected = False

    # ------------------------------------------------------------------ simulation thread
    def _build(self):
        import genesis as gs

        cfg = self.config
        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.gpu if cfg.genesis_device == "gpu" else gs.cpu, precision="32",
                    logging_level="warning", seed=cfg.genesis_seed)
        scene = gs.Scene(show_viewer=cfg.genesis_viewer,
                         sim_options=gs.options.SimOptions(dt=0.01, substeps=cfg.genesis_substeps))
        scene.add_entity(gs.morphs.Plane())
        urdf = resolved_urdf(pathlib.Path(cfg.genesis_urdf) if cfg.genesis_urdf else ASSETS / "gen3_lite_2f_genesis.urdf")
        arm = scene.add_entity(gs.morphs.URDF(file=str(urdf), fixed=True, pos=(0.0, 0.0, 0.0)),
                               material=gs.materials.Rigid(gravity_compensation=1.0))
        obj = scene.add_entity(
            gs.morphs.Cylinder(pos=tuple(cfg.object_start), radius=cfg.object_radius, height=cfg.object_height),
            material=gs.materials.Rigid(rho=cfg.object_density, friction=cfg.object_friction),
            surface=gs.surfaces.Default(color=(0.9, 0.35, 0.1)),
        )
        scene.add_entity(gs.morphs.Box(pos=tuple(cfg.shelf_position), size=tuple(cfg.shelf_size), fixed=True),
                         surface=gs.surfaces.Default(color=(0.35, 0.45, 0.85)))
        cams = {}
        for name, cam_cfg in cfg.cameras.items():
            pose = getattr(cfg, "genesis_camera_poses", {}).get(name, {})
            cams[name] = scene.add_camera(
                res=(cam_cfg.width, cam_cfg.height), pos=tuple(pose.get("pos", (0.95, 0.17, 0.35))),
                lookat=tuple(pose.get("lookat", (0.30, 0.17, 0.10))), fov=float(pose.get("fov", 55)), GUI=False,
            )
        scene.build()
        arm_idx = [arm.get_joint(n).dofs_idx_local[0] for n in self._arm]
        fing_idx = [arm.get_joint(n).dofs_idx_local[0] for n in FINGERS]
        idx = arm_idx + fing_idx
        arm.set_dofs_kp(np.array(cfg.genesis_arm_kp + [cfg.genesis_finger_kp] * 4, float), idx)
        arm.set_dofs_kv(np.array(cfg.genesis_arm_kv + [cfg.genesis_finger_kv] * 4, float), idx)
        force = np.array(cfg.genesis_arm_force + [cfg.genesis_finger_force] * 4, float)
        arm.set_dofs_force_range(-force, force, idx)
        start = np.asarray(cfg.reset_joint_positions or cfg.home_position, float)
        q = np.concatenate([start, finger_targets(THETA_OPEN)])
        arm.set_dofs_position(q, idx)
        arm.control_dofs_position(q, idx)
        self._goal = self._start = start.copy()
        links = [arm.get_link(n) for n in ("right_finger_dist_link", "left_finger_dist_link")]
        # the URDF's fixed tool_frame is merged into end_effector_link by Genesis
        return scene, arm, obj, cams, arm_idx, fing_idx, links

    def _reset_object(self, obj):
        cfg = self.config
        jitter = np.asarray(cfg.object_start_jitter, float)
        pos = np.asarray(cfg.object_start, float) + self._rng.uniform(-jitter, jitter)
        obj.set_pos(pos)
        obj.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
        obj.zero_all_dofs_velocity()

    def _run(self):
        try:
            scene, arm, obj, cams, arm_idx, fing_idx, links = self._build()
        except BaseException as e:  # reported by connect()
            self._error = e
            self._ready.set()
            return
        dt, render_every = 0.01, max(1, int(round(self.config.genesis_render_period_s / 0.01)))
        wall0 = time.monotonic()
        k = 0
        first = True
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                with self._lock:
                    if self._reset_request:
                        self._reset_object(obj)
                        self._reset_request = False
                    target = interpolate(self._start, self._goal, self._t0, self.config.action_time_s, now)
                    theta = self._grip_theta
                arm.control_dofs_position(np.asarray(target, float), arm_idx)
                arm.control_dofs_position(np.asarray(finger_targets(theta), float), fing_idx)
                scene.step()
                k += 1
                q = arm.get_dofs_position(arm_idx).cpu().numpy().astype(float)
                dq = arm.get_dofs_velocity(arm_idx).cpu().numpy().astype(float)
                th = float(arm.get_dofs_position([fing_idx[0]]).cpu().numpy()[0])
                frames = {}
                if first or k % render_every == 0:
                    for name, cam in cams.items():
                        frames[name] = np.ascontiguousarray(np.asarray(cam.render(rgb=True)[0], dtype=np.uint8))
                pos = obj.get_pos().cpu().numpy().astype(float)
                quat = obj.get_quat().cpu().numpy().astype(float)  # w, x, y, z
                w, x, y, z = quat
                up_z = 1.0 - 2.0 * (x * x + y * y)  # z component of the object's axis
                fc = fingertip_center([lk.get_verts().cpu().numpy() for lk in links])
                t_now = time.monotonic()
                with self._lock:
                    self._q, self._dq, self._theta, self._state_time = q, dq, th, t_now
                    for name, f in frames.items():
                        self._frames[name], self._frame_times[name] = f, t_now
                    self._obj_pos, self._obj_up_z, self._finger_center = pos, float(up_z), fc
                    self.sim_steps = k
                if first:
                    first = False
                    wall0 = time.monotonic()
                    k = 0
                    self._ready.set()
                    continue
                # real time: sim time k*dt should track the wall clock
                ahead = wall0 + k * dt - time.monotonic()
                if ahead > 0:
                    time.sleep(ahead)
                else:
                    self.realtime_lag_s = max(self.realtime_lag_s, -ahead)
                    self.late_steps += -ahead > 0.05
                    if -ahead > 0.5:  # do not try to catch up after a stall
                        wall0 = time.monotonic() - k * dt
        except BaseException as e:
            self._error = e
            logger.exception("Genesis simulation thread stopped")
        finally:
            self._ready.set()
            try:  # free the GPU state in the thread that owns it (exit crashed otherwise)
                import genesis as gs

                gs.destroy()
            except Exception:
                pass

    # ------------------------------------------------------------------ state
    def _alive(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Genesis simulation stopped: {self._error}")

    @property
    def joint_state_age(self) -> float | None:
        with self._lock:
            return None if self._state_time is None or self._error is not None else time.monotonic() - self._state_time

    def joint_positions(self) -> dict[str, float]:
        self._alive()
        with self._lock:
            out = {f"{j}.pos": float(v) for j, v in zip(self._arm, self._q, strict=True)}
            if self._gripper_name:
                out[f"{self._gripper_name}.pos"] = theta_to_gripper(self._theta, self._g_open, self._g_close)
        return out

    def joint_velocities(self) -> dict[str, float]:
        with self._lock:
            return {f"{j}.vel": float(v) for j, v in zip(self._arm, self._dq, strict=True)}

    def camera_frame_ages(self) -> dict[str, float]:
        now = time.monotonic()
        with self._lock:
            return {k: now - t for k, t in self._frame_times.items()}

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = self.joint_positions()
        with self._lock:
            for cam in self.config.cameras:
                obs[cam] = self._frames[cam].copy()
        return obs

    # ------------------------------------------------------------------ commands
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self._alive()
        goal = np.array([float(action[f"{j}.pos"]) for j in self._arm])
        now = time.monotonic()
        with self._lock:
            current = interpolate(self._start, self._goal, self._t0, self.config.action_time_s, now)
            self._start, self._goal, self._t0 = current, goal, now
            if "gripper.pos" in action:
                self._grip_theta = gripper_to_theta(float(action["gripper.pos"]), self._g_open, self._g_close)
        return action

    def reset(self) -> dict[str, Any]:
        self.reset_task()
        return self.get_observation()

    # ------------------------------------------------------------------ task (privileged)
    def reset_task(self, wait: bool = True) -> None:
        """Put the cylinder back at its start (with jitter); applied by the sim thread."""
        with self._lock:
            self._reset_request = True
        if wait:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with self._lock:
                    if not self._reset_request:
                        break
                time.sleep(0.01)
            time.sleep(0.3)  # let it settle

    def task_state(self) -> dict[str, Any]:
        cfg = self.config
        with self._lock:
            pos, up, fc = self._obj_pos.copy(), self._obj_up_z, self._finger_center.copy()
        return {
            "object_position": pos,
            "object_upright": up,
            "finger_center": fc,
            "shelf_position": np.asarray(cfg.shelf_position, float),
            "shelf_size": np.asarray(cfg.shelf_size, float),
            "object_height": cfg.object_height,
            "object_radius": cfg.object_radius,
            "on_shelf": on_shelf(pos, up, cfg.shelf_position, cfg.shelf_size, cfg.object_height),
        }

    @property
    def episode_success(self) -> bool | None:
        return None  # success comes from the operator's key (here: the scripted operator)

    @property
    def episode_reward(self) -> float:
        return 0.0
