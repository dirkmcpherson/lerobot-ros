"""lerobot Robot for HIL-SERL on the Kinova Gen3 Lite (see ``config_gen3_lite_hil``).

Command path: end-effector delta -> limits (``ee_delta.limit_step``) -> inverse
kinematics (pinocchio, orientation held) -> absolute joint targets ->
``Gen3LiteRos.send_joint_action`` (per-joint step cap, workspace guard) ->
FollowJointTrajectory goal. Nothing on that path waits on a service or sleeps.

Safety in this class:
  * every command is clipped to a tool box, a step length and a speed (``limit_step``);
  * a watchdog thread commands a hold when actions stop arriving;
  * stale joint state blocks commands and raises, stale camera frames raise;
  * disconnect commands a hold first.
The workspace guard of the parent class (table plane, guard box, self-collision) stays
active underneath as the backstop.
"""

import logging
import threading
import time
from typing import Any

import numpy as np

from .config_gen3_lite_hil import Gen3LiteHilConfig
from .ee_delta import ArmKinematics, StepReport, limit_step
from .gen3_lite_ros import Gen3LiteRos

logger = logging.getLogger(__name__)

GRIPPER_CLOSE, GRIPPER_STAY, GRIPPER_OPEN = 0, 1, 2
EE_KEYS = ("ee.x", "ee.y", "ee.z")
DELTA_KEYS = ("delta_x", "delta_y", "delta_z")


class StaleObservationError(RuntimeError):
    """An observation is older than ``max_obs_age_s``."""


class Gen3LiteHil(Gen3LiteRos):
    config_class = Gen3LiteHilConfig
    name = "gen3_lite_hil"

    def __init__(self, config: Gen3LiteHilConfig, backend=None, kinematics: ArmKinematics | None = None):
        super().__init__(config, backend=backend)
        self.config = config
        self._limits = config.ee_limits()
        self._kin = kinematics
        self._cmd_lock = threading.RLock()

        # Reference the next delta is added to; None means "restart from the measured pose".
        self._ref_pos: np.ndarray | None = None
        self._ref_rot: np.ndarray | None = None
        self._ref_joints: np.ndarray | None = None
        self._gripper_target: float | None = None
        self._last_cmd_time: float | None = None

        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._watchdog_armed = False

        # Counters, readable by tests, the smoke test and the run logs.
        self.watchdog_trips = 0
        self.clipped_commands = 0
        self.ik_rejections = 0
        self.stale_stops = 0
        self.last_report: StepReport | None = None

    # ------------------------------------------------------------------ features

    @property
    def _arm_keys(self) -> list[str]:
        return [f"{j}.pos" for j in self.config.ros2_interface.arm_joint_names]

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        iface = self.config.ros2_interface
        features: dict[str, type | tuple] = dict.fromkeys(self._arm_keys, float)
        if self.config.observe_joint_velocities:
            features.update({f"{j}.vel": float for j in iface.arm_joint_names})
        if iface.gripper_joint_name:
            features["gripper.pos"] = float
        features.update(dict.fromkeys(EE_KEYS, float))
        for cam_name, cam_cfg in self.config.cameras.items():
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)
        return features

    @property
    def state_keys(self) -> list[str]:
        """Observation keys that are scalars, in feature order."""
        return [k for k, v in self.observation_features.items() if v is float]

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = dict.fromkeys(DELTA_KEYS, float)
        if self.config.use_gripper:
            features["gripper"] = float
        return features

    # ----------------------------------------------------------------- lifecycle

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            logger.info(f"{self} already connected.")
            return
        logger.info("Gen3 Lite HIL limits:\n" + self.config.limits_summary())
        if self._kin is None:
            from lerobot_backends.ros2.workspace_guard import gen3_lite_urdf

            self._kin = ArmKinematics(
                self.config.urdf_path or gen3_lite_urdf(),
                self.config.ros2_interface.arm_joint_names,
                self.config.ros2_interface.ee_link,
            )
        self._invalidate_reference()
        try:
            super().connect(calibrate)  # guard setup, backend connect, optional homing
        except Exception:
            if self.backend.is_connected:
                self.backend.disconnect()
            raise
        self._publish_tool_position()
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, name="gen3_lite_watchdog", daemon=True)
        self._watchdog_thread.start()

    def _publish_tool_position(self) -> None:
        """Let the scripted operator (``--teleop.type=scripted_ee``: tests, dry runs, Gazebo)
        read the measured tool position; lerobot gives a teleoperator no robot handle."""
        try:
            from lerobot_teleoperator_devices.scripted_ee import TOOL_POSITION_SOURCE
        except ImportError:
            return
        TOOL_POSITION_SOURCE["get"] = lambda: self._kin.fk(self._measured_arm()[0])[0]

    def disconnect(self) -> None:
        if self._watchdog_thread is not None:
            self._watchdog_stop.set()
            self._watchdog_thread.join(timeout=2.0)
            self._watchdog_thread = None
        if not self.is_connected:
            return
        try:
            self.hold("disconnect")
        except Exception as e:  # still release the backend
            logger.error(f"Could not command a hold before disconnecting: {e}")
        super().disconnect()

    def go_home(self, position: list[float] | None = None) -> None:
        with self._cmd_lock:
            self._watchdog_armed = False
            self._invalidate_reference()
        super().go_home(position)
        with self._cmd_lock:
            self._invalidate_reference()

    # --------------------------------------------------------------- observation

    def _check_joint_state_fresh(self) -> None:
        age = self.backend.joint_state_age
        if age is None or age > self.config.max_obs_age_s:
            self.stale_stops += 1
            shown = "never received" if age is None else f"{age * 1000:.0f} ms old"
            raise StaleObservationError(
                f"Joint state is {shown} (limit {self.config.max_obs_age_s * 1000:.0f} ms). "
                "No command was sent; the position controller holds its last goal."
            )

    def _measured_arm(self) -> tuple[np.ndarray, float]:
        arm, grip = self._measured()
        return np.asarray(arm, dtype=float), float(grip)

    def observation_ages(self) -> dict[str, float | None]:
        """Seconds since each observation source was last updated."""
        ages: dict[str, float | None] = {"joint_state": self.backend.joint_state_age}
        cam_ages = self.backend.camera_frame_ages()
        for cam in self.config.cameras:
            ages[cam] = cam_ages.get(cam)
        return ages

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            from lerobot.utils.errors import DeviceNotConnectedError

            raise DeviceNotConnectedError(f"{self} is not connected.")
        iface = self.config.ros2_interface
        self._check_joint_state_fresh()
        raw = self.backend.get_observation()  # joints + cameras
        self._check_joint_state_fresh()  # joint state may have gone stale during camera reads

        obs: dict[str, Any] = {k: float(raw[k]) for k in self._arm_keys}
        if self.config.observe_joint_velocities:
            vel = self.backend.joint_velocities()
            obs.update({f"{j}.vel": float(vel[f"{j}.vel"]) for j in iface.arm_joint_names})
        if iface.gripper_joint_name:
            obs["gripper.pos"] = float(raw[f"{iface.gripper_joint_name}.pos"])
        pos, _ = self._kin.fk([obs[k] for k in self._arm_keys])
        obs.update(dict(zip(EE_KEYS, pos.tolist(), strict=True)))

        if self.config.cameras:
            ages = self.backend.camera_frame_ages()
            for cam in self.config.cameras:
                age = ages.get(cam)
                if age is None or age > self.config.max_obs_age_s:
                    self.stale_stops += 1
                    shown = "never read" if age is None else f"{age * 1000:.0f} ms old"
                    raise StaleObservationError(
                        f"Camera '{cam}' frame is {shown} (limit {self.config.max_obs_age_s * 1000:.0f} ms)."
                    )
                obs[cam] = raw[cam]
        return obs

    # -------------------------------------------------------------------- action

    def _invalidate_reference(self) -> None:
        self._ref_pos = self._ref_rot = self._ref_joints = None
        self._gripper_target = None

    def _gripper_command(self, action: dict[str, Any], measured: float) -> float | None:
        cfg, iface = self.config, self.config.ros2_interface
        if not iface.gripper_joint_name:
            return None
        if not cfg.use_gripper:
            return cfg.fixed_gripper_position
        if self._gripper_target is None:
            self._gripper_target = measured
        g = action.get("gripper", GRIPPER_STAY)
        g = GRIPPER_STAY if g is None or not np.isfinite(float(g)) else int(round(float(g)))
        if g <= GRIPPER_CLOSE:
            self._gripper_target = iface.gripper_close_position
        elif g >= GRIPPER_OPEN:
            self._gripper_target = iface.gripper_open_position
        return self._gripper_target

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute one end-effector delta. Returns the action actually applied, in the
        same normalised units (smaller than requested where a limit acted)."""
        if not self.is_connected:
            from lerobot.utils.errors import DeviceNotConnectedError

            raise DeviceNotConnectedError(f"{self} is not connected.")
        cfg = self.config
        with self._cmd_lock:
            self._check_joint_state_fresh()
            now = time.monotonic()
            arm, grip_meas = self._measured_arm()
            measured_pos, measured_rot = self._kin.fk(arm)

            dt = None if self._last_cmd_time is None else now - self._last_cmd_time
            if self._ref_pos is None or dt is None or dt > cfg.resync_after_s:
                self._ref_pos, self._ref_rot, self._ref_joints = measured_pos, measured_rot, arm
                self._gripper_target = None
                dt = None

            target_pos, report = limit_step(
                [action.get(k, 0.0) for k in DELTA_KEYS], self._ref_pos, measured_pos, self._limits, dt
            )
            self.last_report = report
            if report.any_clipped:
                self.clipped_commands += 1

            joints, pos_err, rot_err = self._kin.ik(
                self._ref_joints,
                target_pos,
                self._ref_rot,
                lower=cfg.ros2_interface.min_joint_positions,
                upper=cfg.ros2_interface.max_joint_positions,
            )
            if pos_err > cfg.ik_position_tolerance_m or rot_err > cfg.ik_orientation_tolerance_rad:
                # Unreachable with the held orientation: do not move the arm for this step.
                self.ik_rejections += 1
                logger.warning(
                    f"IK rejected a step (position error {pos_err * 1000:.1f} mm, "
                    f"orientation error {rot_err:.3f} rad); holding."
                )
                joints, target_pos = self._ref_joints, self._ref_pos

            joint_action: dict[str, Any] = dict(zip(self._arm_keys, np.asarray(joints).tolist(), strict=True))
            gripper = self._gripper_command(action, grip_meas)
            if gripper is not None:
                joint_action["gripper.pos"] = gripper

            sent = self.send_joint_action(joint_action)  # step cap + workspace guard + ROS
            sent_joints = np.array([float(sent[k]) for k in self._arm_keys])
            sent_pos, _ = self._kin.fk(sent_joints)
            applied = sent_pos - self._ref_pos
            self._ref_pos, self._ref_joints = sent_pos, sent_joints
            self._last_cmd_time = now
            self._watchdog_armed = True

        out: dict[str, Any] = dict(zip(DELTA_KEYS, (applied / cfg.ee_step_size_m).tolist(), strict=True))
        if cfg.use_gripper:
            out["gripper"] = float(action.get("gripper", GRIPPER_STAY))
        return out

    def hold(self, reason: str = "hold") -> None:
        """Command the arm to stay at its measured pose and restart the delta reference."""
        with self._cmd_lock:
            self._watchdog_armed = False
            self._invalidate_reference()
            age = self.backend.joint_state_age
            if age is None or age > self.config.max_obs_age_s:
                # Measured pose unknown: sending it would be a jump. The position
                # controller keeps its last goal, which is at most one step away.
                logger.warning(f"Hold ({reason}) with stale joint state: no command sent.")
                return
            arm, _ = self._measured_arm()
            self.backend.send_action(dict(zip(self._arm_keys, arm.tolist(), strict=True)))
            self._held_arm = arm.tolist()
            logger.info(f"Hold commanded ({reason}).")

    # ------------------------------------------------------------------ watchdog

    def _watchdog_loop(self) -> None:
        cfg = self.config
        timeout = cfg.watchdog_periods * cfg.control_period_s
        while not self._watchdog_stop.wait(cfg.control_period_s / 4):
            with self._cmd_lock:
                if not self._watchdog_armed or self._last_cmd_time is None:
                    continue
                silent = time.monotonic() - self._last_cmd_time
                if silent <= timeout:
                    continue
                self.watchdog_trips += 1
                logger.warning(f"Watchdog: no action for {silent:.2f} s (limit {timeout:.2f} s).")
                try:
                    self.hold("watchdog")
                except Exception as e:
                    self._watchdog_armed = False
                    logger.error(f"Watchdog hold failed: {e}")
