"""SpacemouseUserInput — end-effector Cartesian teleoperation.

The spacemouse produces six continuous axes (x, y, z, roll, pitch, yaw) that
map naturally to end-effector linear and angular velocity. This implementation
uses differential inverse kinematics (Jacobian pseudoinverse) to convert that
Cartesian velocity into joint-space position targets, keeping the UserInput
interface consistent with other devices.

At each control step:
  1. Read spacemouse state (non-blocking via background thread caching device.read()).
  2. Apply dead-zone (on raw axes) then scale to a 6-D Cartesian velocity (m/s, rad/s).
  3. Compute the geometric Jacobian J at the current joint configuration using
     pinocchio and the robot URDF.
  4. Solve dq = pinv(J) · v_cart  (minimum-norm joint velocity).
  5. Integrate: target = current_positions + dq * dt.
  6. Clamp targets to joint limits.

Buttons (no gripper configured)
--------------------------------
Left  (button 0)   Save episode
Right (button 1)   Discard episode
Both simultaneously Quit recording

Buttons (gripper configured)
-----------------------------
Left  (button 0)   Save episode
Right (button 1)   Toggle gripper open/closed
Both simultaneously Discard episode and quit recording

Gripper
-------
Pass gripper_joint_name, gripper_open_position, and gripper_closed_position to
enable gripper control. The gripper is NOT part of the pinocchio IK chain; its
position is tracked separately and exposed via the gripper_position property.
get_action() still returns only arm joint positions; the caller reads
user_input.gripper_position and sends it via send_gripper_command() separately.

Requirements
------------
  pip install pyspacemouse
  conda install pinocchio -c conda-forge

Parameters
----------
urdf_path            : Path to the processed robot URDF file.
base_link            : Root link of the kinematic chain (default: "base_link")
ee_link              : End-effector link (default: "tool_frame" for Kinova Gen3)
linear_scale         : Max end-effector linear velocity, m/s (default: 0.1)
angular_scale        : Max end-effector angular velocity, rad/s (default: 0.5)
dead_zone            : Fractional dead-zone on raw [-1,1] axes (default: 0.05)
gripper_joint_name   : Joint name of the gripper, or None to disable (default: None)
gripper_open_position : Gripper joint position when fully open (default: 0.0)
gripper_closed_position : Gripper joint position when fully closed (default: 0.85)
damping              : DLS damping factor λ for singularity robustness (default: 0.05)
input_smoothing      : EMA alpha on spacemouse axes; lower = smoother but laggier (default: 0.5)
collision_srdf_path  : SRDF file for self-collision exclusion pairs (default: None = no checking)
collision_package_dirs : Directories to resolve package:// mesh paths (default: None)
collision_ground_plane_z : Z height of ground plane obstacle (default: None = no ground plane)
collision_obstacles  : List of box obstacles [{type, half_extents, position, name}] (default: None)
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

from .base import UserInput

logger = logging.getLogger(__name__)


class SpacemouseUserInput(UserInput):

    def __init__(
        self,
        joint_names: list[str],
        min_joint_positions: list[float],
        max_joint_positions: list[float],
        urdf_path: str,
        base_link: str = "base_link",
        ee_link: str = "tool_frame",
        linear_scale: float = 0.5,
        angular_scale: float = 1.5,
        dead_zone: float = 0.05,
        gripper_joint_name: Optional[str] = None,
        gripper_open_position: float = 0.0,
        gripper_closed_position: float = 0.85,
        damping: float = 0.05,
        input_smoothing: float = 0.5,
        collision_srdf_path: Optional[str] = None,
        collision_package_dirs: Optional[list[str]] = None,
        collision_ground_plane_z: Optional[float] = None,
        collision_obstacles: Optional[list[dict]] = None,
    ):
        self._names = joint_names
        self._min = np.array(min_joint_positions)
        self._max = np.array(max_joint_positions)
        self._urdf_path = urdf_path
        self._base_link = base_link
        self._ee_link = ee_link
        self._linear_scale = linear_scale
        self._angular_scale = angular_scale
        self._dead_zone = dead_zone
        self._damping = damping
        self._input_alpha = input_smoothing
        self._collision_srdf_path = collision_srdf_path
        self._collision_package_dirs = collision_package_dirs
        self._collision_ground_plane_z = collision_ground_plane_z
        self._collision_obstacles = collision_obstacles
        self._n = len(joint_names)

        self._gripper_joint_name = gripper_joint_name
        self._gripper_open_position = gripper_open_position
        self._gripper_closed_position = gripper_closed_position
        self._gripper_is_open: bool = True
        self._gripper_position: float = gripper_open_position

        self._targets: np.ndarray = np.zeros(self._n)
        self._smoothed_axes: np.ndarray = np.zeros(6)
        self._last_time: float = time.perf_counter()
        self._collision_checker = None  # CollisionChecker, created in connect()
        self._pin_model = None  # pinocchio model, created in connect()
        self._pin_data = None
        self._ee_frame_id: int = -1
        self._latest_state = None  # latest HID event, written by reader thread
        self._pending_buttons: list[int] | None = None  # button press not yet consumed
        self._device_cm = None
        self._device = None
        self._stop_reader: bool = False
        self._reader_thread: Optional[threading.Thread] = None

        self._end: bool = False
        self._discard: bool = False
        self._quit: bool = False
        self._prev_buttons: list[int] = [0, 0]
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # UserInput.time_from_start_sec override
    # ------------------------------------------------------------------

    @property
    def time_from_start_sec(self) -> float:
        # Short horizon: the trajectory controller should execute each
        # incremental step within one control cycle (~1/FPS seconds).
        return 0.1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        import pyspacemouse

        self._setup_kdl()

        # Open device using the confirmed context-manager pattern.
        self._device_cm = pyspacemouse.open()
        self._device = self._device_cm.__enter__()
        if self._device is None:
            raise RuntimeError(
                "No SpaceMouse device found. Check USB connection and udev rules."
            )

        # Spin a background thread so device.read() (blocking) never stalls
        # the control loop. get_action() reads _latest_state without blocking.
        self._stop_reader = False
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="spacemouse-reader"
        )
        self._reader_thread.start()

        self._last_time = time.perf_counter()
        logger.info(
            "SpaceMouse connected (end-effector Cartesian control via differential IK). "
            "Left=save | Right=discard | Both=quit"
        )

    def _reader_loop(self) -> None:
        """Background thread: blocks on device.read() and caches latest state.

        Button presses are latched separately so they can't be overwritten
        by a subsequent read before get_action() consumes them.
        """
        while not self._stop_reader:
            try:
                state = self._device.read()
                if state is not None:
                    with self._lock:
                        self._latest_state = state
                        # Latch button state so get_action() sees presses AND
                        # releases (needed to reset _prev_buttons for edge
                        # detection on the next press).
                        btns = list(state.buttons) + [0, 0]
                        if btns[0] or btns[1]:
                            self._pending_buttons = btns[:2]
                        elif self._prev_buttons[0] or self._prev_buttons[1]:
                            self._pending_buttons = [0, 0]
            except Exception:
                break

    def disconnect(self) -> None:
        self._stop_reader = True
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        if self._device_cm is not None:
            try:
                self._device_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._device_cm = None
            self._device = None

    # ------------------------------------------------------------------
    # UserInput interface
    # ------------------------------------------------------------------

    @property
    def gripper_position(self) -> float:
        """Current gripper position target (only meaningful when gripper is configured)."""
        with self._lock:
            return self._gripper_position

    def reset(self, current_positions: list[float]) -> None:
        with self._lock:
            self._targets = np.array(current_positions, dtype=float)
            self._smoothed_axes = np.zeros(6)
            self._last_time = time.perf_counter()
            self._end = False
            self._discard = False
            self._prev_buttons = [0, 0]
            self._pending_buttons = None
            self._gripper_is_open = True
            self._gripper_position = self._gripper_open_position

    def get_action(self, current_positions: list[float]) -> list[float]:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        with self._lock:
            # Snapshot latest state from the background thread (non-blocking).
            state = self._latest_state
            self._latest_state = None  # consume so we don't re-process same event
            if state is not None:
                # --- Cartesian velocity from spacemouse ---
                # Spacemouse physical axes → robot base frame:
                #   spacemouse forward (x) → robot +y
                #   spacemouse right   (y) → robot +x
                #   spacemouse up      (z) → robot +z
                axes = np.array([state.y, state.x, state.z,
                                  state.pitch, state.roll, state.yaw])

                # Dead-zone applied to raw [-1, 1] axes BEFORE scaling.
                axes[np.abs(axes) < self._dead_zone] = 0.0

                axes[:3] *= self._linear_scale
                axes[3:] *= self._angular_scale

                # EMA on input: smooth spacemouse noise and dead-zone transitions.
                # When raw input is all zeros (hand off device), snap to zero
                # immediately to prevent residual drift.
                if np.all(axes == 0.0):
                    self._smoothed_axes[:] = 0.0
                else:
                    self._smoothed_axes = (
                        self._input_alpha * axes
                        + (1.0 - self._input_alpha) * self._smoothed_axes
                    )
                axes = self._smoothed_axes

                if np.any(axes != 0.0):
                    # --- Differential IK (damped least squares) ---
                    J = self._jacobian(current_positions)
                    JJT = J @ J.T
                    dq = J.T @ np.linalg.solve(
                        JJT + self._damping**2 * np.eye(JJT.shape[0]), axes
                    )

                    # Integrate and clamp
                    new_targets = np.array(current_positions) + dq * dt
                    new_targets = np.clip(new_targets, self._min, self._max)

                    # Block moves that enter a collision — but allow moves
                    # that escape one (so the arm never gets stuck).
                    if self._collision_checker is not None:
                        q_new = self._build_q(new_targets.tolist())
                        if self._collision_checker.check_collision(q_new):
                            q_cur = self._build_q(current_positions)
                            if not self._collision_checker.check_collision(q_cur):
                                new_targets = self._targets  # block: safe → colliding

                    self._targets = new_targets

            # Process latched button presses (survives reader overwrite race)
            if self._pending_buttons is not None:
                self._handle_buttons(self._pending_buttons)
                self._pending_buttons = None

            return self._targets.tolist()

    @property
    def episode_end_requested(self) -> bool:
        with self._lock:
            return self._end

    @property
    def discard_requested(self) -> bool:
        with self._lock:
            return self._discard

    @property
    def quit_requested(self) -> bool:
        with self._lock:
            return self._quit

    # ------------------------------------------------------------------
    # Pinocchio setup and Jacobian computation
    # ------------------------------------------------------------------

    def _setup_kdl(self) -> None:
        import pinocchio as pin

        self._pin_model = pin.buildModelFromUrdf(self._urdf_path)
        self._pin_data = self._pin_model.createData()

        # Resolve the end-effector frame ID
        if not self._pin_model.existFrame(self._ee_link):
            available = [self._pin_model.frames[i].name for i in range(len(self._pin_model.frames))]
            raise RuntimeError(
                f"Frame '{self._ee_link}' not found in URDF. "
                f"Available frames: {available}"
            )
        self._ee_frame_id = self._pin_model.getFrameId(self._ee_link)

        # Map each arm joint name to its indices in pinocchio's q and v vectors.
        # URDF "continuous" joints use SO(2) representation: nq=2 (cos,sin), nv=1.
        # The Jacobian has shape (6, nv), so Jacobian columns must be indexed by
        # idx_v, while the configuration vector q is indexed by idx_q.
        self._q_indices: list[int] = []   # start index in q (configuration)
        self._v_indices: list[int] = []   # index in v (velocity / Jacobian columns)
        self._is_continuous: list[bool] = []  # True for unbounded revolute joints
        for name in self._names:
            if not self._pin_model.existJointName(name):
                raise RuntimeError(
                    f"Joint '{name}' not found in pinocchio model. "
                    f"Available joints: {[self._pin_model.names[i] for i in range(self._pin_model.njoints)]}"
                )
            jid = self._pin_model.getJointId(name)
            joint = self._pin_model.joints[jid]
            self._q_indices.append(joint.idx_q)
            self._v_indices.append(joint.idx_v)
            # Continuous (unbounded revolute) joints have nq=2, nv=1
            self._is_continuous.append(joint.nq == 2)

        logger.info(
            f"Pinocchio model loaded: nq={self._pin_model.nq}, nv={self._pin_model.nv}, "
            f"controlling {len(self._names)} arm joints, "
            f"v indices {self._v_indices}, "
            f"end-effector frame '{self._ee_link}' (id={self._ee_frame_id})"
        )

        # Set up collision checker if SRDF or ground plane is configured
        if self._collision_srdf_path is not None or self._collision_ground_plane_z is not None:
            from .collision import CollisionChecker

            self._collision_checker = CollisionChecker(
                urdf_path=self._urdf_path,
                package_dirs=self._collision_package_dirs,
                srdf_path=self._collision_srdf_path,
                ground_plane_z=self._collision_ground_plane_z,
                obstacles=self._collision_obstacles,
            )
            self._collision_checker.setup(self._pin_model)

    def _build_q(self, joint_positions: list[float]) -> np.ndarray:
        """Build a full pinocchio configuration vector from arm joint positions."""
        import pinocchio as pin

        q = pin.neutral(self._pin_model)
        for idx_q, pos, continuous in zip(self._q_indices, joint_positions, self._is_continuous):
            if continuous:
                q[idx_q]     = np.cos(pos)
                q[idx_q + 1] = np.sin(pos)
            else:
                q[idx_q] = pos
        return q

    def _jacobian(self, joint_positions: list[float]) -> np.ndarray:
        """Return the 6×N Jacobian (N = number of arm joints) at the given
        configuration, expressed in the world-aligned local frame of the EE."""
        import pinocchio as pin

        q = self._build_q(joint_positions)
        pin.computeJointJacobians(self._pin_model, self._pin_data, q)
        J_full = pin.getFrameJacobian(
            self._pin_model,
            self._pin_data,
            self._ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        # Extract arm-joint columns using v-vector indices (Jacobian is 6 × nv)
        return J_full[:, self._v_indices]

    # ------------------------------------------------------------------
    # Button handling (called inside self._lock)
    # ------------------------------------------------------------------

    def _handle_buttons(self, buttons: list[int]) -> None:
        b = list(buttons) + [0, 0]
        prev = self._prev_buttons

        left_pressed  = b[0] and not prev[0]
        right_pressed = b[1] and not prev[1]
        both_held     = b[0] and b[1]

        # print out the button states
        print(b, prev)

        if self._gripper_joint_name is not None:
            # Gripper mode: right/both = toggle gripper, left alone = save, Ctrl+C = quit
            if right_pressed or (both_held and not prev[1]):
                self._gripper_is_open = not self._gripper_is_open
                self._gripper_position = (
                    self._gripper_open_position if self._gripper_is_open
                    else self._gripper_closed_position
                )
                state = "open" if self._gripper_is_open else "closed"
                logger.info(f"Gripper toggle → {state} ({self._gripper_position:.3f})")
            elif left_pressed and not b[1]:
                logger.info("Left button — save episode.")
                self._end = True

        else:
            # No gripper: original mapping (left=save, right=discard, both=quit)
            if both_held:
                logger.info("Both buttons — quit requested.")
                self._quit = True
                self._end = True
            elif left_pressed:
                logger.info("Left button — save episode.")
                self._end = True
            elif right_pressed:
                logger.info("Right button — discard episode.")
                self._discard = True
                self._end = True

        self._prev_buttons = [b[0], b[1]]
