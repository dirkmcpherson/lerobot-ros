"""SpacemouseUserInput — end-effector Cartesian teleoperation.

The spacemouse produces six continuous axes (x, y, z, roll, pitch, yaw) that
map naturally to end-effector linear and angular velocity. This implementation
uses differential inverse kinematics (Jacobian pseudoinverse) to convert that
Cartesian velocity into joint-space position targets, keeping the UserInput
interface consistent with other devices.

At each control step:
  1. Read spacemouse state (non-blocking).
  2. Apply dead-zone and scale to a 6-D Cartesian velocity vector (m/s, rad/s).
  3. Compute the geometric Jacobian J at the current joint configuration using
     PyKDL and the robot URDF.
  4. Solve dq = pinv(J) · v_cart  (minimum-norm joint velocity).
  5. Integrate: target = current_positions + dq * dt.
  6. Clamp targets to joint limits.

Buttons
-------
Left  (button 0)   Save episode
Right (button 1)   Discard episode
Both simultaneously Quit recording

Requirements
------------
  pip install pyspacemouse
  conda install pinocchio -c conda-forge

Parameters
----------
urdf_path       : Path to the processed robot URDF file.
                  Generate with:
                  source <ws>/install/setup.bash
                  xacro <ws>/src/ros2_kortex/kortex_description/robots/gen3.xacro \
                      dof:=7 vision:=false > /tmp/gen3.urdf
base_link       : Root link of the kinematic chain (default: "base_link")
ee_link         : End-effector link (default: "tool_frame" for Kinova Gen3)
linear_scale    : Max end-effector linear velocity, m/s (default: 0.1)
angular_scale   : Max end-effector angular velocity, rad/s (default: 0.5)
dead_zone       : Fractional axis dead-zone [0, 1] (default: 0.05)
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
        linear_scale: float = 0.1,
        angular_scale: float = 0.5,
        dead_zone: float = 0.05,
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
        self._n = len(joint_names)

        self._targets: np.ndarray = np.zeros(self._n)
        self._last_time: float = time.perf_counter()
        self._pin_model = None  # pinocchio model, created in connect()
        self._pin_data = None
        self._ee_frame_id: int = -1
        self._device = None   # pyspacemouse device, opened in connect()
        self._device_cm = None

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

        # Enter the context manager manually so connect/disconnect can be
        # separate calls while still getting proper cleanup via __exit__.
        self._device_cm = pyspacemouse.open()
        self._device = self._device_cm.__enter__()
        if self._device is None:
            raise RuntimeError(
                "No SpaceMouse device found. Check USB connection and udev rules."
            )
        self._last_time = time.perf_counter()
        logger.info(
            "SpaceMouse connected (end-effector Cartesian control via differential IK). "
            "Left=save | Right=discard | Both=quit"
        )

    def disconnect(self) -> None:
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

    def reset(self, current_positions: list[float]) -> None:
        with self._lock:
            self._targets = np.array(current_positions, dtype=float)
            self._last_time = time.perf_counter()
            self._end = False
            self._discard = False
            self._prev_buttons = [0, 0]

    def get_action(self, current_positions: list[float]) -> list[float]:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        state = self._device.read()

        with self._lock:
            if state is not None:
                # --- Cartesian velocity from spacemouse ---
                axes = np.array([state.x, state.y, state.z,
                                  state.roll, state.pitch, state.yaw])
                axes[:3] *= self._linear_scale
                axes[3:] *= self._angular_scale

                # Dead-zone: zero out small inputs to avoid drift
                axes[np.abs(axes) < self._dead_zone] = 0.0

                if np.any(axes != 0.0):
                    # --- Differential IK ---
                    J = self._jacobian(current_positions)
                    # Minimum-norm joint velocity: dq = J^+ · v_cart
                    J_pinv = np.linalg.pinv(J)
                    dq = J_pinv @ axes

                    # Integrate and clamp
                    new_targets = np.array(current_positions) + dq * dt
                    self._targets = np.clip(new_targets, self._min, self._max)

                self._handle_buttons(state.buttons)

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

        # Map each arm joint name to its index in pinocchio's q vector.
        # The URDF may contain extra joints (gripper, tool, etc.) so nq > len(joint_names).
        # We fill a full q vector of size nq and extract the matching Jacobian columns.
        self._q_indices: list[int] = []
        for name in self._names:
            if not self._pin_model.existJointName(name):
                raise RuntimeError(
                    f"Joint '{name}' not found in pinocchio model. "
                    f"Available joints: {[self._pin_model.names[i] for i in range(self._pin_model.njoints)]}"
                )
            jid = self._pin_model.getJointId(name)
            self._q_indices.append(self._pin_model.joints[jid].idx_q)

        logger.info(
            f"Pinocchio model loaded: nq={self._pin_model.nq}, "
            f"controlling {len(self._names)} arm joints at q indices {self._q_indices}, "
            f"end-effector frame '{self._ee_link}' (id={self._ee_frame_id})"
        )

    def _jacobian(self, joint_positions: list[float]) -> np.ndarray:
        """Return the 6×N Jacobian (N = number of arm joints) at the given
        configuration, expressed in the world-aligned local frame of the EE."""
        import pinocchio as pin

        # Build a full configuration vector (size nq), neutral everywhere except
        # for our arm joints which are set to the current positions.
        q = pin.neutral(self._pin_model)
        for idx_q, pos in zip(self._q_indices, joint_positions):
            q[idx_q] = pos

        pin.computeJointJacobians(self._pin_model, self._pin_data, q)
        J_full = pin.getFrameJacobian(
            self._pin_model,
            self._pin_data,
            self._ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        # Extract only the columns for our controlled joints (6 × n_arm_joints)
        return J_full[:, self._q_indices]

    # ------------------------------------------------------------------
    # Button handling (called inside self._lock)
    # ------------------------------------------------------------------

    def _handle_buttons(self, buttons: list[int]) -> None:
        b = list(buttons) + [0, 0]
        prev = self._prev_buttons

        left_pressed  = b[0] and not prev[0]
        right_pressed = b[1] and not prev[1]
        both_held     = b[0] and b[1]

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
