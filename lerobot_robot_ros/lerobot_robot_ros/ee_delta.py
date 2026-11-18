"""End-effector delta control helpers: step limiting and arm kinematics.

No ROS imports here, so everything in this module is unit-testable without hardware.

``limit_step`` turns one normalised action (each axis in [-1, 1]) into a tool-position
target, applying in order: the action clip, the per-step length cap, the speed limit,
the lead limit (the target may not run ahead of the measured tool position) and the
workspace box.

``ArmKinematics`` wraps pinocchio for forward kinematics and a damped-least-squares
inverse kinematics that moves the tool to a position while holding an orientation.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EeLimits:
    # Metres of tool travel per unit of normalised action, per axis.
    step_size_m: float = 0.01
    # Cap on the Euclidean length of one commanded step (m).
    max_step_m: float = 0.01
    # Speed limit (m/s). A step may not be longer than max_speed_m_s * time since the
    # previous command (capped at control_period_s), so calling faster than the control
    # rate cannot raise the speed.
    max_speed_m_s: float = 0.1
    control_period_s: float = 0.1
    # The target may not be further than this from the measured tool position (m).
    max_lead_m: float = 0.03
    # Workspace box for the tool frame, base frame, metres.
    box_min: list[float] = field(default_factory=lambda: [0.15, 0.03, 0.04])
    box_max: list[float] = field(default_factory=lambda: [0.43, 0.29, 0.40])

    def validate(self) -> None:
        lo, hi = np.asarray(self.box_min, float), np.asarray(self.box_max, float)
        if lo.shape != (3,) or hi.shape != (3,) or not np.all(lo < hi):
            raise ValueError(f"Workspace box must be 3-D with min < max, got {self.box_min}..{self.box_max}")
        for name in ("step_size_m", "max_step_m", "max_speed_m_s", "control_period_s", "max_lead_m"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


@dataclass
class StepReport:
    """What limit_step did to one action."""

    requested_m: np.ndarray  # delta asked for, after the [-1, 1] action clip
    applied_m: np.ndarray  # target - reference
    action_clipped: bool = False
    step_clipped: bool = False
    lead_clipped: bool = False
    box_clipped: bool = False

    @property
    def any_clipped(self) -> bool:
        return self.action_clipped or self.step_clipped or self.lead_clipped or self.box_clipped


def _cap_norm(v: np.ndarray, cap: float) -> tuple[np.ndarray, bool]:
    n = float(np.linalg.norm(v))
    if n > cap:
        return v * (cap / n), True
    return v, False


def limit_step(
    action_xyz,
    reference_pos,
    measured_pos,
    limits: EeLimits,
    dt_since_last_command: float | None = None,
) -> tuple[np.ndarray, StepReport]:
    """Return (target tool position, report) for one normalised xyz action.

    ``reference_pos`` is the position the delta is added to (the previous target),
    ``measured_pos`` the tool position computed from the measured joints.
    """
    a = np.asarray(action_xyz, dtype=float).reshape(3)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"Non-finite action {a}")
    reference = np.asarray(reference_pos, dtype=float).reshape(3)
    measured = np.asarray(measured_pos, dtype=float).reshape(3)

    clipped = np.clip(a, -1.0, 1.0)
    requested = clipped * limits.step_size_m
    report = StepReport(requested_m=requested, applied_m=requested, action_clipped=bool(np.any(clipped != a)))

    dt = limits.control_period_s
    if dt_since_last_command is not None:
        dt = float(np.clip(dt_since_last_command, 0.0, limits.control_period_s))
    cap = min(limits.max_step_m, limits.max_speed_m_s * dt)

    delta, report.step_clipped = _cap_norm(requested, cap)
    target = reference + delta

    lead, report.lead_clipped = _cap_norm(target - measured, limits.max_lead_m)
    target = measured + lead

    # Clip to the box. Where the reference is already outside (the arm started there), the
    # box is widened to the reference on that axis: a zero action never moves the arm,
    # moving back in is allowed, moving further out is not.
    lo = np.minimum(np.asarray(limits.box_min, float), reference)
    hi = np.maximum(np.asarray(limits.box_max, float), reference)
    boxed = np.clip(target, lo, hi)
    report.box_clipped = bool(np.any(boxed != target))
    target = boxed

    report.applied_m = target - reference
    return target, report


class ArmKinematics:
    """Forward and inverse kinematics for the arm joints of a URDF (pinocchio)."""

    def __init__(self, urdf_path: str, joint_names: list[str], frame: str = "tool_frame"):
        import pinocchio as pin

        self._pin = pin
        full = pin.buildModelFromUrdf(urdf_path)
        missing = [n for n in joint_names if not full.existJointName(n)]
        if missing:
            raise ValueError(f"Joints {missing} not in {urdf_path}")
        # Lock everything that is not an arm joint (fingers) at its neutral position.
        lock = [i for i in range(1, full.njoints) if full.names[i] not in joint_names]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full))
        self.data = self.model.createData()
        if not self.model.existFrame(frame):
            raise ValueError(f"Frame {frame} not in {urdf_path}")
        self._frame = self.model.getFrameId(frame)
        self._idx = [self.model.joints[self.model.getJointId(n)].idx_q for n in joint_names]
        if sorted(self._idx) != list(range(self.model.nq)):
            raise ValueError("Arm joints must be single-DoF joints covering the reduced model")
        self.lower = self.model.lowerPositionLimit[self._idx].copy()
        self.upper = self.model.upperPositionLimit[self._idx].copy()

    def _q(self, joints) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[self._idx] = np.asarray(joints, dtype=float)
        return q

    def fk(self, joints) -> tuple[np.ndarray, np.ndarray]:
        """Tool position (3,) and rotation matrix (3, 3) in the base frame."""
        pin = self._pin
        pin.framesForwardKinematics(self.model, self.data, self._q(joints))
        pose = self.data.oMf[self._frame]
        return pose.translation.copy(), pose.rotation.copy()

    def ik(
        self,
        seed,
        target_pos,
        target_rot,
        lower=None,
        upper=None,
        iterations: int = 30,
        damping: float = 0.03,
        rot_weight: float = 0.3,
        max_iter_step: float = 0.05,
    ) -> tuple[np.ndarray, float, float]:
        """Damped least squares from ``seed`` to the target pose.

        Returns (joints, position error in m, orientation error in rad). The caller
        decides whether the errors are acceptable. ``rot_weight`` scales the
        orientation rows (m per rad), so position accuracy wins near singularities.
        """
        pin = self._pin
        lo = self.lower if lower is None else np.asarray(lower, float)
        hi = self.upper if upper is None else np.asarray(upper, float)
        q = self._q(seed)
        target_pos = np.asarray(target_pos, float)
        target_rot = np.asarray(target_rot, float)
        weights = np.array([1, 1, 1, rot_weight, rot_weight, rot_weight])
        e_pos = e_rot = np.zeros(3)
        for _ in range(iterations):
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pose = self.data.oMf[self._frame]
            e_pos = target_pos - pose.translation
            e_rot = pin.log3(target_rot @ pose.rotation.T)  # base-frame rotation vector
            if np.linalg.norm(e_pos) < 1e-5 and np.linalg.norm(e_rot) < 1e-4:
                break
            jac = pin.getFrameJacobian(self.model, self.data, self._frame, pin.LOCAL_WORLD_ALIGNED)
            jw = jac * weights[:, None]
            err = np.concatenate([e_pos, e_rot]) * weights
            dq = jw.T @ np.linalg.solve(jw @ jw.T + damping**2 * np.eye(6), err)
            biggest = np.abs(dq).max()
            if biggest > max_iter_step:
                dq *= max_iter_step / biggest
            q[self._idx] = np.clip(q[self._idx] + dq[self._idx], lo, hi)
        pin.framesForwardKinematics(self.model, self.data, q)
        pose = self.data.oMf[self._frame]
        e_pos = target_pos - pose.translation
        e_rot = pin.log3(target_rot @ pose.rotation.T)
        return q[self._idx].copy(), float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
