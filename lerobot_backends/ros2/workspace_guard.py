"""Workspace guard for joint-position commands: table plane, tool box, self-collision.

Checks a commanded arm configuration against
  * a horizontal plane (the table) that every moving link must clear by ``plane_margin``,
  * an axis-aligned box the tool frame must stay inside,
  * self-collision between links (pairs listed as never-colliding in the SRDF are skipped),
using the URDF collision meshes (pinocchio + hpp-fcl, via ``user_input.collision``).

A command is accepted if it violates nothing, or if it violates less than the current
measured configuration does, so an arm that starts (or is pushed) into a violation
can always be driven back out.
"""

import logging
import tempfile

import numpy as np

logger = logging.getLogger(__name__)


def gen3_lite_urdf() -> str:
    """Write the Gen3 Lite + 2F gripper URDF from kortex_description to a temp file."""
    import xacro
    from ament_index_python.packages import get_package_share_directory

    xacro_path = (
        f"{get_package_share_directory('kortex_description')}/robots/gen3_lite_gen3_lite_2f.xacro"
    )
    urdf = xacro.process_file(xacro_path).toxml()
    with tempfile.NamedTemporaryFile("w", suffix="_gen3_lite.urdf", delete=False) as f:
        f.write(urdf)
    logger.info(f"Generated Gen3 Lite URDF from {xacro_path} -> {f.name}")
    return f.name


def gen3_lite_srdf() -> str | None:
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

    try:
        share = get_package_share_directory("kinova_gen3_lite_moveit_config")
    except PackageNotFoundError:
        return None
    return f"{share}/config/gen3_lite.srdf"


class WorkspaceGuard:
    def __init__(
        self,
        urdf_path: str,
        arm_joint_names: list[str],
        gripper_joint_name: str | None,
        plane_z: float = 0.0,
        plane_margin: float = 0.02,
        self_margin: float = 0.005,
        box_min: list[float] | None = None,
        box_max: list[float] | None = None,
        box_frame: str = "tool_frame",
        srdf_path: str | None = None,
    ):
        import pinocchio as pin
        from ament_index_python.packages import get_package_share_directory

        from user_input.collision import CollisionChecker

        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._arm_idx = [self.model.joints[self.model.getJointId(n)].idx_q for n in arm_joint_names]
        self._gripper = gripper_joint_name
        self._box_frame = self.model.getFrameId(box_frame)
        self._box_min = None if box_min is None else np.asarray(box_min, float)
        self._box_max = None if box_max is None else np.asarray(box_max, float)
        self.plane_margin = plane_margin
        self.self_margin = self_margin

        share_parent = str(
            __import__("pathlib").Path(get_package_share_directory("kortex_description")).parent
        )
        checker = CollisionChecker(
            urdf_path=urdf_path,
            package_dirs=[share_parent],
            srdf_path=srdf_path,
            ground_plane_z=plane_z,
        )
        checker.setup(self.model)
        self._geom_model = checker._geom_model
        self._geom_data = checker._geom_data
        ground = [
            i for i, g in enumerate(self._geom_model.geometryObjects) if g.name == "ground_plane"
        ]
        self._is_plane_pair = np.array(
            [p.first in ground or p.second in ground for p in self._geom_model.collisionPairs]
        )

    def _q(self, arm: list[float], gripper: float) -> np.ndarray:
        q = self._pin.neutral(self.model)
        for idx, v in zip(self._arm_idx, arm, strict=True):
            q[idx] = v
        if self._gripper is not None:
            # The URDF's finger joints mimic right_finger_bottom_joint; pinocchio does not
            # apply <mimic>, so set them here (multipliers/offsets from the URDF).
            g = float(np.clip(gripper, 0.0, 0.85))
            for name, mult, off in (
                ("right_finger_bottom_joint", 1.0, 0.0),
                ("left_finger_bottom_joint", 1.0, 0.0),
                ("right_finger_tip_joint", -0.276, -0.1),
                ("left_finger_tip_joint", -0.276, -0.1),
            ):
                if self.model.existJointName(name):
                    q[self.model.joints[self.model.getJointId(name)].idx_q] = mult * g + off
        return q

    def assess(self, arm: list[float], gripper: float = 0.0) -> tuple[float, str]:
        """Return (violation, reason). violation is 0 when the configuration is allowed,
        otherwise the total metres by which margins and the box are exceeded."""
        pin = self._pin
        q = self._q(arm, gripper)
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.computeDistances(self.model, self.data, self._geom_model, self._geom_data, q)
        dist = np.array([r.min_distance for r in self._geom_data.distanceResults])

        violation, reasons = 0.0, []
        plane = dist[self._is_plane_pair]
        if plane.size and plane.min() < self.plane_margin:
            k = int(np.flatnonzero(self._is_plane_pair)[plane.argmin()])
            link = self._geom_model.geometryObjects[self._geom_model.collisionPairs[k].first].name
            violation += self.plane_margin - plane.min()
            reasons.append(f"{link} {plane.min() * 100:.1f} cm above table plane")
        selfd = dist[~self._is_plane_pair]
        if selfd.size and selfd.min() < self.self_margin:
            k = int(np.flatnonzero(~self._is_plane_pair)[selfd.argmin()])
            pair = self._geom_model.collisionPairs[k]
            names = [self._geom_model.geometryObjects[i].name for i in (pair.first, pair.second)]
            violation += self.self_margin - selfd.min()
            reasons.append(f"self-collision {names[0]}/{names[1]} {selfd.min() * 100:.1f} cm")
        if self._box_min is not None:
            p = self.data.oMf[self._box_frame].translation
            out = np.maximum(self._box_min - p, 0) + np.maximum(p - self._box_max, 0)
            if out.any():
                violation += float(out.sum())
                reasons.append(f"tool at {np.round(p, 3).tolist()} outside workspace box")
        return violation, "; ".join(reasons)

    def tool_position(self, arm: list[float]) -> np.ndarray:
        q = self._q(arm, 0.0)
        self._pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self._box_frame].translation.copy()
