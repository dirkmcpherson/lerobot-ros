"""lerobot Robot for the Kinova Gen3 Lite over ROS 2 (see ``config_gen3_lite_ros``)."""

import logging
import time
from typing import Any

from lerobot_backends.robot import BackendRobot

from .config_gen3_lite_ros import Gen3LiteRosConfig

logger = logging.getLogger(__name__)


class Gen3LiteRos(BackendRobot):
    config_class = Gen3LiteRosConfig
    name = "gen3_lite_ros"

    def __init__(self, config: Gen3LiteRosConfig):
        super().__init__(config)
        self.config = config
        self._guard = None
        self._held_arm: list[float] | None = None  # last arm target the guard accepted
        self._last_block_log = 0.0
        self.blocked_commands = 0

    def connect(self, calibrate: bool = True) -> None:
        cfg = self.config
        if cfg.guard_enabled:
            from lerobot_backends.ros2.workspace_guard import (
                WorkspaceGuard,
                gen3_lite_srdf,
                gen3_lite_urdf,
            )

            iface = cfg.ros2_interface
            self._guard = WorkspaceGuard(
                urdf_path=cfg.urdf_path or gen3_lite_urdf(),
                arm_joint_names=iface.arm_joint_names,
                gripper_joint_name=iface.gripper_joint_name,
                plane_z=cfg.plane_z,
                plane_margin=cfg.plane_margin,
                self_margin=cfg.self_collision_margin,
                box_min=cfg.workspace_min,
                box_max=cfg.workspace_max,
                srdf_path=gen3_lite_srdf(),
            )
        super().connect(calibrate)
        if self._guard is not None:
            arm, grip = self._measured()
            violation, reason = self._guard.assess(arm, grip)
            logger.info(
                f"Workspace guard on: table plane z={cfg.plane_z} (+{cfg.plane_margin} m), "
                f"tool box {cfg.workspace_min}..{cfg.workspace_max}. Start pose: "
                + (f"VIOLATES ({reason}); only moves that reduce it are allowed" if violation else "ok")
            )

    def _measured(self) -> tuple[list[float], float]:
        iface = self.config.ros2_interface
        pos = self.backend.joint_positions()
        grip = pos.get(f"{iface.gripper_joint_name}.pos", 0.0) if iface.gripper_joint_name else 0.0
        return [pos[f"{j}.pos"] for j in iface.arm_joint_names], grip

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        action = self._apply_step_cap(action)
        if self._guard is not None:
            action = self._guard_action(action)
        return self.backend.send_action(action)

    def _guard_action(self, action: dict[str, Any]) -> dict[str, Any]:
        iface = self.config.ros2_interface
        keys = [f"{j}.pos" for j in iface.arm_joint_names]
        lo, hi = iface.min_joint_positions, iface.max_joint_positions
        # Evaluate what the interface will actually send (it clamps to joint limits).
        target = [min(max(float(action[k]), a), b) for k, a, b in zip(keys, lo, hi, strict=True)]
        arm, grip_meas = self._measured()
        grip = float(action.get("gripper.pos", grip_meas))

        violation, reason = self._guard.assess(target, grip)
        if violation > 0:
            current, _ = self._guard.assess(arm, grip_meas)
            if violation >= current - 1e-6:
                self.blocked_commands += 1
                now = time.monotonic()
                if now - self._last_block_log > 1.0:
                    logger.warning(f"Workspace guard blocked a command: {reason}")
                    self._last_block_log = now
                hold = self._held_arm or arm
                return {**action, **dict(zip(keys, hold, strict=True))}
        self._held_arm = target
        return action
