"""SpaceMouse teleop that emits absolute joint targets, for joint-position robots.

Wraps ``user_input.SpacemouseUserInput`` (damped-least-squares differential IK with
pinocchio) as a lerobot ``Teleoperator``. lerobot's ``get_action()`` gets no robot
observation, so this teleop subscribes to ``/joint_states`` itself: each target is
the measured pose plus one IK step, which means it never drifts from the robot.

Handover: when ``get_action()`` has not been called for ``resync_after_sec`` (a DAgger
correction starting, or a new recording episode), the targets restart from the
measured pose and the gripper toggle restarts from the measured gripper, so taking
over neither jumps the arm nor drops what it is holding.

Buttons: right toggles the gripper. Episode flow (save, discard, DAgger pause /
correct) is lerobot's keyboard handling, not the SpaceMouse.
"""

import logging
import math
import threading
import time
from types import SimpleNamespace
from typing import Any

from lerobot.teleoperators import Teleoperator

from .config_spacemouse_joint import SpacemouseJointTeleopConfig

logger = logging.getLogger(__name__)


class SpacemouseJointTeleop(Teleoperator):
    config_class = SpacemouseJointTeleopConfig
    name = "spacemouse_joint"

    def __init__(self, config: SpacemouseJointTeleopConfig):
        super().__init__(config)
        self.config = config
        self._input = None
        self._node = None
        self._executor = None
        self._executor_thread = None
        self._lock = threading.Lock()
        self._measured: dict[str, float] | None = None
        self._last_call: float | None = None
        self._last_targets: list[float] | None = None
        self._t0 = time.monotonic()

    @property
    def action_features(self) -> dict:
        features = {f"{j}.pos": float for j in self.config.joint_names}
        if self.config.gripper_joint_name:
            features["gripper.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {}  # not actuated

    @property
    def is_connected(self) -> bool:
        return self._input is not None

    def connect(self, calibrate: bool = True) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from sensor_msgs.msg import JointState

        from lerobot_backends.ros2.workspace_guard import gen3_lite_urdf
        from user_input.spacemouse import SpacemouseUserInput

        cfg = self.config
        self._input = SpacemouseUserInput(
            joint_names=cfg.joint_names,
            min_joint_positions=cfg.min_joint_positions,
            max_joint_positions=cfg.max_joint_positions,
            urdf_path=cfg.urdf_path or gen3_lite_urdf(),
            base_link=cfg.base_link,
            ee_link=cfg.ee_link,
            linear_scale=cfg.linear_scale,
            angular_scale=cfg.angular_scale,
            dead_zone=cfg.dead_zone,
            gripper_joint_name=cfg.gripper_joint_name,
            gripper_open_position=cfg.gripper_open_position,
            gripper_closed_position=cfg.gripper_closed_position,
            damping=cfg.damping,
            input_smoothing=cfg.input_smoothing,
        )
        if cfg.device == "spacemouse":
            self._input.connect()
        else:
            self._input._setup_kdl()  # IK only; get_action() feeds synthetic device states

        if not rclpy.ok():
            rclpy.init()
        self._node = Node(
            "spacemouse_joint_teleop",
            parameter_overrides=[Parameter("use_sim_time", value=cfg.use_sim_time)],
        )
        self._node.create_subscription(JointState, cfg.joint_states_topic, self._on_joint_state, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

        deadline = time.monotonic() + 10.0
        while self._measured is None:
            if time.monotonic() > deadline:
                self.disconnect()
                raise TimeoutError(
                    f"No complete message on {cfg.joint_states_topic} within 10 s: is the robot driver "
                    "running (kortex_bringup gen3_lite.launch.py), in a shell with the same RMW / "
                    "ROS_DOMAIN_ID (both sourced from gen3_lite_env.sh)?"
                )
            time.sleep(0.05)
        logger.info(f"{self} connected ({cfg.device}).")

    def _on_joint_state(self, msg) -> None:
        names = list(self.config.joint_names)
        if self.config.gripper_joint_name:
            names.append(self.config.gripper_joint_name)
        index = {n: i for i, n in enumerate(msg.name)}
        if all(n in index for n in names):
            with self._lock:
                self._measured = {n: msg.position[index[n]] for n in names}

    def _scripted_state(self) -> SimpleNamespace:
        # Slow 8 s circle in the horizontal plane plus a gentle vertical bob.
        t = time.monotonic() - self._t0
        w = 2 * math.pi / 8.0
        return SimpleNamespace(
            x=0.6 * math.cos(w * t), y=0.6 * math.sin(w * t), z=0.4 * math.sin(2 * w * t),
            roll=0.0, pitch=0.0, yaw=0.0, buttons=[0, 0],
        )

    def get_action(self) -> dict[str, Any]:
        cfg = self.config
        with self._lock:
            measured = dict(self._measured)
        q = [measured[j] for j in cfg.joint_names]

        now = time.monotonic()
        if self._last_call is None or now - self._last_call > cfg.resync_after_sec:
            self._input.reset(q)
            self._last_targets = None
            if cfg.gripper_joint_name:
                self._input.set_gripper(measured[cfg.gripper_joint_name] < cfg.gripper_closed_threshold)
            logger.info("SpaceMouse re-synced to the measured robot pose.")
        self._last_call = now

        if cfg.device == "scripted":
            self._input._latest_state = self._scripted_state()
        ref = q
        if cfg.integrate_from_target and self._last_targets is not None:
            lead = cfg.max_joint_step if cfg.max_joint_step is not None else 0.1
            ref = [min(max(t, m - lead), m + lead) for t, m in zip(self._last_targets, q, strict=True)]
        targets = self._input.get_action(ref)
        if cfg.max_joint_step is not None:
            step = cfg.max_joint_step
            targets = [min(max(t, m - step), m + step) for t, m in zip(targets, q, strict=True)]
        self._last_targets = list(targets)

        action = {f"{j}.pos": float(v) for j, v in zip(cfg.joint_names, targets, strict=True)}
        if cfg.gripper_joint_name:
            action["gripper.pos"] = float(self._input.gripper_position)
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor_thread.join(timeout=2.0)
            self._executor = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._input is not None:
            if self.config.device == "spacemouse":
                self._input.disconnect()
            self._input = None
