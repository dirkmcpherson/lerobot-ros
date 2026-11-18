"""Scripted end-effector delta teleoperator (see ``config_scripted_ee``)."""

from typing import Any

import numpy as np
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

from .config_scripted_ee import ScriptedEeTeleopConfig

# Gen3LiteHil registers functions returning its measured tool position and gripper joint
# here when it connects, because lerobot gives a teleoperator no robot handle.
TOOL_POSITION_SOURCE: dict[str, Any] = {"get": None, "gripper": None}

CLOSE, STAY, OPEN = 0.0, 1.0, 2.0


class ScriptedEeTeleop(Teleoperator):
    config_class = ScriptedEeTeleopConfig
    name = "scripted_ee"

    def __init__(self, config: ScriptedEeTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self.step = 0
        self.episode = 0
        self.grasped = False  # the gripper has closed at the pick point this episode

    @property
    def action_features(self) -> dict:
        names = {"delta_x": 0, "delta_y": 1, "delta_z": 2}
        if self.config.use_gripper:
            names["gripper"] = 3
        return {"dtype": "float32", "shape": (len(names),), "names": names}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def _tool(self) -> np.ndarray:
        get = TOOL_POSITION_SOURCE["get"]
        if get is None:
            raise RuntimeError("scripted_ee needs a connected gen3_lite_hil / gen3_lite_hil_fake robot")
        return np.asarray(get(), float)

    def _gripper(self) -> float:
        get = TOOL_POSITION_SOURCE["gripper"]
        return float(get()) if get is not None else 0.0

    def _near(self, point) -> bool:
        return bool(np.linalg.norm(np.asarray(point, float) - self._tool()) < self.config.success_radius_m)

    def _towards(self, point) -> np.ndarray:
        return np.clip((np.asarray(point, float) - self._tool()) / self.config.ee_step_size_m, -1.0, 1.0)

    def get_action(self) -> dict[str, Any]:
        cfg = self.config
        if not cfg.use_gripper:
            delta, gripper = self._towards(cfg.target), None
        elif not self.grasped:  # to the pick point, then close
            at_pick = self._near(cfg.target)
            delta = np.zeros(3) if at_pick else self._towards(cfg.target)
            gripper = CLOSE if at_pick else OPEN
        else:  # carry to the place point, then open
            at_place = self._near(cfg.place_target)
            delta = np.zeros(3) if at_place else self._towards(cfg.place_target)
            gripper = OPEN if at_place else STAY
        action = {"delta_x": float(delta[0]), "delta_y": float(delta[1]), "delta_z": float(delta[2])}
        if gripper is not None:
            action["gripper"] = gripper
        return action

    def _success(self) -> bool:
        cfg = self.config
        if not cfg.use_gripper:
            return self._near(cfg.target)
        position = self._gripper()
        if not self.grasped and position > cfg.gripper_closed_above and self._near(cfg.target):
            self.grasped = True
        return self.grasped and position < cfg.gripper_open_below and self._near(cfg.place_target)

    def get_teleop_events(self) -> dict[str, Any]:
        cfg = self.config
        allowed = cfg.max_takeover_episodes is None or self.episode < cfg.max_takeover_episodes
        in_window = any(a <= self.step < b for a, b in cfg.takeovers)
        intervening = cfg.always_intervene or (allowed and in_window)
        success = self._success()
        self.step += 1
        if success or self.step >= cfg.episode_steps:
            self.step = 0
            self.episode += 1
            self.grasped = False
        return {
            TeleopEvents.IS_INTERVENTION: intervening,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: False,
        }
