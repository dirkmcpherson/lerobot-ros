"""Scripted end-effector delta teleoperator (see ``config_scripted_ee``)."""

from typing import Any

import numpy as np
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

from .config_scripted_ee import ScriptedEeTeleopConfig

# Gen3LiteHil registers a function returning its measured tool position here when it
# connects, because lerobot gives a teleoperator no robot handle.
TOOL_POSITION_SOURCE: dict[str, Any] = {"get": None}


class ScriptedEeTeleop(Teleoperator):
    config_class = ScriptedEeTeleopConfig
    name = "scripted_ee"

    def __init__(self, config: ScriptedEeTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self.step = 0
        self.episode = 0

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

    def _error(self) -> np.ndarray:
        get = TOOL_POSITION_SOURCE["get"]
        if get is None:
            raise RuntimeError("scripted_ee needs a connected gen3_lite_hil / gen3_lite_hil_fake robot")
        return np.asarray(self.config.target, float) - np.asarray(get(), float)

    def get_action(self) -> dict[str, Any]:
        delta = np.clip(self._error() / self.config.ee_step_size_m, -1.0, 1.0)
        action = {"delta_x": float(delta[0]), "delta_y": float(delta[1]), "delta_z": float(delta[2])}
        if self.config.use_gripper:
            action["gripper"] = 1.0  # stay
        return action

    def get_teleop_events(self) -> dict[str, Any]:
        cfg = self.config
        allowed = cfg.max_takeover_episodes is None or self.episode < cfg.max_takeover_episodes
        in_window = any(a <= self.step < b for a, b in cfg.takeovers)
        intervening = cfg.always_intervene or (allowed and in_window)
        success = bool(np.linalg.norm(self._error()) < cfg.success_radius_m)
        self.step += 1
        if success or self.step >= cfg.episode_steps:
            self.step = 0
            self.episode += 1
        return {
            TeleopEvents.IS_INTERVENTION: intervening,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: False,
        }
