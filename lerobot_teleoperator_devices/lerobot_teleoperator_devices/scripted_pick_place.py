"""Scripted pick-and-place operator with privileged state (see ``config_scripted_pick_place``)."""

from typing import Any

import numpy as np
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

from .config_scripted_pick_place import ScriptedPickPlaceTeleopConfig
from .scripted_ee import TOOL_POSITION_SOURCE

# gen3_lite_hil_genesis registers its backend's task_state() here when it connects.
PICK_PLACE_SOURCE: dict[str, Any] = {"task": None}

CLOSE, STAY, OPEN = 0.0, 1.0, 2.0


def plan_step(cfg, finger_center, gripper, task, closing_steps: int) -> tuple[np.ndarray, float, str, int]:
    """One planner decision from the world state: (desired finger-centre position, gripper
    command, phase name, updated closing-step count). Stateless apart from the count of
    steps spent waiting for a close, so it can take over at any point of an episode."""
    fc = np.asarray(finger_center, float)
    obj = np.asarray(task["object_position"], float)
    h = float(task["object_height"])
    shelf, size = np.asarray(task["shelf_position"], float), np.asarray(task["shelf_size"], float)
    shelf_top = shelf[2] + size[2] / 2
    grasp = obj + np.array([0.0, 0.0, cfg.grasp_dz])
    xy_off = float(np.linalg.norm(fc[:2] - obj[:2]))
    near_object = xy_off < 0.025 and abs(fc[2] - grasp[2]) < 0.03
    held = gripper > cfg.gripper_closed_above and near_object

    if task["on_shelf"] and not held:
        return fc + np.array([0.0, 0.0, 0.02]), OPEN, "release", 0
    if held:
        place_obj_z = shelf_top + h / 2 + cfg.place_drop
        carry_fc_z = place_obj_z + cfg.grasp_dz + cfg.carry_clearance
        place_fc = np.array([shelf[0], shelf[1], place_obj_z + cfg.grasp_dz])
        to_place = float(np.linalg.norm(obj[:2] - shelf[:2]))
        if to_place > cfg.xy_tolerance:
            if fc[2] < carry_fc_z - 0.02:
                return np.array([fc[0], fc[1], carry_fc_z]), CLOSE, "lift", 0
            return np.array([shelf[0] + fc[0] - obj[0], shelf[1] + fc[1] - obj[1], carry_fc_z]), CLOSE, "carry", 0
        if obj[2] - place_obj_z > cfg.z_tolerance:
            return place_fc + (fc - obj - np.array([0, 0, cfg.grasp_dz])) * [1, 1, 0], CLOSE, "lower", 0
        return fc, OPEN, "open", 0
    # not holding: reach, descend, close
    if gripper > cfg.gripper_open_below and closing_steps == 0:
        return fc + np.array([0.0, 0.0, 0.01]), OPEN, "reopen", 0
    pre = grasp + np.array([0.0, 0.0, cfg.approach_height])
    if closing_steps > 0:
        if closing_steps >= cfg.close_wait_steps:  # the close did not catch it: open and retry
            return pre, OPEN, "retry", 0
        return fc, CLOSE, "close", closing_steps + 1
    if xy_off > cfg.xy_tolerance:
        if fc[2] < pre[2] - 0.02 and xy_off > 0.03:
            return np.array([fc[0], fc[1], pre[2]]), OPEN, "rise", 0
        return np.array([grasp[0], grasp[1], max(pre[2], fc[2]) if xy_off > 0.03 else pre[2]]), OPEN, "approach", 0
    if fc[2] - grasp[2] > cfg.z_tolerance:
        return np.array([grasp[0], grasp[1], grasp[2]]), OPEN, "descend", 0
    return fc, CLOSE, "close", 1


class ScriptedPickPlaceTeleop(Teleoperator):
    config_class = ScriptedPickPlaceTeleopConfig
    name = "scripted_pick_place"

    def __init__(self, config: ScriptedPickPlaceTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self.step = 0
        self.episode = 0
        self.closing_steps = 0
        self.phase = ""
        self.rng = np.random.default_rng(config.seed)

    @property
    def action_features(self) -> dict:
        return {"dtype": "float32", "shape": (4,), "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3}}

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

    def _task(self) -> dict:
        get = PICK_PLACE_SOURCE["task"]
        if get is None:
            raise RuntimeError("scripted_pick_place needs a connected gen3_lite_hil_genesis robot")
        return get()

    def _gripper(self) -> float:
        get = TOOL_POSITION_SOURCE["gripper"]
        return float(get()) if get is not None else 0.0

    def get_action(self) -> dict[str, Any]:
        cfg = self.config
        task = self._task()
        fc = np.asarray(task["finger_center"], float)
        target, gripper, self.phase, self.closing_steps = plan_step(cfg, fc, self._gripper(), task, self.closing_steps)
        delta = np.clip((target - fc) / cfg.ee_step_size_m, -1.0, 1.0)
        if cfg.noise_std > 0:
            delta = np.clip(delta + self.rng.normal(0.0, cfg.noise_std, 3), -1.0, 1.0)
        return {"delta_x": float(delta[0]), "delta_y": float(delta[1]), "delta_z": float(delta[2]), "gripper": gripper}

    def success(self) -> bool:
        task = self._task()
        return bool(task["on_shelf"]) and self._gripper() < self.config.gripper_open_below

    def get_teleop_events(self) -> dict[str, Any]:
        cfg = self.config
        allowed = cfg.max_takeover_episodes is None or self.episode < cfg.max_takeover_episodes
        in_window = any(a <= self.step < b for a, b in cfg.takeovers)
        intervening = cfg.always_intervene or (allowed and in_window)
        success = self.success()
        self.step += 1
        if success or self.step >= cfg.episode_steps:
            self.step = 0
            self.episode += 1
            self.closing_steps = 0
        return {
            TeleopEvents.IS_INTERVENTION: intervening,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: False,
        }
