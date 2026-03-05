"""Gym environment adapter — wraps any BackendRobot as a gymnasium.Env."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .robot import BackendRobot

# Large bound for unbounded scalar observations/actions
_INF = 1e6


class RobotGymEnv(gym.Env):
    """Wraps a :class:`BackendRobot` as a ``gymnasium.Env``.

    This allows any backend (ROS2, ManiSkill, etc.) to be used with
    lerobot's built-in eval pipeline (``rollout()`` / ``eval_policy()``).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, robot: BackendRobot, max_episode_steps: int = 200):
        super().__init__()
        self.robot = robot
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        obs_features = robot.observation_features
        action_features = robot.action_features

        # Separate float (state) keys from image keys
        self._obs_float_keys = [k for k, v in obs_features.items() if v is float]
        self._obs_image_keys = [k for k, v in obs_features.items() if isinstance(v, tuple)]
        self._action_keys = list(action_features.keys())

        self.observation_space = self._build_obs_space(obs_features)
        self.action_space = self._build_action_space(action_features)

    # --- Gym interface ---

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict = self.robot.backend.reset()
        self._step_count = 0
        return self._obs_dict_to_gym(obs_dict), {}

    def step(self, action: np.ndarray):
        action_dict = {k: float(action[i]) for i, k in enumerate(self._action_keys)}
        self.robot.send_action(action_dict)
        obs_dict = self.robot.get_observation()
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        reward = 0.0
        info: dict[str, Any] = {}

        # Pull success/reward from backend if available
        backend = self.robot.backend
        if backend.episode_success is not None:
            success = backend.episode_success
            info["is_success"] = success
            terminated = bool(success)
            reward = float(backend.episode_reward)

        # lerobot eval expects this structure
        info["final_info"] = [{"is_success": info.get("is_success", False)}]

        return self._obs_dict_to_gym(obs_dict), reward, terminated, truncated, info

    # --- Space builders ---

    def _build_obs_space(self, features: dict) -> spaces.Dict:
        sub: dict[str, spaces.Space] = {}
        # State vector
        if self._obs_float_keys:
            n = len(self._obs_float_keys)
            sub["state"] = spaces.Box(-_INF, _INF, shape=(n,), dtype=np.float32)
        # Image spaces
        for key in self._obs_image_keys:
            shape = features[key]  # (H, W, 3)
            sub[key] = spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        return spaces.Dict(sub)

    def _build_action_space(self, features: dict) -> spaces.Box:
        n = len(features)
        return spaces.Box(-_INF, _INF, shape=(n,), dtype=np.float32)

    # --- Conversion helpers ---

    def _obs_dict_to_gym(self, obs_dict: dict) -> dict:
        gym_obs: dict[str, Any] = {}
        if self._obs_float_keys:
            gym_obs["state"] = np.array(
                [obs_dict.get(k, 0.0) for k in self._obs_float_keys],
                dtype=np.float32,
            )
        for key in self._obs_image_keys:
            img = obs_dict.get(key)
            if img is not None:
                gym_obs[key] = np.asarray(img, dtype=np.uint8)
            else:
                shape = self.robot.observation_features[key]
                gym_obs[key] = np.zeros(shape, dtype=np.uint8)
        return gym_obs
