"""ManiSkill backend — wraps a ManiSkill gymnasium env as a RobotBackend."""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

import numpy as np

from .config import ManiSkillBackendConfig

logger = logging.getLogger(__name__)


class ManiSkillBackend:
    """RobotBackend implementation wrapping a ManiSkill gymnasium env."""

    def __init__(self, config: ManiSkillBackendConfig):
        self.config = config
        self._env = None  # gymnasium.Env, created in connect()
        self._last_obs: dict[str, Any] | None = None
        self._last_info: dict = {}
        self._cumulative_reward: float = 0.0
        self._connected = False

    # --- Feature descriptors ---

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {
            f"{name}.pos": float for name in self.config.joint_names
        }
        if self.config.gripper_joint_name and self.config.gripper_qpos_index is not None:
            features["gripper.pos"] = float
        for cam_name, (h, w) in self.config.camera_specs.items():
            features[cam_name] = (h, w, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {
            f"{name}.pos": float for name in self.config.joint_names
        }
        if self.config.gripper_joint_name:
            features["gripper.pos"] = float
        return features

    # --- Lifecycle ---

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        import gymnasium as gym

        try:
            import mani_skill.envs  # noqa: F401 — registers ManiSkill envs
        except ImportError as e:
            raise ImportError(
                "ManiSkill is required for the ManiSkill backend. "
                "Install with: pip install mani-skill"
            ) from e

        self._env = gym.make(
            self.config.env_id,
            obs_mode=self.config.obs_mode,
            control_mode=self.config.control_mode,
            render_mode=self.config.render_mode,
            **self.config.env_kwargs,
        )
        self._connected = True
        logger.info(
            f"ManiSkill env '{self.config.env_id}' created "
            f"(obs_mode={self.config.obs_mode}, control_mode={self.config.control_mode})"
        )

    def disconnect(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self._connected = False

    # --- Episode reset ---

    def reset(self) -> dict[str, Any]:
        obs, info = self._env.reset(seed=self.config.seed)
        self._last_info = info
        self._cumulative_reward = 0.0
        self._last_obs = self._map_obs(obs)
        return self._last_obs

    # --- Observation ---

    def get_observation(self) -> dict[str, Any]:
        # ManiSkill is synchronous — return cached obs from last reset/step
        if self._last_obs is None:
            raise ValueError("No observation available. Call reset() first.")
        return self._last_obs

    # --- Action ---

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        action_np = self._map_action_to_numpy(action)
        obs, reward, terminated, truncated, info = self._env.step(action_np)
        self._last_obs = self._map_obs(obs)
        self._last_info = info
        self._cumulative_reward += float(reward)
        return action

    # --- Sim-only properties ---

    @property
    def episode_success(self) -> bool | None:
        return self._last_info.get("success", None)

    @property
    def episode_reward(self) -> float:
        return self._cumulative_reward

    # --- Private mapping methods ---

    def _map_obs(self, raw_obs) -> dict[str, Any]:
        """Convert ManiSkill obs dict/array to flat lerobot dict."""
        flat: dict[str, Any] = {}

        def _to_np(val):
            if hasattr(val, "cpu"):
                val = val.cpu().numpy()
            return np.asarray(val).squeeze()

        # Extract joint positions from agent obs
        if isinstance(raw_obs, dict) and "agent" in raw_obs:
            qpos = _to_np(raw_obs["agent"]["qpos"]).flatten()
            for i, name in enumerate(self.config.joint_names):
                if i < len(qpos):
                    flat[f"{name}.pos"] = float(qpos[i])
            # Gripper observation from qpos
            gi = self.config.gripper_qpos_index
            if self.config.gripper_joint_name and gi is not None and gi < len(qpos):
                flat["gripper.pos"] = float(qpos[gi])
        elif not isinstance(raw_obs, dict):
            # Flat state tensor/array (obs_mode="state")
            arr = _to_np(raw_obs).flatten()
            for i, name in enumerate(self.config.joint_names):
                if i < len(arr):
                    flat[f"{name}.pos"] = float(arr[i])
            gi = self.config.gripper_qpos_index
            if self.config.gripper_joint_name and gi is not None and gi < len(arr):
                flat["gripper.pos"] = float(arr[gi])

        # Extract environment state from obs["extra"]
        if isinstance(raw_obs, dict) and "extra" in raw_obs:
            env_state_parts = []
            for key in self.config.env_state_keys:
                if key in raw_obs["extra"]:
                    env_state_parts.append(_to_np(raw_obs["extra"][key]).flatten())
            if env_state_parts:
                flat["env_state"] = np.concatenate(env_state_parts).astype(np.float32)

        # Extract camera images (ManiSkill v3 uses "sensor_data", v2 uses "image")
        if isinstance(raw_obs, dict):
            image_dict = raw_obs.get("sensor_data") or raw_obs.get("image") or {}
            for our_name, ms_name in self.config.camera_mapping.items():
                if ms_name in image_dict:
                    img = image_dict[ms_name].get("rgb")
                    if img is not None:
                        img = _to_np(img)
                        flat[our_name] = np.asarray(img, dtype=np.uint8)

        return flat

    def _map_action_to_numpy(self, action: dict[str, Any]) -> np.ndarray:
        """Convert flat lerobot action dict to numpy array for env.step()."""
        values = [float(action[f"{name}.pos"]) for name in self.config.joint_names]
        if self.config.gripper_joint_name:
            values.append(float(action.get("gripper.pos", 0.0)))
        return np.array(values, dtype=np.float32)
