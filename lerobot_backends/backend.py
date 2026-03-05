"""RobotBackend Protocol — the interface every robot backend must implement."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RobotBackend(Protocol):
    """Interface every robot backend must implement.

    Backends encapsulate all hardware/simulator-specific logic so that
    recording, evaluation, and gym wrapping code can be written once.
    """

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        """Flat dict describing observations.

        Keys like ``'joint_1.pos'`` map to ``float``;
        keys like ``'camera_rgb'`` map to ``(H, W, 3)``.
        Must be callable before ``connect()``.
        """
        ...

    @property
    def action_features(self) -> dict[str, type]:
        """Flat dict describing actions.

        Keys like ``'joint_1.pos'`` map to ``float``.
        Must be callable before ``connect()``.
        """
        ...

    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None:
        """Initialize hardware / sim. Idempotent — raise if already connected."""
        ...

    def disconnect(self) -> None:
        """Release resources."""
        ...

    def reset(self) -> dict[str, Any]:
        """Reset to episode start. Returns initial observation.

        - ROS2: move to home position, wait, return obs
        - ManiSkill: env.reset(), return mapped obs
        """
        ...

    def get_observation(self) -> dict[str, Any]:
        """Return current state as flat dict matching *observation_features*.

        Includes joint positions, camera images, any extra state.
        """
        ...

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute *action*. Returns the action actually applied.

        - ROS2: publish to controllers, return action
        - ManiSkill: env.step(mapped_action), store new obs, return action
        """
        ...

    # --- Optional (sim-only, with defaults) ---

    @property
    def episode_success(self) -> bool | None:
        """``True``/``False`` if env can determine success, ``None`` otherwise."""
        return None

    @property
    def episode_reward(self) -> float:
        """Cumulative reward for current episode. ``0.0`` for real hardware."""
        return 0.0
