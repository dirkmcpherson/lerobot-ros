"""Evaluation blocks: every ``interval_steps`` environment steps, the next
``block_episodes`` episodes are evaluation episodes (operator hands off, takeover only
for safety). An evaluation episode with a takeover counts as a failure."""

from dataclasses import dataclass, field


@dataclass
class EvalSchedule:
    interval_steps: int = 2000
    block_episodes: int = 10
    enabled: bool = True

    next_eval_at: int = field(init=False)
    remaining: int = field(default=0, init=False)
    block_index: int = field(default=0, init=False)
    current_is_eval: bool = field(default=False, init=False)
    _block: list[dict] = field(default_factory=list, init=False)
    _block_start_steps: int = field(default=0, init=False)

    def __post_init__(self):
        if self.interval_steps <= 0 or self.block_episodes <= 0:
            raise ValueError("interval_steps and block_episodes must be positive")
        self.next_eval_at = self.interval_steps

    def begin_episode(self, env_steps: int) -> bool:
        """Call before each episode with the environment steps done so far."""
        if not self.enabled:
            self.current_is_eval = False
        elif self.remaining > 0:
            self.current_is_eval = True
        elif env_steps >= self.next_eval_at:
            self.remaining = self.block_episodes
            self._block, self._block_start_steps = [], env_steps
            while self.next_eval_at <= env_steps:
                self.next_eval_at += self.interval_steps
            self.current_is_eval = True
        else:
            self.current_is_eval = False
        return self.current_is_eval

    def end_episode(self, success: bool, intervened: bool, length: int, env_steps: int) -> dict | None:
        """Call after each episode. Returns the block summary when a block completes."""
        if not self.current_is_eval:
            return None
        self.current_is_eval = False
        self._block.append({"success": bool(success) and not intervened, "intervened": bool(intervened),
                            "length": int(length)})
        self.remaining -= 1
        if self.remaining > 0:
            return None
        n = len(self._block)
        successes = sum(e["success"] for e in self._block)
        summary = {
            "block": self.block_index,
            "env_steps_at_start": self._block_start_steps,
            "env_steps_at_end": env_steps,
            "episodes": n,
            "successes": successes,
            "success_rate": successes / n,
            "episodes_with_takeover": sum(e["intervened"] for e in self._block),
            "mean_length": sum(e["length"] for e in self._block) / n,
        }
        self.block_index += 1
        self._block = []
        return summary
