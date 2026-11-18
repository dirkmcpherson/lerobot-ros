"""Actor for HIL-SERL on the Gen3 Lite: lerobot's actor, unmodified, with

  * this repo's environment and processors in place of ``gym_manipulator``'s
    (see ``env.py`` for why),
  * provenance tags added to each episode's transitions before they are sent,
  * per-episode and per-evaluation-block CSV logs.

Usage (the operator runs this; it moves the arm):

    python -m hilserl_gen3_lite.actor --config_path=config/hilserl/train_gen3_lite_reach.json \
        --run.dir=runs/reach_baseline_s0 --run.eval_interval_steps=2000 --run.eval_episodes=10
"""

import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

from .env import make_processors, make_robot_env
from .eval_schedule import EvalSchedule
from .provenance import IS_INTERVENTION, tag_episode
from .run_logging import EPISODE_FIELDS, EVAL_FIELDS, CsvLog, read_csv, rise_time, save_run_meta
from .settings import RunSettings, pop_run_settings, require_motion_confirmation, run_dir_and_argv


class ActorRun:
    """Episode bookkeeping for one actor process."""

    def __init__(self, output_dir, settings: RunSettings, say=None):
        self.output_dir = Path(output_dir)
        self.settings = settings
        self._say = say or (lambda text: None)
        self.schedule = EvalSchedule(settings.eval_interval_steps, settings.eval_episodes, settings.eval_enabled)
        self.episodes = CsvLog(self.output_dir / "episodes.csv", EPISODE_FIELDS)
        self.evals = CsvLog(self.output_dir / "eval.csv", EVAL_FIELDS)
        self._state_path = self.output_dir / "actor_state.json"
        self.episode_index = 0
        self.env_steps = 0
        self.train_env_steps = 0
        self.elapsed_before = 0.0
        if self._state_path.exists():  # actor restarted within the same run: keep counting
            state = json.loads(self._state_path.read_text())
            logging.info(f"[ACTOR] Continuing run {self.output_dir} from {state}")
            self.episode_index, self.env_steps = state["episode_index"], state["env_steps"]
            self.train_env_steps, self.elapsed_before = state["train_env_steps"], state["wall_time_s"]
            self.schedule.block_index = state["eval_block_index"]
            self.schedule.next_eval_at = state["next_eval_at"]
        self._t0 = time.monotonic()
        self._was_eval = False

    @property
    def wall_time_s(self) -> float:
        return self.elapsed_before + time.monotonic() - self._t0

    def on_reset(self) -> None:
        """Called by the environment at the start of every reset."""
        is_eval = self.schedule.begin_episode(self.env_steps)
        if is_eval and not self._was_eval:
            msg = f"Evaluation block: {self.settings.eval_episodes} episodes. Hands off; take over only for safety."
            logging.info(f"[ACTOR] {msg}")
            self._say("Evaluation. Hands off.")
        elif self._was_eval and not is_eval:
            logging.info("[ACTOR] Evaluation block finished; back to training with interventions.")
            self._say("Training.")
        self._was_eval = is_eval

    def on_episode(self, transitions: list[dict]) -> list[dict]:
        """Tag and log one finished episode. Returns the tagged transitions."""
        is_eval = self.schedule.current_is_eval
        tag_episode(transitions, self.episode_index)
        infos = [t["complementary_info"] for t in transitions]
        length = len(transitions)
        human = sum(int(i[IS_INTERVENTION]) for i in infos)
        takeovers = sum(int(i["takeover_onset"]) for i in infos)
        rewards = [float(t["reward"]) for t in transitions]
        success = max(rewards) > 0.0

        self.env_steps += length
        if not is_eval:
            self.train_env_steps += length
        self.episodes.write(
            {
                "episode": self.episode_index,
                "env_steps": self.env_steps,
                "train_env_steps": self.train_env_steps,
                "wall_time_s": round(self.wall_time_s, 2),
                "success": int(success),
                "episode_reward": round(sum(rewards), 4),
                "length": length,
                "intervention_steps": human,
                "intervention_fraction": round(human / length, 4),
                "takeovers": takeovers,
                "is_eval": int(is_eval),
            }
        )
        block = self.schedule.end_episode(success, human > 0, length, self.env_steps)
        if block is not None:
            block["wall_time_s"] = round(self.wall_time_s, 2)
            self.evals.write(block)
            summary = rise_time(read_csv(self.evals.path))
            (self.output_dir / "rise_time.json").write_text(json.dumps(summary, indent=2))
            logging.info(
                f"[ACTOR] Evaluation block {block['block']}: {block['successes']}/{block['episodes']} "
                f"at {block['env_steps_at_start']} env steps. {summary}"
            )
        self.episode_index += 1
        self._state_path.write_text(
            json.dumps(
                {
                    "episode_index": self.episode_index,
                    "env_steps": self.env_steps,
                    "train_env_steps": self.train_env_steps,
                    "wall_time_s": self.wall_time_s,
                    "eval_block_index": self.schedule.block_index,
                    "next_eval_at": self.schedule.next_eval_at,
                }
            )
        )
        return transitions


def close_env(env, teleop) -> None:
    """Log the robot's limit counters, hold and disconnect it, then the teleoperator; never raises."""
    counters = getattr(getattr(env, "robot", None), "counters", None)
    if counters is not None:
        logging.info(f"[ACTOR] robot limit counters: {counters()}")
    for name, closer in (("environment", getattr(env, "close", None)), ("teleop", getattr(teleop, "disconnect", None))):
        if closer is None:
            continue
        try:
            closer()
            logging.info(f"[ACTOR] {name} closed")
        except Exception as e:
            logging.error(f"[ACTOR] closing the {name} failed: {e}")


def install(settings: RunSettings, run_dir: Path) -> None:
    """Point lerobot's actor at this repo's environment and add the episode hooks."""
    from . import compat

    compat.apply()
    import lerobot_robot_ros  # noqa: F401  (registers gen3_lite_hil)
    import lerobot_teleoperator_devices  # noqa: F401  (registers scripted_ee)
    from lerobot.rl import actor as upstream
    from lerobot.utils.utils import log_say

    state: dict = {"run": None}
    original_act = upstream.act_with_policy
    original_push = upstream.push_transitions_to_transport_queue

    def act_with_policy(cfg, *args, **kwargs):
        save_run_meta(run_dir, "actor", cfg, {"run_settings": dataclasses.asdict(settings)})
        say = (lambda text: log_say(text, play_sounds=True)) if settings.announce else None
        state["run"] = ActorRun(run_dir, settings, say=say)
        try:
            return original_act(cfg, *args, **kwargs)
        finally:
            # Upstream's actor never closes its environment. Without this the arm is only
            # stopped by the watchdog, and the process can hang at exit with the ROS
            # executor thread still alive (seen in Gazebo).
            close_env(state.get("env"), state.get("teleop"))

    def _make_robot_env(cfg):
        require_motion_confirmation(cfg.robot, settings)
        run = state["run"]
        state["env"], state["teleop"] = make_robot_env(cfg, on_reset=run.on_reset if run is not None else None)
        return state["env"], state["teleop"]

    def push_transitions(transitions, transitions_queue):
        if state["run"] is not None:
            state["run"].on_episode(transitions)
        return original_push(transitions=transitions, transitions_queue=transitions_queue)

    upstream.act_with_policy = act_with_policy
    upstream.make_robot_env = _make_robot_env
    upstream.make_processors = make_processors
    upstream.push_transitions_to_transport_queue = push_transitions


def main() -> None:
    settings, argv = pop_run_settings()
    run_dir, sys.argv = run_dir_and_argv(settings, argv, f"actor_{time.strftime('%Y%m%d_%H%M%S')}")
    install(settings, run_dir)
    from lerobot.rl import actor as upstream

    upstream.actor_cli()


if __name__ == "__main__":
    main()
