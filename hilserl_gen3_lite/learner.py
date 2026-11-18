"""Learner for HIL-SERL on the Gen3 Lite: lerobot's learner, unmodified, with

  * a check that both replay buffers hold the same complementary_info keys, so the
    provenance fields stay aligned in the sampled batch (see ``provenance.py``),
  * a CSV row per training log step: losses, mean Q, temperature, both buffer sizes
    and the share of the batch from each source.

Nothing here changes what is sampled or how losses are computed. Batch construction
stays in one place, ``OnlineOfflineMixer.sample`` / ``get_iterator`` (subclassed here
only to look at the batches it produces), which is where a different mix would go.

    python -m hilserl_gen3_lite.learner --config_path=config/hilserl/train_gen3_lite_reach.json \
        --run.dir=runs/reach_baseline_s0
"""

import logging
import sys
import time
from pathlib import Path

import torch

from .provenance import SOURCE_NAMES
from .run_logging import CsvLog, save_run_meta
from .settings import pop_run_settings, run_dir_and_argv

TRAIN_FIELDS = [
    "optimization_step", "wall_time_s", "loss_critic", "mean_q", "loss_actor", "loss_temperature",
    "temperature", "loss_discrete_critic", "critic_grad_norm", "actor_grad_norm",
    "online_buffer_size", "offline_buffer_size", "batch_size",
    "batch_frac_demo", "batch_frac_intervention", "batch_frac_policy", "batch_frac_is_intervention",
]


def batch_source_fractions(batch: dict) -> dict[str, float]:
    """Share of a sampled batch from each source, from its complementary_info."""
    info = batch.get("complementary_info") or {}
    out = {}
    n = int(batch["reward"].shape[0])
    source = info.get("source")
    if source is not None and source.shape[0] == n:
        for code, name in SOURCE_NAMES.items():
            out[f"batch_frac_{name}"] = float((source.round() == code).float().mean())
    flag = info.get("is_intervention")
    if flag is not None and flag.shape[0] == n:
        out["batch_frac_is_intervention"] = float((flag > 0.5).float().mean())
    out["batch_size"] = n
    return out


def check_buffer_keys(online_buffer, offline_buffer) -> None:
    """Both buffers must store the same complementary_info keys (once both hold data)."""
    if offline_buffer is None or not (online_buffer.initialized and offline_buffer.initialized):
        return
    online, offline = set(online_buffer.complementary_info_keys), set(offline_buffer.complementary_info_keys)
    if online != offline:
        raise RuntimeError(
            "The online and offline replay buffers store different complementary_info keys, so provenance "
            f"would be misaligned in the mixed batch. Online only: {sorted(online - offline)}; offline only: "
            f"{sorted(offline - online)}. Record the demo dataset with hilserl_gen3_lite.record (it writes "
            "the same columns the actor sends), or add the columns with hilserl_gen3_lite.tag_dataset."
        )


def install(run_dir: Path) -> None:
    from . import compat

    compat.apply()
    import lerobot_robot_ros  # noqa: F401
    import lerobot_teleoperator_devices  # noqa: F401
    from lerobot.rl import learner as upstream

    state: dict = {"cfg": None, "log": None, "t0": time.monotonic(), "mixer": None}
    original_train = upstream.train

    class ObservedMixer(upstream.OnlineOfflineMixer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_batch = None
            self._checked = False
            state["mixer"] = self

        def _observe(self, batch):
            self.last_batch = batch
            return batch

        def _check(self):
            if not self._checked:
                check_buffer_keys(self.online_buffer, self.offline_buffer)
                self._checked = self.offline_buffer is None or self.offline_buffer.initialized

        def sample(self, batch_size):
            self._check()
            return self._observe(super().sample(batch_size))

        def get_iterator(self, batch_size, async_prefetch=True, queue_size=2):
            self._check()
            for batch in super().get_iterator(batch_size, async_prefetch=async_prefetch, queue_size=queue_size):
                yield self._observe(batch)

    class LoggedTrainer(upstream.RLTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if state["cfg"] is not None:
                save_run_meta(run_dir, "learner", state["cfg"])

        def training_step(self):
            stats = super().training_step()
            cfg, mixer = state["cfg"], state["mixer"]
            step = self.algorithm.optimization_step
            if cfg is None or mixer is None or mixer.last_batch is None or step % cfg.log_freq != 0:
                return stats
            batch = mixer.last_batch
            extra = batch_source_fractions(batch)
            extra["mean_q"] = mean_q(self.algorithm, batch)
            stats.extra.update(extra)  # reaches wandb too, through lerobot's own logging
            row = stats.to_log_dict()
            row.update(
                optimization_step=step,
                wall_time_s=round(time.monotonic() - state["t0"], 2),
                critic_grad_norm=stats.grad_norms.get("critic", ""),
                actor_grad_norm=stats.grad_norms.get("actor", ""),
                online_buffer_size=len(mixer.online_buffer),
                offline_buffer_size=len(mixer.offline_buffer) if mixer.offline_buffer is not None else 0,
            )
            if state["log"] is None:
                state["log"] = CsvLog(run_dir / "train.csv", TRAIN_FIELDS)
            state["log"].write(row)
            return stats

    def train(cfg, job_name=None):
        state["cfg"] = cfg
        state["t0"] = time.monotonic()
        return original_train(cfg, job_name=job_name)

    upstream.OnlineOfflineMixer = ObservedMixer
    upstream.RLTrainer = LoggedTrainer
    upstream.train = train


def mean_q(algorithm, batch) -> float:
    """Mean critic prediction on a batch (logging only; an extra no-grad forward pass)."""
    try:
        with torch.no_grad():
            fb = algorithm._prepare_forward_batch(batch, include_complementary_info=False)
            actions = fb["action"]
            if algorithm.policy_config.num_discrete_actions is not None:
                actions = actions[:, :-1]  # critics see the continuous part only
            q = algorithm._critic_forward(
                fb["state"], actions, use_target=False, observation_features=fb.get("observation_feature")
            )
        return float(q.mean())
    except Exception as e:  # never let a logging extra stop training
        logging.debug(f"mean_q failed: {e}")
        return float("nan")


def main() -> None:
    settings, argv = pop_run_settings()  # the learner takes the same --run.* arguments as the actor
    run_dir, sys.argv = run_dir_and_argv(settings, argv, "learner")
    install(run_dir)
    from lerobot.rl import learner as upstream

    upstream.train_cli()


if __name__ == "__main__":
    main()
