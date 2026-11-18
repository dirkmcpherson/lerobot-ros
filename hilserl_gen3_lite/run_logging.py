"""CSV logging for HIL-SERL runs, and the rise-time summary.

Files under ``<output_dir>/`` (lerobot's run directory):

    run_meta_<role>.json   full config, git commit, seed, start time (one per process)
    episodes.csv           one row per episode (actor)
    eval.csv               one row per evaluation block (actor)
    train.csv              one row per training log step (learner)

The repo has no experiment tracker of its own; lerobot's wandb logging still works when
``wandb.enable`` is set, these CSVs are written either way.
"""

import csv
import json
import subprocess
import time
from pathlib import Path

EPISODE_FIELDS = [
    "episode", "env_steps", "train_env_steps", "wall_time_s", "success", "episode_reward",
    "length", "intervention_steps", "intervention_fraction", "takeovers", "is_eval",
]
EVAL_FIELDS = [
    "block", "env_steps_at_start", "env_steps_at_end", "wall_time_s", "episodes", "successes",
    "success_rate", "episodes_with_takeover", "mean_length",
]


class CsvLog:
    """Append-only CSV with a fixed header, flushed on every row."""

    def __init__(self, path: Path, fields: list[str]):
        self.path, self.fields = Path(path), list(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._file = open(self.path, "a", newline="")  # noqa: SIM115
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields, extrasaction="ignore")
        if new:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fields})
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def read_csv(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def git_state(repo_dir: Path | None = None) -> dict:
    repo_dir = repo_dir or Path(__file__).resolve().parent.parent

    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
        except Exception:
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None}


def save_run_meta(output_dir: Path, role: str, cfg, extra: dict | None = None) -> Path:
    """Write the full config, git commit and seed next to the run's logs."""
    import lerobot

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        config = cfg.to_dict()
    except Exception:
        import draccus

        config = draccus.encode(cfg)
    meta = {
        "role": role,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": getattr(cfg, "seed", None),
        "git": git_state(),
        "lerobot_version": lerobot.__version__,
        "config": config,
        **(extra or {}),
    }
    path = output_dir / f"run_meta_{role}.json"
    if path.exists():  # a resumed run keeps the earlier record
        path = output_dir / f"run_meta_{role}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    return path


def rise_time(eval_rows: list[dict], thresholds=(0.5, 0.9)) -> dict:
    """Environment steps at which evaluation success first reached each threshold
    (None if never), and the final evaluation success rate."""
    out: dict = {f"env_steps_to_{int(t * 100)}pct": None for t in thresholds}
    for row in eval_rows:
        rate, steps = float(row["success_rate"]), int(float(row["env_steps_at_start"]))
        for t in thresholds:
            key = f"env_steps_to_{int(t * 100)}pct"
            if out[key] is None and rate >= t:
                out[key] = steps
    out["final_eval_success_rate"] = float(eval_rows[-1]["success_rate"]) if eval_rows else None
    out["eval_blocks"] = len(eval_rows)
    return out
