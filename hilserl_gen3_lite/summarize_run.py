"""Summarise a HIL-SERL run directory: the rise-time numbers, and curves if matplotlib
is installed.

    python -m hilserl_gen3_lite.summarize_run runs/reach_baseline_s0 [more run dirs ...]

Prints, per run: environment steps to first reach 50 % and 90 % evaluation success, the
final evaluation success rate, and totals. Writes ``summary.json`` and (with matplotlib)
``curves.png`` into each run directory.
"""

import json
import sys
from pathlib import Path

from .run_logging import read_csv, rise_time


def _f(rows, key):
    return [float(r[key]) for r in rows if r.get(key) not in ("", None)]


def rolling(values, window):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def summarize(run_dir: Path, window: int = 20) -> dict:
    run_dir = Path(run_dir)
    episodes = read_csv(run_dir / "episodes.csv")
    evals = read_csv(run_dir / "eval.csv")
    train = read_csv(run_dir / "train.csv")
    summary = rise_time(evals)
    training = [e for e in episodes if e["is_eval"] == "0"]
    summary.update(
        run=str(run_dir),
        episodes=len(episodes),
        env_steps=int(float(episodes[-1]["env_steps"])) if episodes else 0,
        wall_time_s=float(episodes[-1]["wall_time_s"]) if episodes else 0.0,
        mean_intervention_fraction=(sum(_f(training, "intervention_fraction")) / len(training)) if training else None,
        last_training_success_rate=(sum(_f(training[-window:], "success")) / len(training[-window:]))
        if training
        else None,
        optimization_steps=int(float(train[-1]["optimization_step"])) if train else 0,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return summary

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    steps = _f(training, "env_steps")
    ax = axes[0, 0]
    ax.plot(steps, rolling(_f(training, "success"), window), label=f"training episodes (rolling {window})")
    if evals:
        ax.plot(_f(evals, "env_steps_at_start"), _f(evals, "success_rate"), "o-", label="evaluation blocks")
    ax.axhline(0.5, ls=":", c="gray")
    ax.axhline(0.9, ls=":", c="gray")
    ax.set(title="Success rate", xlabel="environment steps", ylim=(-0.05, 1.05))
    ax.legend()
    axes[0, 1].plot(steps, rolling(_f(training, "intervention_fraction"), window))
    axes[0, 1].set(title=f"Fraction of steps under human control (rolling {window})", xlabel="environment steps")
    axes[0, 2].plot(steps, rolling(_f(training, "length"), window))
    axes[0, 2].set(title=f"Episode length (rolling {window})", xlabel="environment steps")
    opt = _f(train, "optimization_step")
    for key in ("loss_critic", "loss_actor"):
        vals = _f(train, key)
        axes[1, 0].plot(opt[: len(vals)], vals, label=key)
    axes[1, 0].set(title="Losses", xlabel="optimization steps", yscale="symlog")
    axes[1, 0].legend()
    for key in ("mean_q", "temperature"):
        vals = _f(train, key)
        axes[1, 1].plot(opt[: len(vals)], vals, label=key)
    axes[1, 1].set(title="Mean Q and temperature", xlabel="optimization steps")
    axes[1, 1].legend()
    for key in ("batch_frac_demo", "batch_frac_intervention", "batch_frac_policy"):
        vals = _f(train, key)
        axes[1, 2].plot(opt[: len(vals)], rolling(vals, 10), label=key.replace("batch_frac_", ""))
    axes[1, 2].set(title="Share of the batch by source (rolling 10)", xlabel="optimization steps", ylim=(0, 1))
    axes[1, 2].legend()
    fig.suptitle(str(run_dir))
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=110)
    plt.close(fig)
    return summary


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for run_dir in sys.argv[1:]:
        print(json.dumps(summarize(Path(run_dir)), indent=2))


if __name__ == "__main__":
    main()
