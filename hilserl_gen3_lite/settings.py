"""Run settings that lerobot's config does not have (evaluation blocks), taken from
``--run.<name>=<value>`` command-line arguments and removed before lerobot parses the rest."""

import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunSettings:
    # Directory for this run's logs (episodes.csv, eval.csv, train.csv, run_meta_*.json).
    # Give the learner and the actor the same one. Required.
    dir: str = ""
    # Must be true for anything that can move the real arm (not needed for gen3_lite_hil_fake).
    confirm_motion: bool = False
    # An evaluation block starts every eval_interval_steps environment steps...
    eval_interval_steps: int = 2000
    # ...and lasts this many episodes (hands off; takeover only for safety).
    eval_episodes: int = 10
    eval_enabled: bool = True
    # Speak "evaluation" / "training" at block boundaries (lerobot's log_say).
    announce: bool = True


def _convert(value: str, kind: type):
    if kind is bool:
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {value}")
    return kind(value)


def pop_run_settings(argv: list[str] | None = None) -> tuple[RunSettings, list[str]]:
    """Split ``--run.*`` arguments off ``argv``. Returns (settings, remaining argv)."""
    argv = list(sys.argv if argv is None else argv)
    settings = RunSettings()
    fields = {f.name: f.type for f in dataclasses.fields(RunSettings)}
    rest = []
    for arg in argv:
        if not arg.startswith("--run."):
            rest.append(arg)
            continue
        name, sep, value = arg[len("--run.") :].partition("=")
        if not sep or name not in fields:
            raise SystemExit(f"Unknown or malformed run setting '{arg}'. Known: {sorted(fields)} (use --run.name=value)")
        setattr(settings, name, _convert(value, type(getattr(settings, name))))
    return settings, rest


def run_dir_and_argv(settings: RunSettings, argv: list[str], role_subdir: str) -> tuple[Path, list[str]]:
    """Resolve the run directory and give lerobot its own output_dir inside it.

    lerobot refuses to start in an output_dir that already exists (unless resuming), so
    the learner and the actor cannot share one; each gets a subdirectory of the run dir.
    """
    if not settings.dir:
        raise SystemExit("Pass --run.dir=runs/<name> (the same for learner and actor).")
    run_dir = Path(settings.dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not any(a.startswith("--output_dir") for a in argv):
        argv = [*argv, f"--output_dir={run_dir / role_subdir}"]
    return run_dir, argv


def require_motion_confirmation(robot_cfg, settings: RunSettings) -> None:
    """Print the robot's limits and refuse to continue on a real arm without --run.confirm_motion=true."""
    robot_type = getattr(robot_cfg, "type", "?")
    summary = robot_cfg.limits_summary() if hasattr(robot_cfg, "limits_summary") else "  (no limits summary)"
    print(f"\nRobot '{robot_type}' limits:\n{summary}\n", flush=True)
    if str(robot_type).endswith("_fake"):
        return
    if not settings.confirm_motion:
        raise SystemExit(
            "This command moves the real arm. Check the limits above, keep a hand on the emergency stop, "
            "and re-run with --run.confirm_motion=true."
        )
