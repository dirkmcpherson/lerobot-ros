"""Interactive workspace setup: teach the tool box, the no-go boxes and the episode start
pose by driving the arm, try them out live, and save them into the task's config files.
THIS MOVES THE ARM (while the gamepad's takeover button is held).

    python -m hilserl_gen3_lite.workspace --config_path=config/hilserl/env_gen3_lite_cup.json \
        --run.confirm_motion=true

You drive with the gamepad and type short commands in this terminal (Enter after each):

    w               teach the tool box: sweep the tool through the whole region the task
                    needs (pick spot, place spot, the top of the carry, the lowest point of
                    the grasp), then `done`. The box is what you swept plus the padding.
    n NAME          teach a no-go box: bring the tool tip NEAR (not onto) the extremes of
                    the object and press the gamepad's success button (or type `p`) at
                    each; `done` makes the box that contains the points plus the padding.
                    No moving part of the arm, fingers included, may then come within the
                    no-go margin of it.
    n NAME x0 y0 z0 x1 y1 z1
                    a no-go box from measured coordinates (base frame, metres), for
                    things the tool cannot get near.
    yaw DEG         turn the gripper left (+) or right (-) about the vertical, on the spot
    pitch DEG       tilt the gripper's pointing direction down (+) or up (-)
    roll DEG        rotate the gripper about the direction it points
                    (these turn slowly, 15 degrees per second, through all the limits;
                    the policy cannot turn the tool, so this is where its orientation
                    for the whole task is chosen)
    r               make the current pose the episode start pose. The tool's roll, pitch
                    and yaw are frozen at this pose's orientation.
    p               add the current tool position as a point (same as the success button)
    done / cancel   finish / abandon the box being taught
    del NAME        delete a no-go box
    open            widen the tool box to the workspace guard box (to explore again)
    pad METRES      padding added around taught boxes (default 0.02 tool box, 0.03 no-go)
    show            what is set now; also writes a top and side view to runs/workspace.png
    save            write to the config files (both the env_ and the train_ file)
    quit            leave without saving anything not yet saved
    (empty line)    print the current tool position, joints and gripper

Everything takes effect immediately, so a box can be tested as soon as it is taught: drive
at a wall of the tool box and the tool stops there; drive at a no-go box and the arm
holds. The session starts with the tool box opened up to the workspace guard box. The
table plane, the guard box and self-collision are enforced throughout, as always.

Going near real objects with a real arm: approach slowly, stop two or three centimetres
short, and let the padding cover the rest. The boxes should err outward.
"""

import json
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import parser
from lerobot.processor import TransitionKey
from lerobot.rl.gym_manipulator import (
    GymManipulatorConfig,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep

from .env import make_processors, make_robot_env
from .settings import RunSettings, pop_run_settings, require_motion_confirmation

_SETTINGS = RunSettings()
GUARD_INSET_M = 0.01  # the tool box is kept this far inside the workspace guard box
TURN_STEP_RAD = np.deg2rad(1.5)  # per control step: 15 degrees per second at 10 Hz


def axis_rotation(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def describe_orientation(rot: np.ndarray) -> str:
    """Where the gripper points (the tool frame's z axis), in words and numbers."""
    approach = rot[:, 2]
    heading = np.degrees(np.arctan2(approach[1], approach[0]))  # 0 = straight ahead (+x), + = to the left
    elevation = np.degrees(np.arcsin(np.clip(approach[2], -1, 1)))  # + = up, -90 = straight down
    if elevation < -75:
        words = "pointing straight down"
    elif elevation > 75:
        words = "pointing straight up"
    else:
        side = "straight ahead" if abs(heading) < 10 else f"{abs(heading):.0f} deg to the {'left' if heading > 0 else 'right'} of straight ahead"
        tilt = "level" if abs(elevation) < 10 else f"tilted {abs(elevation):.0f} deg {'down' if elevation < 0 else 'up'}"
        words = f"{tilt}, {side}"
    return f"gripper {words} (pointing along {np.round(approach, 2).tolist()})"


def bounding_box(points, pad: float) -> tuple[list[float], list[float]]:
    pts = np.asarray(points, float).reshape(-1, 3)
    return np.round(pts.min(0) - pad, 3).tolist(), np.round(pts.max(0) + pad, 3).tolist()


@dataclass
class WorkspaceDraft:
    """What is being taught. Pure bookkeeping: no robot, no I/O."""

    guard_min: list[float]
    guard_max: list[float]
    tool_min: list[float]
    tool_max: list[float]
    no_go: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)
    reset_joints: list[float] | None = None
    tool_pad: float = 0.02
    no_go_pad: float = 0.03
    capture: str | None = None  # None, "tool" or the name of the no-go box being taught
    points: list[list[float]] = field(default_factory=list)
    dirty: bool = False  # changed since the last save

    def widest_tool_box(self) -> tuple[list[float], list[float]]:
        lo = (np.asarray(self.guard_min) + GUARD_INSET_M).round(3).tolist()
        hi = (np.asarray(self.guard_max) - GUARD_INSET_M).round(3).tolist()
        return lo, hi

    def open_tool_box(self) -> None:
        self.tool_min, self.tool_max = self.widest_tool_box()

    def begin(self, what: str) -> None:
        self.capture, self.points = what, []

    def add_point(self, position) -> int:
        self.points.append([float(v) for v in position])
        return len(self.points)

    def cancel(self) -> None:
        self.capture, self.points = None, []

    def finish(self) -> str:
        """Turn the captured points into a box. Returns a description; raises ValueError."""
        if self.capture is None:
            raise ValueError("nothing is being taught (use `w` or `n NAME` first)")
        if not self.points:
            raise ValueError("no points yet")
        what = self.capture
        if what == "tool":
            lo, hi = bounding_box(self.points, self.tool_pad)
            wide_lo, wide_hi = self.widest_tool_box()
            lo, hi = np.maximum(lo, wide_lo).round(3).tolist(), np.minimum(hi, wide_hi).round(3).tolist()
            if not all(a < b for a, b in zip(lo, hi, strict=True)):
                raise ValueError(f"swept region {lo} .. {hi} is empty inside the guard box")
            self.tool_min, self.tool_max = lo, hi
            message = f"tool box set to {lo} .. {hi} (from {len(self.points)} positions, padding {self.tool_pad} m)"
        else:
            lo, hi = bounding_box(self.points, self.no_go_pad)
            self.no_go[what] = (lo, hi)
            message = f"no-go box '{what}' set to {lo} .. {hi} (from {len(self.points)} points, padding {self.no_go_pad} m)"
        self.cancel()
        self.dirty = True
        return message

    def set_no_go(self, name: str, corners: list[float]) -> str:
        a, b = np.asarray(corners[:3], float), np.asarray(corners[3:], float)
        lo, hi = np.minimum(a, b).round(3).tolist(), np.maximum(a, b).round(3).tolist()
        if not all(x < y for x, y in zip(lo, hi, strict=True)):
            raise ValueError("the two corners must differ on every axis")
        self.no_go[name] = (lo, hi)
        self.dirty = True
        return f"no-go box '{name}' set to {lo} .. {hi}"

    def tool_inside(self, position, margin: float = 0.0) -> bool:
        p = np.asarray(position, float)
        return bool(np.all(p >= np.asarray(self.tool_min) + margin) and np.all(p <= np.asarray(self.tool_max) - margin))

    def summary(self, orientation: str = "") -> str:
        lines = [
            f"  tool box        {self.tool_min} .. {self.tool_max}",
            f"  guard box       {self.guard_min} .. {self.guard_max}  (fixed hard limit)",
        ]
        if self.no_go:
            lines += [f"  no-go '{name}'  {lo} .. {hi}" for name, (lo, hi) in self.no_go.items()]
        else:
            lines.append("  no-go boxes     none")
        lines.append(f"  start pose      {self.reset_joints if self.reset_joints is not None else 'not set'}")
        if orientation:
            lines.append(f"  orientation     {orientation}")
        if self.capture is not None:
            what = "the tool box" if self.capture == "tool" else f"no-go box '{self.capture}'"
            lines.append(f"  teaching {what}: {len(self.points)} points so far (`done` to finish)")
        if self.dirty:
            lines.append("  (unsaved changes: `save` writes them to the config files)")
        return "\n".join(lines)

    def robot_fields(self) -> dict:
        """The values as they go under env.robot in the config files."""
        return {
            "ee_bounds_min": list(self.tool_min),
            "ee_bounds_max": list(self.tool_max),
            "no_go_boxes": [{"name": n, "min": lo, "max": hi} for n, (lo, hi) in self.no_go.items()],
            "reset_joint_positions": None if self.reset_joints is None else list(self.reset_joints),
        }


def task_config_files(env_config_path: str | Path) -> list[Path]:
    """The config files of one task: the given file and its env_/train_ sibling if it exists."""
    path = Path(env_config_path)
    files = [path]
    for a, b in (("env_", "train_"), ("_env", "_train"), ("train_", "env_"), ("_train", "_env")):
        if a in path.name:
            sibling = path.with_name(path.name.replace(a, b))
            if sibling.exists() and sibling not in files:
                files.append(sibling)
    return files


def save_to_configs(draft: WorkspaceDraft, files: list[Path]) -> list[Path]:
    """Write the draft under env.robot in each file. All files are restored if any of
    them no longer loads as a valid config afterwards."""
    import draccus
    from lerobot.rl.train_rl import TrainRLServerPipelineConfig

    originals = {f: f.read_text() for f in files}
    try:
        for f in files:
            data = json.loads(originals[f])
            robot = data["env"]["robot"]
            fields_ = draft.robot_fields()
            if fields_["reset_joint_positions"] is None:
                fields_.pop("reset_joint_positions")  # keep what the file has
            robot.update(fields_)
            f.write_text(json.dumps(data, indent=2) + "\n")
            cls = TrainRLServerPipelineConfig if "policy" in data else GymManipulatorConfig
            with draccus.config_type("json"):
                draccus.parse(cls, f, args=[])  # runs the robot config's own validation
    except Exception:
        for f, text in originals.items():
            f.write_text(text)
        raise
    draft.dirty = False
    return files


def plot(draft: WorkspaceDraft, trail, path: Path) -> Path | None:
    """Top (x-y) and side (x-z) views of the boxes and where the tool has been."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    trail = np.asarray(trail, float).reshape(-1, 3)
    for ax, (i, j), title in zip(axes, ((0, 1), (0, 2)), ("top view: x forward, y left", "side view: x forward, z up"), strict=True):
        def box(lo, hi, i=i, j=j, ax=ax, **kw):
            ax.add_patch(Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j], **kw))

        box(draft.guard_min, draft.guard_max, fill=False, ec="gray", ls="--", label="guard box")
        box(draft.tool_min, draft.tool_max, fill=False, ec="tab:green", lw=2, label="tool box")
        for k, (name, (lo, hi)) in enumerate(draft.no_go.items()):
            box(lo, hi, fc="tab:red", alpha=0.35, ec="tab:red", label="no-go" if k == 0 else None)
            ax.annotate(name, ((lo[i] + hi[i]) / 2, (lo[j] + hi[j]) / 2), ha="center", va="center", fontsize=8)
        if len(trail):
            ax.plot(trail[:, i], trail[:, j], ".", ms=2, c="tab:blue", alpha=0.5, label="tool positions")
            ax.plot(trail[-1, i], trail[-1, j], "o", c="black", label="tool now")
        if j == 2:
            ax.axhline(0.0, c="saddlebrown", lw=1, label="table plane (z = 0)")
        ax.plot(0, 0, "s", c="dimgray", ms=10, label="arm base")
        ax.set(title=title, xlabel="x (m)", ylabel="yz"[j - 1] + " (m)", aspect="equal")
        ax.autoscale_view()
        ax.margins(0.1)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


class Session:
    """The draft, the live robot it is applied to, and the commands."""

    def __init__(self, env, config_files: list[Path], plot_path: Path = Path("runs/workspace.png")):
        self.env, self.robot = env, env.robot
        cfg = self.robot.config
        self.files = config_files
        self.plot_path = plot_path
        self.draft = WorkspaceDraft(
            guard_min=list(cfg.workspace_min),
            guard_max=list(cfg.workspace_max),
            tool_min=list(cfg.ee_bounds_min),
            tool_max=list(cfg.ee_bounds_max),
            no_go={b.name: (list(b.min), list(b.max)) for b in cfg.no_go_boxes},
            reset_joints=None if cfg.reset_joint_positions is None else list(cfg.reset_joint_positions),
        )
        self.configured_tool_box = (list(cfg.ee_bounds_min), list(cfg.ee_bounds_max))
        self.trail: list[list[float]] = []
        self.quit = False
        self.turn: tuple[str, bool, float] | None = None  # axis, about a tool axis?, radians left to turn
        self._turn_check: tuple[np.ndarray, int] | None = None  # orientation and rejections before the last step

    # ---- robot state
    def pose(self) -> tuple[np.ndarray, list[float], float]:
        raw = self.env.get_raw_joint_positions()
        tool = np.array([raw["ee.x"], raw["ee.y"], raw["ee.z"]])
        return tool, [round(raw[k], 3) for k in self.robot._arm_keys], raw.get("gripper.pos", float("nan"))

    def apply(self) -> None:
        """Push the draft's boxes to the live robot."""
        from lerobot_robot_ros.config_gen3_lite_ros import NoGoBox

        self.robot.set_tool_box(self.draft.tool_min, self.draft.tool_max)
        wanted = [NoGoBox(name=n, min=lo, max=hi) for n, (lo, hi) in self.draft.no_go.items()]
        if wanted != list(self.robot.config.no_go_boxes):
            self.robot.set_no_go_boxes(wanted)

    def open(self) -> str:
        self.draft.open_tool_box()
        self.apply()
        return f"tool box opened to {self.draft.tool_min} .. {self.draft.tool_max}"

    # ---- per control step
    def orientation(self) -> np.ndarray:
        return self.robot._kin.fk(self.robot._measured()[0])[1]

    def _advance_turn(self) -> str | None:
        """One small step of a yaw / pitch / roll command; undone if the arm could not follow."""
        robot = self.robot
        if self._turn_check is not None:
            before, rejections = self._turn_check
            self._turn_check = None
            if robot.ik_rejections > rejections:  # the last step could not be reached: take it back
                robot.set_held_orientation(before)
                self.turn = None
                return (
                    "cannot turn further here (a joint limit, or out of reach at this position); stopped at: "
                    + describe_orientation(self.orientation())
                )
        if self.turn is None:
            return None
        axis, in_tool, left = self.turn
        step = float(np.clip(left, -TURN_STEP_RAD, TURN_STEP_RAD))
        current = robot.held_orientation()
        rotation = axis_rotation(axis, step)
        robot.set_held_orientation(current @ rotation if in_tool else rotation @ current)
        self._turn_check = (current, robot.ik_rejections)
        left -= step
        if abs(left) > 1e-6:
            self.turn = (axis, in_tool, left)
            return None
        self.turn = None
        return "turned: " + describe_orientation(robot.held_orientation()) + " (give it a moment to settle)"

    def on_step(self, human: bool, marked: bool) -> str | None:
        tool, _, _ = self.pose()
        self.trail.append(tool.tolist())
        note = self._advance_turn()
        if note:
            return note
        draft = self.draft
        if draft.capture == "tool" and human:
            draft.add_point(tool)
        elif marked and draft.capture is not None:
            n = draft.add_point(tool)
            return f"point {n} at {np.round(tool, 3).tolist()}"
        elif marked:
            return "success button pressed, but nothing is being taught (`w` or `n NAME` first)"
        return None

    # ---- typed commands
    def command(self, line: str) -> str:
        draft = self.draft
        words = line.split()
        if not words:
            tool, joints, gripper = self.pose()
            inside = "inside" if draft.tool_inside(tool) else "OUTSIDE"
            return (
                f"tool {np.round(tool, 3).tolist()} ({inside} the tool box)  joints {joints}  gripper {gripper:.2f}\n"
                f"  {describe_orientation(self.orientation())}"
            )
        cmd, args = words[0].lower(), words[1:]
        try:
            if cmd == "w":
                draft.begin("tool")
                self.open()
                return "teaching the tool box: hold takeover and sweep the whole region the task needs, then `done`"
            if cmd == "n":
                if not args:
                    return "usage: n NAME   or   n NAME x0 y0 z0 x1 y1 z1"
                if len(args) == 7:
                    message = draft.set_no_go(args[0], [float(v) for v in args[1:]])
                    self.apply()
                    return message + self._no_go_check()
                if len(args) != 1:
                    return "usage: n NAME   or   n NAME x0 y0 z0 x1 y1 z1"
                draft.begin(args[0])
                return (
                    f"teaching no-go box '{args[0]}': bring the tool tip near each extreme of the object and "
                    "press the success button (or type `p`), then `done`"
                )
            if cmd == "p":
                if draft.capture is None:
                    return "nothing is being taught (`w` or `n NAME` first)"
                tool, _, _ = self.pose()
                return f"point {draft.add_point(tool)} at {np.round(tool, 3).tolist()}"
            if cmd == "done":
                was_no_go = draft.capture not in (None, "tool")
                message = draft.finish()
                self.apply()
                return message + (self._no_go_check() if was_no_go else self._reset_check())
            if cmd == "cancel":
                draft.cancel()
                return "cancelled"
            if cmd == "del":
                if not args or args[0] not in draft.no_go:
                    return f"no such no-go box; have {list(draft.no_go)}"
                del draft.no_go[args[0]]
                draft.dirty = True
                self.apply()
                return f"deleted '{args[0]}'"
            if cmd in ("yaw", "pitch", "roll"):
                degrees = float(args[0])
                if abs(degrees) > 90:
                    return "at most 90 degrees at a time"
                # yaw: about the vertical. pitch: about the tool's sideways (x) axis, positive
                # tips the pointing direction down. roll: about the pointing (z) axis.
                axis, in_tool, sign = {"yaw": ("z", False, 1), "pitch": ("x", True, 1), "roll": ("z", True, 1)}[cmd]
                self.turn = (axis, in_tool, sign * np.deg2rad(degrees))
                return f"turning {cmd} {degrees:+.0f} deg; watch the arm"
            if cmd == "r":
                tool, joints, _ = self.pose()
                draft.reset_joints, draft.dirty = joints, True
                self.env.reset_joint_positions = joints
                self.robot.freeze_orientation_at(joints)
                return (
                    f"start pose set to {joints} (tool at {np.round(tool, 3).tolist()})\n"
                    f"  orientation frozen: {describe_orientation(self.orientation())}" + self._reset_check()
                )
            if cmd == "open":
                return self.open()
            if cmd == "pad":
                value = float(args[0])
                if not 0 <= value <= 0.2:
                    return "padding must be between 0 and 0.2 m"
                if draft.capture not in (None, "tool"):
                    draft.no_go_pad = value
                    return f"no-go padding {value} m"
                draft.tool_pad = value
                return f"tool-box padding {value} m (type `pad` while teaching a no-go box to change that one)"
            if cmd == "show":
                try:
                    out = plot(draft, self.trail, self.plot_path)
                    picture = f"\n  picture: {out}" if out else "\n  (no picture: matplotlib is not installed)"
                except OSError as e:
                    picture = f"\n  (no picture: {e})"
                return draft.summary(describe_orientation(self.orientation())) + picture
            if cmd == "save":
                if draft.capture is not None:
                    return "finish (`done`) or `cancel` the box being taught first"
                problem = self._reset_check()
                files = save_to_configs(draft, self.files)
                return "saved to " + ", ".join(str(f) for f in files) + problem
            if cmd in ("quit", "q", "exit"):
                self.quit = True
                return "leaving" + (" WITHOUT saving the changes" if draft.dirty else "")
            if cmd in ("help", "h", "?"):
                return __doc__.split("You drive with the gamepad")[1].split("Everything takes effect")[0]
        except (ValueError, IndexError) as e:
            return f"not done: {e}"
        return f"unknown command '{cmd}' (`help` lists them)"

    def _reset_check(self) -> str:
        if self.draft.reset_joints is None:
            return ""
        warnings = ""
        tool = self.robot._kin.fk(self.draft.reset_joints)[0]
        if not self.draft.tool_inside(tool, margin=0.01):
            warnings += f"\n  WARNING: the start pose's tool position {np.round(tool, 3).tolist()} is not 1 cm inside the tool box"
        iface = self.robot.config.ros2_interface
        if iface.min_joint_positions is not None and iface.max_joint_positions is not None:
            joints = np.asarray(self.draft.reset_joints)
            room = np.minimum(joints - np.asarray(iface.min_joint_positions), np.asarray(iface.max_joint_positions) - joints)
            tight = [f"{name} ({r:.2f} rad left)" for name, r in zip(iface.arm_joint_names, room, strict=True) if r < 0.15]
            if tight:
                warnings += (
                    "\n  WARNING: at this start pose " + ", ".join(tight) + " is close to its limit; the arm will have "
                    "little room to move with this orientation. Consider a different position or orientation."
                )
        return warnings

    def _no_go_check(self) -> str:
        guard = getattr(self.robot, "_guard", None)
        if guard is None:
            return "\n  NOTE: the workspace guard is off here, so no-go boxes are saved but not enforced in this session"
        arm, grip = self.robot._measured()
        violation, reason = guard.assess(arm, grip)
        return f"\n  WARNING: the arm is inside the margin right now ({reason}); back away" if violation > 0 else ""


def run(env, env_processor, action_processor, cfg: GymManipulatorConfig, session: Session, commands=None,
        max_steps: int | None = None) -> Session:
    """Drive loop. ``commands`` is a queue of typed lines (the terminal by default)."""
    use_gripper = env.use_gripper
    neutral = torch.tensor([0.0, 0.0, 0.0] + ([1.0] if use_gripper else []), dtype=torch.float32)
    dt = 1.0 / cfg.env.fps
    env_processor.steps = [s for s in env_processor.steps if type(s).__name__ != "TimeLimitProcessorStep"]
    transition = reset_and_build_transition(env, env_processor, action_processor)
    if commands is None:
        commands = queue.Queue()
        threading.Thread(target=lambda: [commands.put(line.rstrip("\n")) for line in sys.stdin], daemon=True).start()

    print(session.open())
    print(session.draft.summary(describe_orientation(session.orientation())))
    print("Type `help` for the commands. Hold the takeover button to move.", flush=True)
    steps, blocked_seen = 0, session.robot.blocked_commands
    try:
        while not session.quit and (max_steps is None or steps < max_steps):
            start = time.perf_counter()
            steps += 1
            transition = step_env_and_process_transition(
                env=env, transition=transition, action=neutral.clone(),
                env_processor=env_processor, action_processor=action_processor,
            )
            info = transition[TransitionKey.INFO]
            note = session.on_step(
                human=bool(info.get(TeleopEvents.IS_INTERVENTION, False)),
                marked=bool(info.get(TeleopEvents.SUCCESS, False)),
            )
            if note:
                print(note, flush=True)
            if session.robot.blocked_commands > blocked_seen + 5:  # the arm is being held by the guard
                blocked_seen = session.robot.blocked_commands
                print("(the workspace guard is holding the arm here; see the warning above for why)", flush=True)
            while True:
                try:
                    line = commands.get_nowait()
                except queue.Empty:
                    break
                try:
                    print(session.command(line), flush=True)
                except Exception as e:  # a failed command must not end a session that is driving the arm
                    logging.exception("command failed")
                    print(f"that command failed ({type(e).__name__}: {e}); nothing was changed by it", flush=True)
            precise_sleep(max(dt - (time.perf_counter() - start), 0.0))
    except KeyboardInterrupt:
        print("\ninterrupted" + (" (unsaved changes were NOT written)" if session.draft.dirty else ""))
    return session


@parser.wrap()
def workspace_cli(cfg: GymManipulatorConfig) -> None:
    logging.basicConfig(level=logging.WARNING)
    require_motion_confirmation(cfg.env.robot, _SETTINGS)
    config_path = parser.parse_arg("config_path")
    files = task_config_files(config_path) if config_path else []
    print("Will save to: " + (", ".join(str(f) for f in files) or "nothing (no --config_path given)"))
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)
    try:
        run(env, env_processor, action_processor, cfg, Session(env, files))
    finally:
        env.close()


def main() -> None:
    global _SETTINGS
    _SETTINGS, sys.argv = pop_run_settings()
    from . import compat

    compat.apply()
    import lerobot_robot_ros  # noqa: F401
    import lerobot_teleoperator_devices  # noqa: F401

    workspace_cli()


if __name__ == "__main__":
    main()
