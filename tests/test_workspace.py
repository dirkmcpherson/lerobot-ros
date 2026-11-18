"""Interactive workspace setup: teaching logic, live application, saving, no-go enforcement."""

import json
import queue
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from lerobot.rl.gym_manipulator import DatasetConfig, GymManipulatorConfig

from hilserl_gen3_lite.env import make_processors, make_robot_env
from hilserl_gen3_lite.workspace import Session, WorkspaceDraft, run, save_to_configs, task_config_files
from lerobot_robot_ros.config_gen3_lite_ros import NoGoBox

from .conftest import KORTEX_SRC, START_JOINTS
from .test_gripper import env_config

CONFIG_DIR = Path(__file__).parent.parent / "config" / "hilserl"
GUARD = ([0.13, 0.01, 0.02], [0.45, 0.31, 0.50])


def draft(**kw):
    return WorkspaceDraft(guard_min=GUARD[0], guard_max=GUARD[1], tool_min=[0.15, 0.03, 0.04], tool_max=[0.43, 0.29, 0.40], **kw)


def test_tool_box_is_the_swept_region_plus_padding_kept_inside_the_guard_box():
    d = draft()
    d.begin("tool")
    for p in ([0.25, 0.10, 0.06], [0.35, 0.20, 0.25], [0.30, 0.15, 0.10]):
        d.add_point(p)
    assert "tool box set" in d.finish()
    assert (d.tool_min, d.tool_max) == ([0.23, 0.08, 0.04], [0.37, 0.22, 0.27])
    assert d.capture is None and d.dirty
    d.begin("tool")  # a sweep that touches the guard box is cut 1 cm inside it
    d.add_point([0.135, 0.02, 0.03])
    d.add_point([0.449, 0.30, 0.49])
    d.finish()
    assert (d.tool_min, d.tool_max) == ([0.14, 0.02, 0.03], [0.44, 0.30, 0.49])


def test_no_go_box_from_marked_points_and_from_typed_corners():
    d = draft()
    d.begin("camera")
    d.add_point([0.40, 0.25, 0.10])
    d.add_point([0.42, 0.28, 0.30])
    assert "no-go box 'camera'" in d.finish()
    assert d.no_go["camera"] == ([0.37, 0.22, 0.07], [0.45, 0.31, 0.33])
    d.set_no_go("shelf", [0.5, 0.3, 0.2, 0.3, -0.1, 0.0])  # corners in any order
    assert d.no_go["shelf"] == ([0.3, -0.1, 0.0], [0.5, 0.3, 0.2])
    assert d.robot_fields()["no_go_boxes"][0] == {"name": "camera", "min": [0.37, 0.22, 0.07], "max": [0.45, 0.31, 0.33]}
    with pytest.raises(ValueError):
        d.set_no_go("flat", [0.1, 0.1, 0.1, 0.2, 0.2, 0.1])
    with pytest.raises(ValueError, match="nothing is being taught"):
        d.finish()
    d.begin("empty")
    with pytest.raises(ValueError, match="no points"):
        d.finish()


class Script:
    """Typed lines released at given control steps (a stand-in for the terminal)."""

    def __init__(self, lines: dict[int, list[str]]):
        self.lines, self.step, self.pending = lines, 0, []

    def get_nowait(self):
        if not self.pending:
            self.step += 1
            self.pending = list(self.lines.pop(self.step, [])) + [None]
        line = self.pending.pop(0)
        if line is None:
            raise queue.Empty
        return line


@pytest.fixture
def task_files(tmp_path):
    files = []
    for name in ("dryrun_cup_env.json", "dryrun_cup_train.json"):
        shutil.copy(CONFIG_DIR / name, tmp_path / name)
        files.append(tmp_path / name)
    return files


def test_task_config_files_finds_the_sibling(task_files):
    assert task_config_files(task_files[0]) == task_files
    assert task_config_files(task_files[1]) == [task_files[1], task_files[0]]


def test_interactive_session_teaches_tests_and_saves(tmp_path, task_files, capsys):
    """A whole session on the fake arm: sweep the tool box while the scripted operator does
    its pick and place, set the start pose, add a no-go box by coordinates, save. The tool
    box is live straight away, and both config files load afterwards."""
    cfg = GymManipulatorConfig(env=env_config(tmp_path), dataset=DatasetConfig(repo_id="local/unused", task="setup"))
    env, teleop = make_robot_env(cfg.env)
    env_proc, act_proc = make_processors(env, teleop, cfg.env)
    session = Session(env, task_files, plot_path=tmp_path / "workspace.png")
    script = Script({
        1: ["w"],
        45: ["done", "", "r", "n camera 0.40 0.26 0.05 0.44 0.30 0.30", "show", "nonsense", "save", "quit"],
    })
    run(env, env_proc, act_proc, cfg, session, commands=script, max_steps=80)
    out = capsys.readouterr().out
    assert session.quit and not session.draft.dirty
    assert "tool box set to" in out and "start pose set to" in out and "unknown command" in out
    assert (tmp_path / "workspace.png").stat().st_size > 5000

    d = session.draft
    lo, hi = np.array(d.tool_min), np.array(d.tool_max)
    for point in ([0.30, 0.16, 0.22], [0.32, 0.14, 0.22], [0.28, 0.18, 0.22]):  # start, pick, place
        assert np.all(lo < point) and np.all(point < hi)
    assert np.all(hi - lo < [0.12, 0.12, 0.08])  # a tight box, not the 30 cm default
    assert env.robot.config.ee_bounds_min == d.tool_min  # applied to the live robot

    # ... and enforced by it: drive out in +x and the tool stops at the new wall
    for _ in range(30):
        env.robot.send_action({"delta_x": 1.0, "delta_y": 0.0, "delta_z": 0.0})
        time.sleep(0.02)
    assert env.robot._kin.fk(env.robot.backend.joints)[0][0] <= d.tool_max[0] + 5e-4

    for f in task_files:
        robot = json.loads(f.read_text())["env"]["robot"]
        assert robot["ee_bounds_min"] == d.tool_min and robot["ee_bounds_max"] == d.tool_max
        assert robot["no_go_boxes"] == [{"name": "camera", "min": [0.40, 0.26, 0.05], "max": [0.44, 0.30, 0.30]}]
        np.testing.assert_allclose(robot["reset_joint_positions"], d.reset_joints)
    env.close()


def test_save_is_undone_if_a_config_would_not_load(task_files):
    before = [f.read_text() for f in task_files]
    bad = draft()
    bad.tool_min, bad.tool_max = [0.0, 0.0, 0.0], [0.9, 0.9, 0.9]  # outside the guard box
    with pytest.raises(Exception):  # noqa: B017  (draccus wraps the robot config's ValueError)
        save_to_configs(bad, task_files)
    assert [f.read_text() for f in task_files] == before


def test_no_go_boxes_parse_from_json_and_need_the_guard(urdf_path, make_robot):
    import draccus
    from lerobot.rl.train_rl import TrainRLServerPipelineConfig

    from hilserl_gen3_lite import compat

    compat.apply()
    data = json.loads((CONFIG_DIR / "train_gen3_lite_cup.json").read_text())
    data["env"]["robot"]["no_go_boxes"] = [{"name": "camera", "min": [0.5, 0.0, 0.0], "max": [0.6, 0.1, 0.4]}]
    path = Path(urdf_path).parent.parent / "_tmp_no_go.json"
    try:
        path.write_text(json.dumps(data))
        with draccus.config_type("json"):
            cfg = draccus.parse(TrainRLServerPipelineConfig, path, args=[])
    finally:
        path.unlink(missing_ok=True)
    assert cfg.env.robot.no_go_boxes == [NoGoBox(name="camera", min=[0.5, 0.0, 0.0], max=[0.6, 0.1, 0.4])]
    cfg.to_dict()  # the learner serialises its config on start-up
    with pytest.raises(ValueError, match="min < max"):
        NoGoBox(name="flat", min=[0, 0, 0], max=[1, 1, 0])
    with pytest.raises(ValueError, match="would not be enforced"):
        make_robot(no_go_boxes=[NoGoBox(name="x", min=[0.5, 0, 0], max=[0.6, 0.1, 0.1])])  # guard is off on the fake arm


@pytest.mark.skipif(not (KORTEX_SRC / "kortex_description").is_dir(), reason="kortex_description meshes not found")
def test_arm_stops_short_of_a_no_go_box_and_can_back_away(make_robot, urdf_path):
    """Mesh-based: a box 8 cm ahead of the tool. Driving at it, no part of the arm (fingers
    included) gets within the margin; the reason names the box; backing away works."""
    from lerobot_backends.ros2.workspace_guard import WorkspaceGuard

    srdf = KORTEX_SRC / "kortex_moveit_config/kinova_gen3_lite_moveit_config/config/gen3_lite.srdf"
    robot = make_robot(max_obs_age_s=1.0)
    cfg = robot.config

    def build_guard():
        return WorkspaceGuard(
            urdf_path=urdf_path, arm_joint_names=cfg.ros2_interface.arm_joint_names,
            gripper_joint_name=cfg.ros2_interface.gripper_joint_name, plane_z=cfg.plane_z,
            plane_margin=cfg.plane_margin, box_min=cfg.workspace_min, box_max=cfg.workspace_max,
            srdf_path=str(srdf) if srdf.is_file() else None, package_dirs=[str(KORTEX_SRC)],
            no_go_boxes=[{"name": b.name, "min": b.min, "max": b.max} for b in cfg.no_go_boxes],
            no_go_margin=cfg.no_go_margin,
        )

    robot._build_guard = build_guard
    cfg.guard_enabled = True
    # At this pose the gripper points mostly +y; with the gripper open a finger reaches
    # about 6 cm past the tool point. The box starts 11 cm away in +y.
    start_y = robot._kin.fk(START_JOINTS)[0][1]
    box_y = start_y + 0.11
    robot.set_no_go_boxes([NoGoBox(name="camera", min=[0.20, box_y, 0.12], max=[0.40, box_y + 0.06, 0.32])])
    assert robot._guard.no_go_names == ["camera"]
    assert robot._guard.assess(list(robot.backend.joints), 0.0)[0] == 0.0

    for _ in range(20):
        robot.send_action({"delta_x": 0.0, "delta_y": 1.0, "delta_z": 0.0})
        time.sleep(0.1)
    assert robot.blocked_commands > 5
    arm = list(robot.backend.joints)
    assert robot._guard.assess(arm, 0.0)[0] == 0.0  # no link, finger included, inside the margin
    tool_y = robot._kin.fk(arm)[0][1]
    assert start_y + 0.01 < tool_y < box_y - cfg.no_go_margin  # it moved, and stopped short
    closer, _, _ = robot._kin.ik(arm, robot._kin.fk(arm)[0] + [0, 0.03, 0], robot._kin.fk(arm)[1])
    violation, reason = robot._guard.assess(list(closer), 0.0)
    assert violation > 0 and "no-go box 'camera'" in reason

    before = robot.blocked_commands
    for _ in range(3):
        robot.send_action({"delta_x": 0.0, "delta_y": -1.0, "delta_z": 0.0})
        time.sleep(0.1)
    assert robot.blocked_commands == before and robot._kin.fk(robot.backend.joints)[0][1] < tool_y - 0.02


def test_turn_commands_choose_the_frozen_orientation(tmp_path, task_files, capsys):
    """`yaw` turns the gripper on the spot; `r` then freezes that orientation for the task."""
    from hilserl_gen3_lite.workspace import describe_orientation

    cfg = GymManipulatorConfig(
        env=env_config(tmp_path, always_intervene=False), dataset=DatasetConfig(repo_id="local/unused", task="setup")
    )
    env, teleop = make_robot_env(cfg.env)
    env_proc, act_proc = make_processors(env, teleop, cfg.env)
    session = Session(env, task_files, plot_path=tmp_path / "workspace.png")
    kin, backend = env.robot._kin, env.robot.backend
    start_pos, start_rot = kin.fk(START_JOINTS)
    heading0 = np.degrees(np.arctan2(start_rot[1, 2], start_rot[0, 2]))
    assert "60 deg to the left" in describe_orientation(start_rot)  # Kinova home's orientation

    run(env, env_proc, act_proc, cfg, session, commands=Script({2: ["yaw -20"], 40: ["r", ""]}), max_steps=42)
    pos, rot = kin.fk(backend.joints)
    heading = np.degrees(np.arctan2(rot[1, 2], rot[0, 2]))
    assert heading == pytest.approx(heading0 - 20, abs=1.0)
    assert abs(rot[2, 2]) < 0.05  # still level
    np.testing.assert_allclose(pos, start_pos, atol=2e-3)  # turned on the spot
    assert "40 deg to the left" in capsys.readouterr().out
    np.testing.assert_allclose(session.draft.reset_joints, backend.joints, atol=5e-3)

    # the new orientation is the frozen one: it holds while the tool moves
    for _ in range(8):
        env.robot.send_action({"delta_x": 1.0, "delta_y": 0.0, "delta_z": 1.0})
        time.sleep(0.02)
    _, moved_rot = kin.fk(backend.joints)
    assert np.degrees(np.arccos(np.clip((np.trace(rot.T @ moved_rot) - 1) / 2, -1, 1))) < 0.5
    env.close()


def test_a_turn_the_arm_cannot_make_stops_and_leaves_the_arm_usable(tmp_path, task_files, capsys):
    """At this position the elbow reaches its limit about 26 degrees into a yaw towards
    straight ahead. The turn stops there, says so, and the arm still moves afterwards."""
    cfg = GymManipulatorConfig(
        env=env_config(tmp_path, always_intervene=False), dataset=DatasetConfig(repo_id="local/unused", task="setup")
    )
    env, teleop = make_robot_env(cfg.env)
    env_proc, act_proc = make_processors(env, teleop, cfg.env)
    session = Session(env, task_files, plot_path=tmp_path / "workspace.png")
    kin, backend = env.robot._kin, env.robot.backend
    heading0 = np.degrees(np.arctan2(*kin.fk(START_JOINTS)[1][[1, 0], 2]))

    run(env, env_proc, act_proc, cfg, session, commands=Script({2: ["yaw -60"]}), max_steps=60)
    assert "cannot turn further here" in capsys.readouterr().out
    assert session.turn is None
    assert "joint_3" in session.command("r") and "close to its limit" in session.command("r")  # and says why it is a poor start pose
    turned = heading0 - np.degrees(np.arctan2(*kin.fk(backend.joints)[1][[1, 0], 2]))
    assert 20 < turned < 32
    rejections = env.robot.ik_rejections
    before = kin.fk(backend.joints)[0]
    for _ in range(5):  # not stuck: an ordinary move is accepted and executed
        env.robot.send_action({"delta_x": 0.0, "delta_y": 0.0, "delta_z": 1.0})
        time.sleep(0.02)
    assert env.robot.ik_rejections == rejections
    assert kin.fk(backend.joints)[0][2] > before[2] + 0.02
    env.close()


def test_a_failing_command_does_not_end_the_session(tmp_path, task_files, capsys, monkeypatch):
    """Seen in the Gazebo rehearsal: `show` could not write its picture and the exception
    ended the session. Commands that fail are reported and the loop carries on."""
    cfg = GymManipulatorConfig(
        env=env_config(tmp_path, always_intervene=False), dataset=DatasetConfig(repo_id="local/unused", task="setup")
    )
    env, teleop = make_robot_env(cfg.env)
    env_proc, act_proc = make_processors(env, teleop, cfg.env)
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("")
    session = Session(env, task_files, plot_path=blocker / "workspace.png")  # cannot be written
    monkeypatch.setattr(session, "_reset_check", lambda: 1 / 0)  # and a command that blows up outright
    run(env, env_proc, act_proc, cfg, session, commands=Script({2: ["show"], 4: ["r"], 6: ["", "quit"]}), max_steps=20)
    out = capsys.readouterr().out
    assert "no picture" in out and "tool box" in out  # show still reported the settings
    assert "that command failed (ZeroDivisionError" in out
    assert session.quit  # it reached the later commands
    env.close()
