"""Evaluation schedule, CSV logs, rise-time numbers, run settings, actor bookkeeping."""

import json

import pytest

from hilserl_gen3_lite.actor import ActorRun
from hilserl_gen3_lite.eval_schedule import EvalSchedule
from hilserl_gen3_lite.run_logging import CsvLog, read_csv, rise_time
from hilserl_gen3_lite.settings import RunSettings, pop_run_settings, require_motion_confirmation, run_dir_and_argv


def test_eval_blocks_start_at_the_interval_and_last_the_block_size():
    sched = EvalSchedule(interval_steps=100, block_episodes=2)
    steps, kinds, blocks = 0, [], []
    for _ in range(12):
        is_eval = sched.begin_episode(steps)
        kinds.append(is_eval)
        steps += 30
        block = sched.end_episode(success=True, intervened=False, length=30, env_steps=steps)
        if block:
            blocks.append(block)
    # steps before each episode: 0 30 60 90 120* 150* 180 210* 240* 270 300* 330*   (* = evaluation)
    assert kinds == [False, False, False, False, True, True, False, True, True, False, True, True]
    assert [b["env_steps_at_start"] for b in blocks] == [120, 210, 300]
    assert all(b["episodes"] == 2 and b["success_rate"] == 1.0 for b in blocks)


def test_eval_episode_with_a_takeover_counts_as_a_failure():
    sched = EvalSchedule(interval_steps=10, block_episodes=2)
    sched.begin_episode(10)
    assert sched.end_episode(success=True, intervened=True, length=5, env_steps=15) is None
    sched.begin_episode(15)
    block = sched.end_episode(success=True, intervened=False, length=5, env_steps=20)
    assert block["successes"] == 1 and block["success_rate"] == 0.5 and block["episodes_with_takeover"] == 1


def test_disabled_schedule_never_evaluates():
    sched = EvalSchedule(interval_steps=1, block_episodes=1, enabled=False)
    assert not sched.begin_episode(1000)
    assert sched.end_episode(True, False, 1, 1001) is None


def test_rise_time_numbers():
    rows = [
        {"success_rate": "0.1", "env_steps_at_start": "2000"},
        {"success_rate": "0.6", "env_steps_at_start": "4000"},
        {"success_rate": "0.4", "env_steps_at_start": "6000"},
        {"success_rate": "0.9", "env_steps_at_start": "8000"},
        {"success_rate": "0.8", "env_steps_at_start": "10000"},
    ]
    assert rise_time(rows) == {
        "env_steps_to_50pct": 4000,
        "env_steps_to_90pct": 8000,
        "final_eval_success_rate": 0.8,
        "eval_blocks": 5,
    }
    assert rise_time([])["env_steps_to_50pct"] is None


def test_csv_log_appends_across_restarts(tmp_path):
    path = tmp_path / "x.csv"
    log = CsvLog(path, ["a", "b"])
    log.write({"a": 1, "b": 2, "ignored": 3})
    log.close()
    log = CsvLog(path, ["a", "b"])
    log.write({"a": 3})
    log.close()
    assert read_csv(path) == [{"a": "1", "b": "2"}, {"a": "3", "b": ""}]


def test_run_settings_are_split_from_the_command_line(tmp_path):
    argv = ["prog", "--config_path=c.json", "--run.eval_interval_steps=500", "--run.announce=false",
            f"--run.dir={tmp_path / 'r'}", "--policy.device=cpu"]
    settings, rest = pop_run_settings(argv)
    assert settings.eval_interval_steps == 500 and settings.announce is False
    assert rest == ["prog", "--config_path=c.json", "--policy.device=cpu"]
    run_dir, rest = run_dir_and_argv(settings, rest, "learner")
    assert run_dir.is_dir() and rest[-1] == f"--output_dir={run_dir / 'learner'}"
    with pytest.raises(SystemExit):
        pop_run_settings(["prog", "--run.nonsense=1"])
    with pytest.raises(SystemExit):
        run_dir_and_argv(RunSettings(), ["prog"], "learner")


def test_real_arm_needs_explicit_confirmation(urdf_path, capsys):
    from lerobot_robot_ros import Gen3LiteHilConfig, Gen3LiteHilFakeConfig

    with pytest.raises(SystemExit, match="moves the real arm"):
        require_motion_confirmation(Gen3LiteHilConfig(), RunSettings())
    assert "max tool speed" in capsys.readouterr().out  # limits are printed before refusing
    require_motion_confirmation(Gen3LiteHilConfig(), RunSettings(confirm_motion=True))
    require_motion_confirmation(Gen3LiteHilFakeConfig(urdf_path=urdf_path), RunSettings())


def _episode(n, takeover=(), reward_last=1.0):
    return [
        {"reward": reward_last if i == n - 1 else 0.0, "complementary_info": {"is_intervention": i in takeover}}
        for i in range(n)
    ]


def test_actor_run_logs_episodes_and_eval_blocks_and_survives_a_restart(tmp_path):
    settings = RunSettings(eval_interval_steps=20, eval_episodes=2)
    run = ActorRun(tmp_path, settings)
    plan = [  # (length, takeover steps, final reward)
        (10, {3, 4, 7}, 1.0),
        (10, set(), 0.0),
        (5, set(), 1.0),  # eval
        (5, {2}, 1.0),  # eval, with a takeover: failure
        (10, set(), 1.0),
    ]
    for length, takeover, reward in plan:
        run.on_reset()
        tagged = run.on_episode(_episode(length, takeover, reward))
        assert all("source" in t["complementary_info"] for t in tagged)
    rows = read_csv(tmp_path / "episodes.csv")
    assert [r["is_eval"] for r in rows] == ["0", "0", "1", "1", "0"]
    assert [r["env_steps"] for r in rows] == ["10", "20", "25", "30", "40"]
    assert [r["train_env_steps"] for r in rows] == ["10", "20", "20", "20", "30"]
    assert rows[0]["takeovers"] == "2" and rows[0]["intervention_fraction"] == "0.3" and rows[0]["success"] == "1"
    assert rows[1]["success"] == "0"
    evals = read_csv(tmp_path / "eval.csv")
    assert len(evals) == 1 and evals[0]["success_rate"] == "0.5" and evals[0]["env_steps_at_start"] == "20"
    assert json.loads((tmp_path / "rise_time.json").read_text())["env_steps_to_50pct"] == 20

    again = ActorRun(tmp_path, settings)  # the actor process was restarted
    assert again.episode_index == 5 and again.env_steps == 40 and again.schedule.next_eval_at == 40
    again.on_reset()
    again.on_episode(_episode(4))
    assert read_csv(tmp_path / "episodes.csv")[-1]["episode"] == "5"
