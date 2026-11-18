"""The gripper path (pick-and-place tasks), on the fake arm."""

import time

import numpy as np
import torch
from lerobot.envs.configs import GripperConfig, HILSerlProcessorConfig, HILSerlRobotEnvConfig, ResetConfig
from lerobot.processor import TransitionKey
from lerobot.rl.gym_manipulator import (
    reset_and_build_transition,
    step_env_and_process_transition,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep
from lerobot_teleoperator_devices import ScriptedEeTeleopConfig

from hilserl_gen3_lite.env import GRIPPER_TOGGLED, GripperTogglePenaltyStep, make_processors, make_robot_env
from lerobot_robot_ros import Gen3LiteHilFakeConfig

from .conftest import START_JOINTS, URDF

FPS = 50
PICK, PLACE = [0.32, 0.14, 0.22], [0.28, 0.18, 0.22]
EPISODE_STEPS = 40


def env_config(tmp_path, always_intervene=True, cameras=None) -> HILSerlRobotEnvConfig:
    period = 1.0 / FPS
    return HILSerlRobotEnvConfig(
        fps=FPS,
        task="pick_place",
        robot=Gen3LiteHilFakeConfig(
            urdf_path=URDF, calibration_dir=tmp_path / "calib", control_period_s=period,
            max_ee_speed_m_s=0.01 / period, use_gripper=True, reset_joint_positions=START_JOINTS,
            home_speed=50.0, cameras=cameras or {},
        ),
        teleop=ScriptedEeTeleopConfig(
            use_gripper=True, target=PICK, place_target=PLACE, always_intervene=always_intervene,
            takeovers=[], episode_steps=EPISODE_STEPS,
        ),
        processor=HILSerlProcessorConfig(
            gripper=GripperConfig(use_gripper=True, gripper_penalty=-0.02),
            reset=ResetConfig(reset_time_s=0.0, control_time_s=(EPISODE_STEPS + 1.5) / FPS),
            max_gripper_pos=None,
        ),
    )


def run_episode(env, env_proc, act_proc, policy_action):
    transition = reset_and_build_transition(env, env_proc, act_proc)
    steps = []
    while True:
        start = time.perf_counter()
        transition = step_env_and_process_transition(
            env=env, transition=transition, action=policy_action.clone(), env_processor=env_proc,
            action_processor=act_proc,
        )
        steps.append(transition)
        if transition.get(TransitionKey.DONE, False) or transition.get(TransitionKey.TRUNCATED, False):
            return steps
        precise_sleep(max(1 / FPS - (time.perf_counter() - start), 0.0))  # the speed limit assumes the control rate


def test_scripted_pick_and_place_through_the_pipeline(tmp_path):
    """Human-driven episode: 4-number actions, the gripper closes at the pick point and opens
    at the place point, success is reported, and exactly the two toggles are penalised."""
    cfg = env_config(tmp_path)
    env, teleop = make_robot_env(cfg)
    env_proc, act_proc = make_processors(env, teleop, cfg)
    assert env.action_space.shape == (4,)
    steps = run_episode(env, env_proc, act_proc, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    backend = env.robot.backend

    assert float(steps[-1][TransitionKey.REWARD]) == 1.0 and steps[-1][TransitionKey.DONE]
    assert len(steps) < EPISODE_STEPS
    actions = np.array([s[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"].reshape(-1).numpy() for s in steps])
    assert actions.shape[1] == 4 and set(actions[:, 3]) <= {0.0, 1.0, 2.0}
    assert all(s[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] for s in steps)
    penalties = [s[TransitionKey.COMPLEMENTARY_DATA]["discrete_penalty"] for s in steps]
    assert sorted(set(penalties)) == [-0.02, 0.0] and penalties.count(-0.02) == 2
    # the first toggle is the close at the pick point, the second the open at the place point
    first, second = [i for i, p in enumerate(penalties) if p < 0]
    assert actions[first, 3] == 0.0 and actions[second, 3] == 2.0
    assert backend.gripper == env.robot.config.ros2_interface.gripper_open_position
    tool = env.robot._kin.fk(backend.joints)[0]
    assert np.linalg.norm(tool - PLACE) < 0.015
    env.close()


def test_policy_gripper_commands_are_executed_and_only_changes_are_penalised(tmp_path):
    cfg = env_config(tmp_path, always_intervene=False)
    env, teleop = make_robot_env(cfg)
    env_proc, act_proc = make_processors(env, teleop, cfg)
    iface = env.robot.config.ros2_interface
    transition = reset_and_build_transition(env, env_proc, act_proc)
    seen = []
    for gripper in (1, 0, 0, 0, 1, 2, 2, 0):  # stay, close x3, stay, open x2, close
        transition = step_env_and_process_transition(
            env=env, transition=transition, action=torch.tensor([[0.0, 0.0, 0.0, float(gripper)]]),
            env_processor=env_proc, action_processor=act_proc,
        )
        seen.append((env.robot.backend.gripper, transition[TransitionKey.COMPLEMENTARY_DATA]["discrete_penalty"]))
    o, c = iface.gripper_open_position, iface.gripper_close_position
    assert [g for g, _ in seen] == [o, c, c, c, c, o, o, c]
    assert [p for _, p in seen] == [0.0, -0.02, 0.0, 0.0, 0.0, -0.02, 0.0, -0.02]
    env.close()


def test_penalty_step_defaults_to_zero_without_a_toggle_report():
    step = GripperTogglePenaltyStep(penalty=-0.05)
    out = step({TransitionKey.INFO: {}, TransitionKey.COMPLEMENTARY_DATA: {"x": 1}})
    assert out[TransitionKey.COMPLEMENTARY_DATA] == {"x": 1, "discrete_penalty": 0.0}
    out = step({TransitionKey.INFO: {GRIPPER_TOGGLED: True}, TransitionKey.COMPLEMENTARY_DATA: None})
    assert out[TransitionKey.COMPLEMENTARY_DATA]["discrete_penalty"] == -0.05


def test_observe_only_session_sends_nothing(make_robot):
    robot = make_robot()
    robot.get_observation()
    robot.disconnect()
    assert robot.backend.sent == []  # no hold either: nothing was ever commanded
