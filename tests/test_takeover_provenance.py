"""Phase 2: a takeover recorded through lerobot's own pipeline steps reaches the stored
transitions, and a batch sampled from lerobot's replay buffers, with the right
is_intervention / source / takeover_onset / episode_index / step_index."""

import time

import numpy as np
import pytest
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.envs.configs import (
    GripperConfig,
    HILSerlProcessorConfig,
    HILSerlRobotEnvConfig,
    ImagePreprocessingConfig,
    ResetConfig,
)
from lerobot.processor import TransitionKey
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.data_sources import OnlineOfflineMixer
from lerobot.rl.gym_manipulator import (
    DatasetConfig,
    GymManipulatorConfig,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import Transition
from lerobot_teleoperator_devices import ScriptedEeTeleopConfig

from hilserl_gen3_lite.env import make_processors, make_robot_env
from hilserl_gen3_lite.learner import batch_source_fractions, check_buffer_keys
from hilserl_gen3_lite.provenance import (
    COMPLEMENTARY_KEYS,
    SOURCE_DEMO,
    SOURCE_INTERVENTION,
    SOURCE_POLICY,
    provenance_tags,
    tag_episode,
)
from hilserl_gen3_lite.record import record
from lerobot_robot_ros import Gen3LiteHilFakeConfig

from .conftest import START_JOINTS

STATE_KEYS = ["observation.state", "observation.images.front"]
FPS = 50  # the fake arm has no dynamics, so run the loop fast
TAKEOVER = [5, 9]  # the scripted operator drives on steps 5..8 of each episode
EPISODE_STEPS = 12


def env_config(urdf_path, tmp_path, always_intervene=False) -> HILSerlRobotEnvConfig:
    period = 1.0 / FPS
    robot = Gen3LiteHilFakeConfig(
        urdf_path=urdf_path,
        calibration_dir=tmp_path / "calib",
        control_period_s=period,
        max_ee_speed_m_s=0.01 / period,
        cameras={"front": OpenCVCameraConfig(index_or_path=0, fps=FPS, width=48, height=48)},
        home_speed=50.0,
        reset_joint_positions=START_JOINTS,
    )
    teleop = ScriptedEeTeleopConfig(
        takeovers=[TAKEOVER],
        always_intervene=always_intervene,
        episode_steps=EPISODE_STEPS,
        target=[0.30, 0.16, 0.50],  # out of reach within an episode: no early success
    )
    return HILSerlRobotEnvConfig(
        fps=FPS,
        task="reach",
        robot=robot,
        teleop=teleop,
        processor=HILSerlProcessorConfig(
            image_preprocessing=ImagePreprocessingConfig(crop_params_dict=None, resize_size=(32, 32)),
            gripper=GripperConfig(use_gripper=False),
            reset=ResetConfig(
                reset_time_s=0.0,
                # upstream's time limit also counts the reset observation: N steps need a limit of N + 1
                control_time_s=(EPISODE_STEPS + 1.5) / FPS,
            ),
            max_gripper_pos=None,
        ),
    )


def run_actor_episode(env, env_proc, act_proc, policy_action) -> list[Transition]:
    """The transition-building part of lerobot.rl.actor.act_with_policy, for one episode."""
    transition = reset_and_build_transition(env, env_proc, act_proc)
    out = []
    while True:
        start = time.perf_counter()
        observation = {k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in STATE_KEYS}
        new = step_env_and_process_transition(
            env=env, transition=transition, action=policy_action.clone(), env_processor=env_proc, action_processor=act_proc
        )
        next_observation = {k: v for k, v in new[TransitionKey.OBSERVATION].items() if k in STATE_KEYS}
        info = new[TransitionKey.INFO]
        out.append(
            Transition(
                state=observation,
                action=new[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"],
                reward=new[TransitionKey.REWARD],
                next_state=next_observation,
                done=new.get(TransitionKey.DONE, False),
                truncated=new.get(TransitionKey.TRUNCATED, False),
                complementary_info={
                    "discrete_penalty": torch.tensor(
                        [new[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
                    ),
                    TeleopEvents.IS_INTERVENTION.value: bool(info.get(TeleopEvents.IS_INTERVENTION, False)),
                },
            )
        )
        transition = new
        if out[-1]["done"] or out[-1]["truncated"]:
            return out
        precise_sleep(max(1 / FPS - (time.perf_counter() - start), 0.0))


def test_provenance_tags_mark_each_takeover_once():
    flags = [False, True, True, False, True, False]
    tags = provenance_tags(flags, episode_index=7)
    assert [t["takeover_onset"] for t in tags] == [0, 1, 0, 0, 1, 0]
    assert [t["source"] for t in tags] == [2, 1, 1, 2, 1, 2]
    assert [t["step_index"] for t in tags] == [0, 1, 2, 3, 4, 5]
    assert {t["episode_index"] for t in tags} == {7}
    demo = provenance_tags(flags, episode_index=0, demo=True)
    assert {t["source"] for t in demo} == {SOURCE_DEMO} and not any(t["takeover_onset"] for t in demo)
    assert [t["is_intervention"] for t in demo] == [0, 1, 1, 0, 1, 0]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """Two online episodes with a takeover each, and a demo dataset, through the real pipeline."""
    from .conftest import URDF

    tmp = tmp_path_factory.mktemp("takeover")
    policy_action = torch.tensor([[0.0, -1.0, 0.0]])  # what "the policy" asks for: -y

    cfg = env_config(URDF, tmp)
    env, teleop = make_robot_env(cfg)
    env_proc, act_proc = make_processors(env, teleop, cfg)
    online = [
        tag_episode(run_actor_episode(env, env_proc, act_proc, policy_action), episode_index=i) for i in range(2)
    ]
    env.close()

    demo_cfg = GymManipulatorConfig(
        env=env_config(URDF, tmp, always_intervene=True),
        dataset=DatasetConfig(
            repo_id="local/takeover_test_demos", task="reach", root=str(tmp / "demos"), num_episodes_to_record=2
        ),
        mode="record",
    )
    env, teleop = make_robot_env(demo_cfg.env)
    env_proc, act_proc = make_processors(env, teleop, demo_cfg.env)
    dataset = record(env, env_proc, act_proc, demo_cfg)
    return {"online": online, "demo_root": tmp / "demos", "dataset": dataset}


def test_takeover_steps_are_flagged_and_carry_the_human_action(recorded):
    for episode_index, episode in enumerate(recorded["online"]):
        assert len(episode) == EPISODE_STEPS
        for step, transition in enumerate(episode):
            info = transition["complementary_info"]
            human = TAKEOVER[0] <= step < TAKEOVER[1]
            assert info["is_intervention"] == float(human), step
            assert info["source"] == (SOURCE_INTERVENTION if human else SOURCE_POLICY)
            assert info["takeover_onset"] == float(step == TAKEOVER[0])
            assert info["episode_index"] == episode_index and info["step_index"] == step
            action = transition["action"].reshape(-1)
            if human:  # the scripted operator drives up (+z) and back in y, not the policy's -y
                assert action[2] == pytest.approx(1.0) and action[1] > 0
            else:
                np.testing.assert_allclose(action.numpy(), [0, -1, 0], atol=1e-6)


def test_executed_motion_matches_who_was_driving(recorded):
    """The stored states move in -y under the policy and in +z under the human."""
    episode = recorded["online"][0]
    ee = np.array([t["state"]["observation.state"].reshape(-1)[-3:].numpy() for t in episode])
    moves = np.diff(ee, axis=0)  # moves[i] is the effect of step i
    for step in range(1, EPISODE_STEPS - 1):
        human = TAKEOVER[0] <= step < TAKEOVER[1]
        if human:
            assert moves[step, 2] > 0.004 and moves[step, 1] >= -1e-4, (step, moves[step])
        else:
            assert moves[step, 1] < -0.008 and abs(moves[step, 2]) < 1e-3, (step, moves[step])


def test_sampled_batch_keeps_provenance_aligned_across_both_buffers(recorded):
    from lerobot.datasets import LeRobotDataset

    from hilserl_gen3_lite import compat

    compat.apply()
    # Loaded as the learner loads it: lerobot's make_dataset passes return_uint8=True, so
    # images arrive as uint8 in [0, 255]. (Loading with the default float images, as this
    # test first did, hid that the offline buffer then held images 255 times brighter
    # than the online one; compat.apply() rescales them.)
    dataset = LeRobotDataset("local/takeover_test_demos", root=recorded["demo_root"], return_uint8=True)
    assert dataset[0]["observation.images.front"].dtype == torch.uint8
    assert {f"complementary_info.{k}" for k in COMPLEMENTARY_KEYS} <= set(dataset.features)

    offline = ReplayBuffer.from_lerobot_dataset(
        dataset, device="cpu", state_keys=STATE_KEYS, storage_device="cpu", optimize_memory=True, capacity=1000
    )
    n_demo = len(offline)
    online = ReplayBuffer(capacity=1000, device="cpu", state_keys=STATE_KEYS, storage_device="cpu", optimize_memory=True)
    # What lerobot.rl.learner.process_transitions does with each incoming transition:
    for episode in recorded["online"]:
        for transition in episode:
            online.add(**transition)
            if transition["complementary_info"]["is_intervention"]:
                offline.add(**transition)

    check_buffer_keys(online, offline)  # same keys on both sides
    assert set(online.complementary_info_keys) == set(COMPLEMENTARY_KEYS)
    assert len(offline) == n_demo + 2 * (TAKEOVER[1] - TAKEOVER[0])

    torch.manual_seed(0)
    mixer = OnlineOfflineMixer(online_buffer=online, offline_buffer=offline, online_ratio=0.5)
    seen_sources, onsets = set(), 0
    for _ in range(20):
        batch = mixer.sample(64)
        info = batch["complementary_info"]
        n = batch["action"].shape[0]
        for key in COMPLEMENTARY_KEYS:
            assert info[key].shape[0] == n, key  # every field covers the whole batch
        source, flag, onset = info["source"], info["is_intervention"], info["takeover_onset"]
        step, episode = info["step_index"], info["episode_index"]
        action = batch["action"]

        policy = source == SOURCE_POLICY
        human = source == SOURCE_INTERVENTION
        demo = source == SOURCE_DEMO
        assert (policy | human | demo).all()
        # rows are aligned: the tags agree with each other and with the action in the same row
        assert (flag[policy] == 0).all() and (flag[human] == 1).all() and (flag[demo] == 1).all()
        assert ((step[human] >= TAKEOVER[0]) & (step[human] < TAKEOVER[1])).all()
        assert (onset[human] == (step[human] == TAKEOVER[0]).float()).all()
        assert (onset[~human] == 0).all()
        assert (episode <= 1).all()
        assert torch.allclose(action[policy], torch.tensor([0.0, -1.0, 0.0]).expand(int(policy.sum()), 3))
        assert (action[human | demo][:, 2] > 0.9).all()
        # online half first, then the offline half (demos and interventions only)
        assert not policy[n // 2 :].any()
        # images from both buffers are on the same [0, 1] scale, demos included
        images = batch["state"]["observation.images.front"]
        assert images.min() >= 0.0 and images.max() <= 1.0
        if demo.any() and policy.any():
            assert abs(float(images[demo].mean()) - float(images[policy].mean())) < 0.2

        fractions = batch_source_fractions(batch)
        assert fractions["batch_frac_demo"] + fractions["batch_frac_intervention"] + fractions[
            "batch_frac_policy"
        ] == pytest.approx(1.0)
        seen_sources |= set(source.int().tolist())
        onsets += int(onset.sum())
    assert seen_sources == {SOURCE_DEMO, SOURCE_INTERVENTION, SOURCE_POLICY}
    assert onsets > 0


def test_learner_refuses_buffers_with_different_keys(recorded):
    online = ReplayBuffer(capacity=10, device="cpu", state_keys=STATE_KEYS, storage_device="cpu")
    offline = ReplayBuffer(capacity=10, device="cpu", state_keys=STATE_KEYS, storage_device="cpu")
    transition = dict(recorded["online"][0][0])
    online.add(**transition)
    stripped = {**transition, "complementary_info": {"discrete_penalty": torch.tensor([0.0])}}
    offline.add(**stripped)  # e.g. demos recorded with stock gym_manipulator
    with pytest.raises(RuntimeError, match="different complementary_info keys"):
        check_buffer_keys(online, offline)
