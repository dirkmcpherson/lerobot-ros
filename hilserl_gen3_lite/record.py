"""Record demonstrations for HIL-SERL on the Gen3 Lite.

Same configuration file as lerobot's ``python -m lerobot.rl.gym_manipulator`` in record
mode (``env`` + ``dataset``), same loop: the robot stays still unless the operator holds
the takeover trigger, and the teleoperator's action is what is stored. Differences:

  * this repo's environment and processors (``env.py``),
  * every frame carries the provenance columns (``provenance.py``) with
    ``source = demo``, so the offline replay buffer built from this dataset has the
    same complementary_info keys as the transitions the actor sends.

    python -m hilserl_gen3_lite.record --config_path=config/hilserl/env_gen3_lite_reach.json \
        --run.confirm_motion=true
"""

import logging
import sys
import time

import numpy as np
import torch
from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.processor import TransitionKey
from lerobot.rl.gym_manipulator import (
    GymManipulatorConfig,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD
from lerobot.utils.robot_utils import precise_sleep

from .env import make_processors, make_robot_env
from .provenance import COMPLEMENTARY_KEYS, DISCRETE_PENALTY, provenance_tags
from .settings import RunSettings, pop_run_settings, require_motion_confirmation

_SETTINGS = RunSettings()


def dataset_features(first_observation: dict, action_dim: int) -> dict:
    features = {
        ACTION: {"dtype": "float32", "shape": (action_dim,), "names": None},
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
    }
    for key in COMPLEMENTARY_KEYS:
        features[f"complementary_info.{key}"] = {"dtype": "float32", "shape": (1,), "names": [key]}
    for key, value in first_observation.items():
        if key == OBS_STATE:
            features[key] = {"dtype": "float32", "shape": tuple(value.squeeze(0).shape), "names": None}
        elif "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": tuple(value.squeeze(0).shape),
                "names": ["channels", "height", "width"],
            }
    return features


def record(env, env_processor, action_processor, cfg: GymManipulatorConfig) -> LeRobotDataset:
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else False
    neutral = torch.tensor([0.0, 0.0, 0.0] + ([1.0] if use_gripper else []), dtype=torch.float32)
    dt = 1.0 / cfg.env.fps

    transition = reset_and_build_transition(env, env_processor, action_processor)
    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.env.fps,
        root=cfg.dataset.root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=dataset_features(transition[TransitionKey.OBSERVATION], neutral.numel()),
    )
    print(
        f"Recording {cfg.dataset.num_episodes_to_record} demos at {cfg.env.fps} Hz.\n"
        "Hold the takeover button (RB) and drive; the arm stays still otherwise.\n"
        "Success key ends the episode with reward 1; the re-record key discards it. Ctrl+C stops.",
        flush=True,
    )

    episode, frames, human_flags = 0, [], []
    try:
        while episode < cfg.dataset.num_episodes_to_record:
            start = time.perf_counter()
            observation = {
                k: v.squeeze(0).cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor) and (k == OBS_STATE or "image" in k)
            }
            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral.clone(),
                env_processor=env_processor,
                action_processor=action_processor,
            )
            done = bool(transition.get(TransitionKey.DONE, False))
            truncated = bool(transition.get(TransitionKey.TRUNCATED, False))
            info = transition[TransitionKey.INFO]
            data = transition[TransitionKey.COMPLEMENTARY_DATA]
            action = data.get("teleop_action", transition[TransitionKey.ACTION])
            frames.append(
                {
                    **observation,
                    ACTION: torch.as_tensor(action, dtype=torch.float32).reshape(-1).cpu(),
                    REWARD: np.array([float(transition[TransitionKey.REWARD])], dtype=np.float32),
                    DONE: np.array([done or truncated], dtype=bool),
                    f"complementary_info.{DISCRETE_PENALTY}": np.array(
                        [float(data.get(DISCRETE_PENALTY, 0.0))], dtype=np.float32
                    ),
                }
            )
            human_flags.append(bool(info.get(TeleopEvents.IS_INTERVENTION, False)))

            if done or truncated:
                reward = float(transition[TransitionKey.REWARD])
                if info.get(TeleopEvents.RERECORD_EPISODE, False):
                    logging.info(f"Discarding episode {episode} ({len(frames)} frames); record it again.")
                else:
                    for frame, tags in zip(frames, provenance_tags(human_flags, episode, demo=True), strict=True):
                        frame.update(
                            {f"complementary_info.{k}": np.array([v], dtype=np.float32) for k, v in tags.items()}
                        )
                        frame["task"] = cfg.dataset.task
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    logging.info(
                        f"Saved demo {episode}: {len(frames)} frames, {sum(human_flags)} human-driven, "
                        f"final reward {reward}"
                    )
                    episode += 1
                frames, human_flags = [], []
                if episode < cfg.dataset.num_episodes_to_record:
                    transition = reset_and_build_transition(env, env_processor, action_processor)
            precise_sleep(max(dt - (time.perf_counter() - start), 0.0))
    except KeyboardInterrupt:
        logging.info("Interrupted; the unfinished episode is dropped.")
    finally:
        try:
            env.close()  # holds the arm, then disconnects
        finally:
            dataset.finalize()
    if cfg.dataset.push_to_hub:
        dataset.push_to_hub()
    return dataset


@parser.wrap()
def record_cli(cfg: GymManipulatorConfig) -> None:
    import lerobot_robot_ros  # noqa: F401
    import lerobot_teleoperator_devices  # noqa: F401

    logging.basicConfig(level=logging.INFO)
    require_motion_confirmation(cfg.env.robot, _SETTINGS)
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)
    record(env, env_processor, action_processor, cfg)


def main() -> None:
    global _SETTINGS
    _SETTINGS, sys.argv = pop_run_settings()
    from . import compat

    compat.apply()
    # Device plugins must be registered before lerobot parses --env.robot.type / --env.teleop.type.
    import lerobot_robot_ros  # noqa: F401
    import lerobot_teleoperator_devices  # noqa: F401

    record_cli()


if __name__ == "__main__":
    main()
