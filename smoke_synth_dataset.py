#!/usr/bin/env python3
"""Synthesize a gen3_lite-shaped LeRobot dataset with NO robot and NO ROS.

This is the first half of a pure-software smoke test of the collect->train->deploy
pipeline on this box (ROS 2 Lyrical, Python 3.14, lerobot 0.5.0, CPU-only torch).
It exists to validate the *lerobot-side* mechanics that have never been exercised on
this stack — dataset format, DP trainability, checkpoint round-trip — without needing
the physical arm, a camera, or any motion.

The feature signature matches the REAL Kinova Gen3 Lite so the schema and the trained
policy config transfer directly to a real record->train run later:

    observation.state             : 7-dim  (joint_1..joint_6 + gripper)
    observation.environment_state : 7-dim  (per-episode target pose)
    action                        : 7-dim  (next waypoint toward the target)

State-only (``use_videos=False``): DiffusionPolicy accepts this because it requires
"at least one image OR observation.environment_state" (verified against the installed
DiffusionConfig.validate_features).

The synthetic task is a trivial, learnable reach: each episode picks a random target
pose and drives a straight line from home to it, so action = the next waypoint. A DP
model should fit this easily, which is the point — we are testing plumbing, not
difficulty.

Episodes are 60 frames so DP's windowing (horizon=16, n_obs_steps=2,
drop_n_last_frames=7) yields plenty of training samples per episode.
"""
import logging
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

init_logging()
logger = logging.getLogger(__name__)

# --- gen3_lite feature signature ---------------------------------------------
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
STATE_DIM = len(JOINT_NAMES)  # 7

# Real gen3_lite home (6 joints) + gripper open (0.0)
HOME = np.array([0.0, -0.52, 1.22, -1.22, -0.52, 0.0, 0.0], dtype=np.float32)
# Per-joint sampling half-range for random targets (well inside the URDF limits;
# gripper target in [0, 0.85]).
TARGET_SPAN = np.array([1.0, 0.8, 0.8, 0.8, 0.8, 1.0, 0.42], dtype=np.float32)
TARGET_MID = np.array([0.0, 0.0, 0.6, -0.6, 0.0, 0.0, 0.42], dtype=np.float32)

# --- dataset config -----------------------------------------------------------
REPO_ID = "kinova_gen3_lite_smoke"
ROOT_DIR = Path("data/lerobot/kinova_gen3_lite_smoke")
FPS = 10
NUM_EPISODES = 15
EPISODE_LENGTH_FRAMES = 60
SEED = 0
TASK = "synthetic reach (smoke test)"


def main():
    if ROOT_DIR.exists():
        logger.info(f"Removing existing dataset at {ROOT_DIR}")
        shutil.rmtree(ROOT_DIR)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": JOINT_NAMES,
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["target_" + n for n in JOINT_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": JOINT_NAMES,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=ROOT_DIR,
        robot_type="kinova_gen3_lite",
        features=features,
        use_videos=False,
    )

    rng = np.random.default_rng(SEED)
    logger.info(f"Synthesizing {NUM_EPISODES} episodes x {EPISODE_LENGTH_FRAMES} frames...")

    for ep_idx in range(NUM_EPISODES):
        target = (TARGET_MID + (rng.random(STATE_DIM).astype(np.float32) - 0.5) * 2.0 * TARGET_SPAN)
        target = target.astype(np.float32)
        # Straight-line trajectory home -> target across the episode.
        traj = np.linspace(HOME, target, EPISODE_LENGTH_FRAMES).astype(np.float32)

        for i in range(EPISODE_LENGTH_FRAMES):
            state = traj[i]
            # Action = next waypoint (last frame repeats the final pose).
            action = traj[min(i + 1, EPISODE_LENGTH_FRAMES - 1)]
            frame = {
                "observation.state": state,
                "observation.environment_state": target,
                "action": action,
                "task": TASK,
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        dataset.episode_buffer = None
        logger.info(f"  episode {ep_idx + 1}/{NUM_EPISODES} saved")

    logger.info(f"Done. Dataset at {ROOT_DIR.resolve()}")
    logger.info(f"  {NUM_EPISODES * EPISODE_LENGTH_FRAMES} frames, {STATE_DIM}-dim state/action, no video")


if __name__ == "__main__":
    main()
