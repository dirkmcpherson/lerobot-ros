"""Play back a recorded episode from the LeRobot dataset on the Kinova Gen3.

Usage:
    python playback_kinova_trajectory.py [--episode N] [--dataset PATH]

The script:
  1. Loads the requested episode from the parquet dataset.
  2. Reads the first frame's observation.state and moves the robot there,
     falling back to HOME_POSITION if state data is unavailable.
  3. Replays the recorded actions at the original FPS.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.utils.utils import init_logging
from lerobot_backends import BackendRobot
from lerobot_backends.ros2.config import KinovaGen3Config

init_logging()
logger = logging.getLogger(__name__)

# Fallback if the dataset does not contain a starting state
HOME_POSITION = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]

DATASET_ROOT = Path("data/lerobot/kinova_gen3_reach")
FPS = 10
SETTLE_SEC = 6.0  # Time to wait after moving to start position


def load_episode(dataset_root: Path, episode_index: int) -> pd.DataFrame:
    """Load all frames for a given episode from the parquet chunks."""
    chunk_dir = dataset_root / "data"
    parquet_files = sorted(chunk_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {chunk_dir}")

    frames = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        episode_frames = df[df["episode_index"] == episode_index]
        if not episode_frames.empty:
            frames.append(episode_frames)

    if not frames:
        raise ValueError(
            f"Episode {episode_index} not found in dataset. "
            f"Available episodes: 0-{df['episode_index'].max()}"
        )

    episode_df = pd.concat(frames).sort_values("frame_index").reset_index(drop=True)
    logger.info(f"Loaded episode {episode_index}: {len(episode_df)} frames")
    return episode_df


def get_start_position(episode_df: pd.DataFrame) -> list[float]:
    """Return the starting joint positions from the first frame, or HOME as fallback."""
    first_row = episode_df.iloc[0]
    if "observation.state" in episode_df.columns:
        state = first_row["observation.state"]
        if state is not None and len(state) > 0:
            return list(float(v) for v in state)
    logger.warning("observation.state not found in dataset -- using HOME_POSITION as start.")
    return HOME_POSITION


def main():
    parser = argparse.ArgumentParser(description="Play back a recorded Kinova Gen3 episode.")
    parser.add_argument(
        "--episode", type=int, default=0, help="Episode index to replay (default: 0)"
    )
    parser.add_argument(
        "--dataset", type=Path, default=DATASET_ROOT, help="Path to the dataset root directory"
    )
    args = parser.parse_args()

    # Load episode data
    episode_df = load_episode(args.dataset, args.episode)

    # Determine start position
    start_position = get_start_position(episode_df)
    source = "dataset first frame" if start_position != HOME_POSITION else "HOME_POSITION (fallback)"
    logger.info(f"Start position source: {source}")

    # Connect robot via backend
    logger.info("Connecting to robot...")
    config = KinovaGen3Config()
    config.home_position = start_position
    config.home_settle_sec = SETTLE_SEC
    robot = BackendRobot(config)
    robot.connect()
    time.sleep(2.0)

    for col in episode_df.columns:
        print(f"{col} - {episode_df[col].shape} - {episode_df[col][0].shape}")

    joint_names = config.ros2_interface.arm_joint_names

    try:
        # Move to start via backend reset
        logger.info(f"Moving to start position: {[f'{v:.3f}' for v in start_position]}")
        robot.backend.reset()

        # Play back actions
        logger.info(f"Playing back {len(episode_df)} frames at {FPS} FPS...")
        for i, row in episode_df.iterrows():
            step_start = time.perf_counter()

            action = list(float(v) for v in row["action"])
            action_dict = {f"{name}.pos": val for name, val in zip(joint_names, action)}
            robot.send_action(action_dict)

            if int(row["frame_index"]) % 10 == 0:
                logger.info(f"  Frame {int(row['frame_index']):3d} | action: {[f'{v:.3f}' for v in action]}")

            dt = time.perf_counter() - step_start
            sleep_time = 1.0 / FPS - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Playback complete.")

    except KeyboardInterrupt:
        logger.info("Playback interrupted.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
