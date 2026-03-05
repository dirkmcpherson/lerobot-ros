
import time
import logging
import numpy as np
from pathlib import Path

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

# Backend imports
from lerobot_backends import BackendRobot
from lerobot_backends.ros2.config import KinovaGen3Config

# Configure logging
init_logging()
logger = logging.getLogger(__name__)

# --- Task Definition ---
HOME_POSITION = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
TARGET_POSITION = [1.57, 2.0, 3.14, -0.5, 0.0, 0.0, 1.57]

# --- Dataset Config ---
DATASET_REPO_ID = "lerobot/kinova_gen3_reach"
ROOT_DIR = Path("data/lerobot/kinova_gen3_reach")
FPS = 10
NUM_EPISODES = 50
EPISODE_LENGTH_FRAMES = 100  # 10 seconds at 10 FPS
ROBOT_TYPE = "kinova_gen3"


def main():
    logger.info("Initializing Robot...")
    config = KinovaGen3Config()
    # Override home_position to match this script's task
    config.home_position = HOME_POSITION
    robot = BackendRobot(config)
    robot.connect()
    time.sleep(2.0)

    joint_names = config.ros2_interface.arm_joint_names
    num_joints = len(joint_names)

    dataset_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": joint_names,
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": ["target_" + n for n in joint_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": joint_names,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        root=ROOT_DIR,
        robot_type=ROBOT_TYPE,
        features=dataset_features,
        use_videos=False,
    )

    target_vec = np.array(TARGET_POSITION, dtype=np.float32)

    logger.info(f"Target position: {TARGET_POSITION}")
    logger.info(f"Starting data collection for {NUM_EPISODES} episodes...")

    try:
        for ep_idx in range(NUM_EPISODES):
            logger.info(f"Episode {ep_idx + 1}/{NUM_EPISODES}: resetting to home...")
            robot.backend.reset()

            # Build a linear trajectory from home to target over the episode
            trajectory = np.linspace(HOME_POSITION, TARGET_POSITION, EPISODE_LENGTH_FRAMES)

            for i in range(EPISODE_LENGTH_FRAMES):
                step_start = time.perf_counter()

                obs = robot.get_observation()
                current_state = np.array(
                    [obs[f"{j}.pos"] for j in joint_names], dtype=np.float32
                )

                if i % 10 == 0:
                    logger.info(
                        f"  Frame {i:3d} | state: {[f'{v:.3f}' for v in current_state[:3]]}"
                    )

                # Action: next waypoint along the interpolated trajectory
                action_vec = trajectory[min(i + 1, EPISODE_LENGTH_FRAMES - 1)].astype(np.float32)

                action_dict = {f"{name}.pos": val for name, val in zip(joint_names, action_vec)}
                robot.send_action(action_dict)

                frame = {
                    "observation.state": current_state,
                    "observation.environment_state": target_vec,
                    "action": action_vec,
                    "task": "Reach target: extended down and to the left",
                }
                dataset.add_frame(frame)

                dt = time.perf_counter() - step_start
                sleep_time = 1.0 / FPS - dt
                if sleep_time > 0:
                    time.sleep(sleep_time)

            dataset.save_episode()
            dataset.episode_buffer = None

    except KeyboardInterrupt:
        logger.info("Data collection interrupted.")
    finally:
        robot.disconnect()
        logger.info(f"Dataset saved to {ROOT_DIR}")


if __name__ == "__main__":
    main()
