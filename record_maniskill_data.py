"""Record reach-target demonstrations in ManiSkill.

Each episode linearly interpolates from the Panda home pose to a target
joint pose (with per-episode noise).  The target pose is stored as
observation.environment_state so the policy can be conditioned on it.

Usage:
    python record_maniskill_data.py --num-episodes 50
    python record_maniskill_data.py --num-episodes 200 --overwrite
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

from lerobot_backends import BackendRobot
from lerobot_backends.maniskill.config import ManiSkillPickCubeConfig

init_logging()
logger = logging.getLogger(__name__)

# Panda home pose (what ManiSkill resets to, approximately)
HOME_POSITION = [0.0, 0.39, -0.05, -1.94, -0.03, 2.34, 0.81]

# Target: arm stretched forward and down
TARGET_POSITION = [0.5, 0.8, 0.3, -1.2, 0.2, 1.8, 0.4]

# Joint limits (from ManiSkill Panda action space)
JOINT_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

FPS = 20
EPISODE_LENGTH = 100  # 5 seconds at 20 FPS
DATASET_REPO_ID = "lerobot/maniskill_reach"
NUM_JOINTS = 7


def main():
    parser = argparse.ArgumentParser(description="Record ManiSkill reach data.")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true", help="Open viewer window")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root_dir = Path(f"data/{DATASET_REPO_ID}")

    render_mode = "human" if args.render else None
    config = ManiSkillPickCubeConfig(obs_mode="state", render_mode=render_mode)
    robot = BackendRobot(config)
    robot.connect()

    # Probe to get actual home position
    obs = robot.backend.reset()
    actual_home = np.array([obs[f"joint_{i+1}.pos"] for i in range(NUM_JOINTS)], dtype=np.float32)
    logger.info(f"Actual home: {[f'{v:.3f}' for v in actual_home]}")

    if root_dir.exists():
        if not args.overwrite:
            logger.error(f"{root_dir} exists. Use --overwrite to replace.")
            robot.disconnect()
            return
        shutil.rmtree(root_dir)

    joint_names = [f"joint_{i+1}" for i in range(NUM_JOINTS)]

    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        root=root_dir,
        robot_type="maniskill_panda",
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (NUM_JOINTS,),
                "names": joint_names,
            },
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (NUM_JOINTS,),
                "names": ["target_" + n for n in joint_names],
            },
            "action": {
                "dtype": "float32",
                "shape": (NUM_JOINTS,),
                "names": joint_names,
            },
        },
        use_videos=False,
    )

    target_base = np.array(TARGET_POSITION, dtype=np.float32)

    try:
        for ep in range(args.num_episodes):
            obs = robot.backend.reset()
            home = np.array([obs[f"joint_{i+1}.pos"] for i in range(NUM_JOINTS)], dtype=np.float32)

            # Add noise to the target each episode so the policy generalises
            noise = np.random.uniform(-0.15, 0.15, size=NUM_JOINTS).astype(np.float32)
            target = np.clip(target_base + noise, JOINT_LOW, JOINT_HIGH).astype(np.float32)
            target_vec = target.copy()

            trajectory = np.linspace(home, target, EPISODE_LENGTH).astype(np.float32)

            for i in range(EPISODE_LENGTH):
                current = np.array(
                    [obs[f"joint_{i+1}.pos"] for i in range(NUM_JOINTS)], dtype=np.float32
                )

                # Action: next waypoint
                action_vec = trajectory[min(i + 1, EPISODE_LENGTH - 1)]

                # Send action (arm joints only, no gripper for reach task)
                action_dict = {f"joint_{j+1}.pos": float(action_vec[j]) for j in range(NUM_JOINTS)}
                if config.gripper_joint_name:
                    action_dict["gripper.pos"] = 1.0  # keep gripper open
                robot.backend.send_action(action_dict)
                obs = robot.backend.get_observation()

                dataset.add_frame({
                    "observation.state": current,
                    "observation.environment_state": target_vec,
                    "action": action_vec,
                    "task": "Reach target joint pose",
                })

            dataset.save_episode()
            dataset.episode_buffer = None

            final = np.array(
                [obs[f"joint_{i+1}.pos"] for i in range(NUM_JOINTS)], dtype=np.float32
            )
            error = np.linalg.norm(final - target)
            logger.info(
                f"Episode {ep + 1}/{args.num_episodes} | "
                f"target_noise={np.linalg.norm(noise):.3f} | "
                f"final_error={error:.4f}"
            )

    except KeyboardInterrupt:
        logger.info("Recording interrupted.")
    finally:
        robot.disconnect()
        logger.info(f"Done. {dataset.num_episodes} episodes saved to {root_dir}")


if __name__ == "__main__":
    main()
