"""Auto-generated 'move to the left' demos for DreamZero fine-tuning.

Each episode:
  1. Resets to home (with small randomization on joints 1-3 for variation).
  2. Slowly increments joint_1 (base yaw) by a fixed delta per step.
  3. Records observation.state + action + wrist/exterior camera frames.
"""

import argparse
import logging
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import lerobot.cameras.opencv.camera_opencv as _lerobot_opencv_cam
from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

from lerobot_backends import BackendRobot
from lerobot_backends.ros2.config import KinovaGen3LiteConfig

# Force V4L2 backend so MJPG negotiation works.
_lerobot_opencv_cam.get_cv2_backend = lambda: int(cv2.CAP_V4L2)

init_logging()
logger = logging.getLogger(__name__)

FPS = 30
EPISODE_DURATION_SEC = 3.0
JOINT1_DELTA_PER_STEP = 0.004  # rad/step → ~0.36 rad over 3s ≈ 21°
HOME_JITTER_RAD = 0.05         # small randomization per episode

CAMERA_CONFIGS = {
    "observation.images.wrist": OpenCVCameraConfig(
        index_or_path=Path("/dev/video0"),
        fps=30, width=320, height=240, fourcc="MJPG",
    ),
    "observation.images.exterior": OpenCVCameraConfig(
        index_or_path=Path("/dev/video2"),
        fps=30, width=320, height=240, fourcc="MJPG",
        rotation=Cv2Rotation.ROTATE_180,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--dataset-name", type=str, default="kinova_gen3_lite_smoke")
    parser.add_argument("--task", type=str, default="move to the left")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--direction", choices=["left", "right"], default="left",
                        help="Which way to swing joint_1 (flip if 'left' goes the wrong way)")
    args = parser.parse_args()

    delta_sign = -1.0 if args.direction == "left" else 1.0

    root_dir = Path(f"data/lerobot/{args.dataset_name}")
    repo_id = f"lerobot/{args.dataset_name}"

    if args.overwrite and root_dir.exists():
        logger.info(f"--overwrite: removing {root_dir}")
        shutil.rmtree(root_dir)

    config = KinovaGen3LiteConfig()
    config.cameras = dict(CAMERA_CONFIGS)
    iface = config.ros2_interface
    arm_joint_names = iface.arm_joint_names
    gripper_joint_name = iface.gripper_joint_name
    all_joint_names = arm_joint_names + [gripper_joint_name]
    state_dim = len(all_joint_names)

    robot = BackendRobot(config)
    logger.info("Connecting to gen3_lite + cameras...")
    robot.connect()
    time.sleep(2.0)

    dataset_features = {
        "observation.state": {
            "dtype": "float32", "shape": (state_dim,), "names": all_joint_names,
        },
        "action": {
            "dtype": "float32", "shape": (state_dim,), "names": all_joint_names,
        },
    }
    for cam_key, cam_cfg in CAMERA_CONFIGS.items():
        dataset_features[cam_key] = {
            "dtype": "video",
            "shape": (cam_cfg.height, cam_cfg.width, 3),
            "names": ["height", "width", "channels"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        root=root_dir,
        robot_type="kinova_gen3_lite",
        features=dataset_features,
        use_videos=True,
    )

    frames_per_episode = int(EPISODE_DURATION_SEC * FPS)
    saved = 0

    try:
        for ep in range(args.num_episodes):
            logger.info(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")

            # Reset home + jitter joint_1..3 a bit
            home = list(config.home_position)
            for i in range(3):
                home[i] += random.uniform(-HOME_JITTER_RAD, HOME_JITTER_RAD)
            config.home_position = home
            robot.backend.reset()
            time.sleep(0.5)

            obs = robot.get_observation()
            current_target = np.array([obs[f"{j}.pos"] for j in arm_joint_names], dtype=np.float32)
            gripper_obs = obs.get(f"{gripper_joint_name}.pos", 0.0)

            episode_frames = []
            for f in range(frames_per_episode):
                step_start = time.perf_counter()

                obs = robot.get_observation()
                arm_state = [obs[f"{j}.pos"] for j in arm_joint_names]
                gripper_obs = obs.get(f"{gripper_joint_name}.pos", 0.0)
                state_vec = np.array(arm_state + [gripper_obs], dtype=np.float32)

                # Step the commanded target on joint_1
                current_target[0] += delta_sign * JOINT1_DELTA_PER_STEP
                action_vec = np.array(list(current_target) + [gripper_obs], dtype=np.float32)

                action_dict = {f"{j}.pos": float(action_vec[i]) for i, j in enumerate(arm_joint_names)}
                action_dict["gripper.pos"] = float(gripper_obs)
                robot.backend.send_action(action_dict)

                frame = {
                    "observation.state": state_vec,
                    "action": action_vec,
                    "task": args.task,
                }
                for cam_key in CAMERA_CONFIGS:
                    img = obs.get(cam_key)
                    if img is None:
                        logger.warning(f"Missing {cam_key} at frame {f}")
                        continue
                    frame[cam_key] = img
                episode_frames.append(frame)

                if f % 15 == 0:
                    logger.info(f"  Frame {f:3d} | joint_1: {state_vec[0]:+.3f} → cmd {action_vec[0]:+.3f}")

                dt = time.perf_counter() - step_start
                sleep_time = 1.0 / FPS - dt
                if sleep_time > 0:
                    time.sleep(sleep_time)

            logger.info(f"Saving episode ({len(episode_frames)} frames)...")
            for fr in episode_frames:
                dataset.add_frame(fr)
            dataset.save_episode()
            dataset.episode_buffer = None
            saved += 1
            logger.info(f"Saved. Total: {saved}")

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        robot.disconnect()
        logger.info(f"Done. {saved} episodes saved to {root_dir}")


if __name__ == "__main__":
    main()
