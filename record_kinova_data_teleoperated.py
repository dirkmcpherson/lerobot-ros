"""Teleoperated data recording for the Kinova Gen3 reach task.

The operator controls the robot in real time using a UserInput device.
Each episode runs until the operator presses Enter (save) or D (discard).
On save the episode is written to the dataset; on discard it is dropped.

Usage:
    python record_kinova_data_teleoperated.py [--input keyboard|spacemouse]
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

from lerobot_robot_ros.config import ActionType, GripperActionType, ROS2Config, ROS2InterfaceConfig
from lerobot_robot_ros.robot import ROS2Robot
from user_input import KeyboardUserInput, SpacemouseUserInput, UserInput

init_logging()
logger = logging.getLogger(__name__)

# --- Task definition (keep in sync with record_kinova_data.py / eval) --------
HOME_POSITION = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
TARGET_POSITION = [1.57, 2.0, 3.14, -0.5, 0.0, 0.0, 1.57]

# --- Dataset config ----------------------------------------------------------
DATASET_REPO_ID = "lerobot/kinova_gen3_teleop"
ROOT_DIR = Path("data/lerobot/kinova_gen3_teleop")
FPS = 10
MAX_EPISODE_FRAMES = 300  # safety cap: 30 s
HOME_SETTLE_SEC = 6.0
ROBOT_TYPE = "kinova_gen3"


@dataclass
class KinovaGen3Config(ROS2Config):
    action_type: ActionType = ActionType.JOINT_TRAJECTORY

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            arm_joint_names=[
                "joint_1", "joint_2", "joint_3", "joint_4",
                "joint_5", "joint_6", "joint_7",
            ],
            gripper_joint_name=None,
            namespace="",
            arm_topic="/joint_trajectory_controller/joint_trajectory",
            min_joint_positions=[-6.2832, -2.24, -6.2832, -2.57, -6.2832, -2.09, -6.2832],
            max_joint_positions=[6.2832, 2.24, 6.2832, 2.57, 6.2832, 2.09, 6.2832],
            gripper_open_position=0.0,
            gripper_close_position=0.8,
            gripper_action_type=GripperActionType.ACTION,
        )
    )


def make_input_device(name: str, config: KinovaGen3Config, urdf_path: Optional[str] = None) -> UserInput:
    iface = config.ros2_interface
    if name == "keyboard":
        return KeyboardUserInput(
            joint_names=iface.arm_joint_names,
            min_joint_positions=iface.min_joint_positions,
            max_joint_positions=iface.max_joint_positions,
            step_size=0.05,
        )
    if name == "spacemouse":
        if urdf_path is None:
            raise ValueError("--urdf is required when using the spacemouse input device")
        return SpacemouseUserInput(
            joint_names=iface.arm_joint_names,
            min_joint_positions=iface.min_joint_positions,
            max_joint_positions=iface.max_joint_positions,
            urdf_path=urdf_path,
        )
    raise ValueError(f"Unknown input device '{name}'. Available: keyboard, spacemouse")


def reset_to_home(robot: ROS2Robot, joint_names: list[str]) -> list[float]:
    """Move to home and return the actual position after settling."""
    logger.info(f"Resetting to home: {HOME_POSITION}")
    robot.ros2_interface.send_joint_position_command(
        HOME_POSITION, unnormalize=False, time_from_start_sec=5.0
    )
    time.sleep(HOME_SETTLE_SEC)
    obs = robot.get_observation()
    actual = [obs[f"{j}.pos"] for j in joint_names]
    logger.info(f"At home (actual): {[f'{v:.3f}' for v in actual]}")
    return actual


def main():
    parser = argparse.ArgumentParser(description="Teleoperated Kinova Gen3 data recording.")
    parser.add_argument(
        "--input", default="keyboard",
        help="Input device to use (default: keyboard)"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50,
        help="Target number of saved episodes (default: 50)"
    )
    parser.add_argument(
        "--urdf", default=None,
        help="Path to robot URDF file (required for spacemouse)"
    )
    args = parser.parse_args()

    config = KinovaGen3Config()
    joint_names = config.ros2_interface.arm_joint_names
    num_joints = len(joint_names)

    # Build input device
    user_input = make_input_device(args.input, config, urdf_path=args.urdf)

    # Connect robot
    logger.info("Connecting to robot...")
    robot = ROS2Robot(config)
    robot.connect()
    time.sleep(2.0)

    # Connect input device
    user_input.connect()

    dataset_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": joint_names,
        },
        # Kept for goal-conditioned policy training; always TARGET_POSITION here.
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
    saved_episodes = 0

    logger.info(f"Starting teleoperated recording. Goal: {args.num_episodes} saved episodes.")

    try:
        while saved_episodes < args.num_episodes and not user_input.quit_requested:
            logger.info(
                f"\n=== Episode {saved_episodes + 1}/{args.num_episodes} ==="
            )

            # Reset robot to home and sync input device
            actual_home = reset_to_home(robot, joint_names)
            user_input.reset(actual_home)

            episode_frames: list[dict] = []
            frame_count = 0

            logger.info("Recording. Use input device to move robot. Enter=save, D=discard.")

            while not user_input.episode_end_requested:
                if frame_count >= MAX_EPISODE_FRAMES:
                    logger.warning(
                        f"Reached max frames ({MAX_EPISODE_FRAMES}). Auto-saving episode."
                    )
                    break

                step_start = time.perf_counter()

                # Observation
                obs = robot.get_observation()
                current_state = np.array(
                    [obs[f"{j}.pos"] for j in joint_names], dtype=np.float32
                )

                # Action from input device
                action_vec = np.array(
                    user_input.get_action(current_state.tolist()), dtype=np.float32
                )

                # Send to robot, using the device's preferred trajectory horizon.
                # Velocity-integrated devices (spacemouse) use a short horizon
                # (~0.1 s) so incremental steps execute immediately.
                robot.ros2_interface.send_joint_position_command(
                    action_vec.tolist(),
                    unnormalize=False,
                    time_from_start_sec=user_input.time_from_start_sec,
                )

                if frame_count % 10 == 0:
                    logger.info(
                        f"  Frame {frame_count:3d} | "
                        f"state: {[f'{v:.3f}' for v in current_state[:3]]} | "
                        f"action: {[f'{v:.3f}' for v in action_vec[:3]]}"
                    )

                episode_frames.append({
                    "observation.state": current_state,
                    "observation.environment_state": target_vec,
                    "action": action_vec,
                    "task": "Reach target: extended down and to the left",
                })
                frame_count += 1

                dt = time.perf_counter() - step_start
                sleep_time = 1.0 / FPS - dt
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Episode ended — save or discard
            if user_input.discard_requested:
                logger.info(f"Episode discarded ({frame_count} frames dropped).")
            elif frame_count == 0:
                logger.warning("Empty episode — skipping.")
            else:
                logger.info(f"Saving episode ({frame_count} frames)...")
                for frame in episode_frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                dataset.episode_buffer = None
                saved_episodes += 1
                logger.info(f"Saved. Total saved episodes: {saved_episodes}")

            if user_input.quit_requested:
                break

    except KeyboardInterrupt:
        logger.info("Recording interrupted.")
    finally:
        user_input.disconnect()
        robot.disconnect()
        logger.info(f"Done. {saved_episodes} episodes saved to {ROOT_DIR}")


if __name__ == "__main__":
    main()
