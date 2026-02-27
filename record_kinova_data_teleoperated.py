"""Teleoperated data recording for the Kinova Gen3 / Gen3 Lite reach task.

The operator controls the robot in real time using a UserInput device.
Each episode runs until the operator presses Enter (save) or D (discard).
On save the episode is written to the dataset; on discard it is dropped.

Usage:
    python record_kinova_data_teleoperated.py [--robot gen3|gen3_lite] [--input keyboard|spacemouse]

Robots
------
gen3      : Kinova Gen3 7-DOF (no gripper by default)
gen3_lite : Kinova Gen3 Lite 6-DOF + integrated 2-finger gripper

SpaceMouse button mapping (with gripper)
-----------------------------------------
  Left button        : save episode
  Right button       : toggle gripper open / closed
  Both buttons       : discard episode and quit
"""

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

from lerobot_robot_ros.config import ROS2Config, KinovaGen3Config, KinovaGen3LiteConfig
from lerobot_robot_ros.robot import ROS2Robot
from user_input import KeyboardUserInput, SpacemouseUserInput, UserInput

init_logging()
logger = logging.getLogger(__name__)

# --- Dataset config ----------------------------------------------------------
FPS = 10
MAX_EPISODE_FRAMES = 300  # safety cap: 30 s
HOME_SETTLE_SEC = 6.0


# Per-robot home and target positions (adjust as needed for your task).
# Gen3 home/target are for the 7-DOF reach task used in earlier experiments.
# Gen3 Lite positions are all-zero (neutral); tune to your actual task.
_ROBOT_CONFIGS: dict[str, tuple] = {
    "gen3": (
        KinovaGen3Config,
        [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57],   # home
        [1.57, 2.0, 3.14, -0.5, 0.0, 0.0, 1.57],      # target
        "lerobot/kinova_gen3_teleop",
        Path("data/lerobot/kinova_gen3_teleop"),
        "kinova_gen3",
    ),
    "gen3_lite": (
        KinovaGen3LiteConfig,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],               # home — adjust to your task
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],               # target — adjust to your task
        "lerobot/kinova_gen3_lite_teleop",
        Path("data/lerobot/kinova_gen3_lite_teleop"),
        "kinova_gen3_lite",
    ),
}


def make_input_device(name: str, config: ROS2Config, urdf_path: Optional[str] = None) -> UserInput:
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
            gripper_joint_name=iface.gripper_joint_name,
            gripper_open_position=iface.gripper_open_position,
            gripper_closed_position=iface.gripper_close_position,
        )
    raise ValueError(f"Unknown input device '{name}'. Available: keyboard, spacemouse")


def reset_to_home(
    robot: ROS2Robot,
    arm_joint_names: list[str],
    home_position: list[float],
    gripper_joint_name: Optional[str],
) -> list[float]:
    """Move arm to home, open gripper, return actual arm positions after settling."""
    logger.info(f"Resetting to home: {home_position}")
    robot.ros2_interface.send_joint_position_command(
        home_position, unnormalize=False, time_from_start_sec=5.0
    )
    if gripper_joint_name:
        robot.ros2_interface.send_gripper_command(
            robot.ros2_interface.config.gripper_open_position, unnormalize=False
        )
    time.sleep(HOME_SETTLE_SEC)
    obs = robot.get_observation()
    actual = [obs[f"{j}.pos"] for j in arm_joint_names]
    logger.info(f"At home (actual): {[f'{v:.3f}' for v in actual]}")
    return actual


def main():
    parser = argparse.ArgumentParser(description="Teleoperated Kinova data recording.")
    parser.add_argument(
        "--robot", default="gen3_lite", choices=list(_ROBOT_CONFIGS.keys()),
        help="Robot model to use (default: gen3_lite)"
    )
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
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete and recreate the dataset directory if it already exists"
    )

    args = parser.parse_args()

    config_cls, home_position, target_position, dataset_repo_id, root_dir, robot_type = (
        _ROBOT_CONFIGS[args.robot]
    )
    config: ROS2Config = config_cls()
    iface = config.ros2_interface
    arm_joint_names = iface.arm_joint_names
    gripper_joint_name: Optional[str] = iface.gripper_joint_name or None
    num_arm_joints = len(arm_joint_names)

    # Full state/action names include gripper when present
    all_joint_names = arm_joint_names + ([gripper_joint_name] if gripper_joint_name else [])
    state_dim = len(all_joint_names)

    if args.overwrite and root_dir.exists():
        logger.info(f"--overwrite: removing existing dataset at {root_dir}")
        shutil.rmtree(root_dir)

    # Build input device
    user_input = make_input_device(args.input, config, urdf_path=args.urdf)

    # Connect robot
    logger.info(f"Connecting to {args.robot}...")
    robot = ROS2Robot(config)
    robot.connect()
    time.sleep(2.0)

    # Connect input device
    user_input.connect()

    dataset_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": all_joint_names,
        },
        # Kept for goal-conditioned policy training; target covers arm joints only.
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (num_arm_joints,),
            "names": ["target_" + n for n in arm_joint_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": all_joint_names,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=dataset_repo_id,
        fps=FPS,
        root=root_dir,
        robot_type=robot_type,
        features=dataset_features,
        use_videos=False,
    )

    target_vec = np.array(target_position, dtype=np.float32)
    saved_episodes = 0

    logger.info(
        f"Starting teleoperated recording on {args.robot}. "
        f"Gripper: {'yes (' + gripper_joint_name + ')' if gripper_joint_name else 'no'}. "
        f"Goal: {args.num_episodes} saved episodes."
    )

    try:
        while saved_episodes < args.num_episodes and not user_input.quit_requested:
            logger.info(f"\n=== Episode {saved_episodes + 1}/{args.num_episodes} ===")

            # Reset robot to home and sync input device
            actual_home = reset_to_home(robot, arm_joint_names, home_position, gripper_joint_name)
            user_input.reset(actual_home)

            episode_frames: list[dict] = []
            frame_count = 0

            if gripper_joint_name:
                logger.info(
                    "Recording. SpaceMouse: move EE | Left=save | Right=gripper toggle | Both=discard+quit"
                )
            else:
                logger.info("Recording. Use input device to move robot. Enter=save, D=discard.")

            while not user_input.episode_end_requested:
                if frame_count >= MAX_EPISODE_FRAMES:
                    logger.warning(
                        f"Reached max frames ({MAX_EPISODE_FRAMES}). Auto-saving episode."
                    )
                    break

                step_start = time.perf_counter()

                # Observation — arm joints (+ gripper if configured)
                obs = robot.get_observation()
                arm_state = [obs[f"{j}.pos"] for j in arm_joint_names]
                if gripper_joint_name:
                    gripper_obs = obs.get(f"{gripper_joint_name}.pos", iface.gripper_open_position)
                    current_state = np.array(arm_state + [gripper_obs], dtype=np.float32)
                else:
                    current_state = np.array(arm_state, dtype=np.float32)

                # Action: arm from IK, gripper from button toggle
                arm_targets = user_input.get_action(arm_state)
                if gripper_joint_name:
                    gripper_target = user_input.gripper_position
                    action_vec = np.array(arm_targets + [gripper_target], dtype=np.float32)
                else:
                    action_vec = np.array(arm_targets, dtype=np.float32)

                # Send arm command
                robot.ros2_interface.send_joint_position_command(
                    arm_targets,
                    unnormalize=False,
                    time_from_start_sec=user_input.time_from_start_sec,
                )
                # Send gripper command separately (its own controller)
                if gripper_joint_name:
                    robot.ros2_interface.send_gripper_command(
                        action_vec[-1], unnormalize=False
                    )

                if frame_count % 10 == 0:
                    gripper_str = f" | gripper: {action_vec[-1]:.3f}" if gripper_joint_name else ""
                    logger.info(
                        f"  Frame {frame_count:3d} | "
                        f"state: {[f'{v:.3f}' for v in current_state[:3]]} | "
                        f"action: {[f'{v:.3f}' for v in action_vec[:3]]}"
                        f"{gripper_str}"
                    )

                episode_frames.append({
                    "observation.state": current_state,
                    "observation.environment_state": target_vec,
                    "action": action_vec,
                    "task": "Reach target",
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
        logger.info(f"Done. {saved_episodes} episodes saved to {root_dir}")


if __name__ == "__main__":
    main()
