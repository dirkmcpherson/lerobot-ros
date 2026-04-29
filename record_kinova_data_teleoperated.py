"""Teleoperated data recording for robot manipulation tasks.

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
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

import cv2
import lerobot.cameras.opencv.camera_opencv as _lerobot_opencv_cam
from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

# Force V4L2 backend on Linux — the default CAP_ANY picks FFMPEG, which can't
# negotiate MJPG/resolution properly and triggers VIDIOC_QBUF failures.
_lerobot_opencv_cam.get_cv2_backend = lambda: int(cv2.CAP_V4L2)

from lerobot_backends import BackendRobot
from lerobot_backends.ros2.config import KinovaGen3Config, KinovaGen3LiteConfig, ROS2BackendConfig
from user_input import KeyboardUserInput, SpacemouseUserInput, UserInput

init_logging()
logger = logging.getLogger(__name__)

# --- Dataset config ----------------------------------------------------------
DEFAULT_FPS = 5
MAX_EPISODE_SECONDS = 60  # safety cap

# Camera presets — wrist + exterior, MJPG @ 320x240 @ 30fps
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

# Box pose ROS topics (bridged from Gazebo)
BOX_POSE_TOPICS = {
    "task_box_green": "/model/task_box_green/pose",
    "task_box_red": "/model/task_box_red/pose",
}

# Collision checking paths
_KORTEX_PKG_DIR = "/home/james/workspace/ros2_kortex_ws/src/ros2_kortex"
_SRDF_PATHS = {
    "gen3": (
        f"{_KORTEX_PKG_DIR}/kortex_moveit_config/"
        "kinova_gen3_7dof_robotiq_2f_85_moveit_config/config/gen3.srdf"
    ),
    "gen3_lite": (
        f"{_KORTEX_PKG_DIR}/kortex_moveit_config/"
        "kinova_gen3_lite_moveit_config/config/gen3_lite.srdf"
    ),
}

# Per-robot configs: (config_class, dataset_repo_id, root_dir, robot_type)
_ROBOT_CONFIGS: dict[str, tuple] = {
    "gen3": (
        KinovaGen3Config,
        "lerobot/kinova_gen3_teleop",
        Path("data/lerobot/kinova_gen3_teleop"),
        "kinova_gen3",
    ),
    "gen3_lite": (
        KinovaGen3LiteConfig,
        "lerobot/kinova_gen3_lite_teleop",
        Path("data/lerobot/kinova_gen3_lite_teleop"),
        "kinova_gen3_lite",
    ),
}


class BoxPoseTracker:
    """Subscribes to Gazebo-bridged box pose topics and caches latest positions."""

    def __init__(self, node, topics: dict[str, str]):
        from geometry_msgs.msg import Pose

        self._poses: dict[str, list[float]] = {
            name: [0.0, 0.0, 0.0] for name in topics
        }
        self._subs = []
        for name, topic in topics.items():
            sub = node.create_subscription(
                Pose, topic,
                lambda msg, n=name: self._pose_callback(n, msg),
                10,
            )
            self._subs.append(sub)
            logger.info(f"Subscribed to box pose: {topic}")

    def _pose_callback(self, name: str, msg) -> None:
        self._poses[name] = [msg.position.x, msg.position.y, msg.position.z]

    def get_positions(self) -> np.ndarray:
        """Return flat array: [green_x, green_y, green_z, red_x, red_y, red_z]."""
        values = []
        for name in ["task_box_green", "task_box_red"]:
            values.extend(self._poses[name])
        return np.array(values, dtype=np.float32)

    def destroy(self):
        for sub in self._subs:
            sub.destroy()
        self._subs.clear()


def make_input_device(
    name: str,
    config: ROS2BackendConfig,
    urdf_path: Optional[str] = None,
    robot_key: Optional[str] = None,
    ground_plane_z: Optional[float] = 0.0,
) -> UserInput:
    """Create a UserInput device from the backend config."""
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
        srdf_path = _SRDF_PATHS.get(robot_key) if robot_key else None
        return SpacemouseUserInput(
            joint_names=iface.arm_joint_names,
            min_joint_positions=iface.min_joint_positions,
            max_joint_positions=iface.max_joint_positions,
            urdf_path=urdf_path,
            ee_link=iface.ee_link,
            linear_scale=0.15,
            angular_scale=0.4,
            gripper_joint_name=iface.gripper_joint_name,
            gripper_open_position=iface.gripper_open_position,
            gripper_closed_position=iface.gripper_close_position,
            collision_srdf_path=srdf_path,
            collision_package_dirs=[_KORTEX_PKG_DIR],
            collision_ground_plane_z=ground_plane_z,
        )
    raise ValueError(f"Unknown input device '{name}'. Available: keyboard, spacemouse")


def randomize_boxes() -> None:
    """Teleport boxes to random coordinates using Gazebo Transport."""
    gx, gy = random.uniform(0.3, 0.6), random.uniform(-0.2, 0.2)
    rx, ry = random.uniform(0.3, 0.6), random.uniform(-0.2, 0.2)

    while np.hypot(gx - rx, gy - ry) < 0.1:
        rx, ry = random.uniform(0.3, 0.6), random.uniform(-0.2, 0.2)

    logger.info(f"Teleporting boxes: Green({gx:.3f}, {gy:.3f}), Red({rx:.3f}, {ry:.3f})")

    subprocess.run([
        "gz", "service", "-s", "/world/empty/set_pose",
        "--reqtype", "gz.msgs.Entity",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "1000",
        "--req", f'name: "task_box_green", pose: {{position: {{x: {gx}, y: {gy}, z: 0.05}}}}'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([
        "gz", "service", "-s", "/world/empty/set_pose",
        "--reqtype", "gz.msgs.Entity",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "1000",
        "--req", f'name: "task_box_red", pose: {{position: {{x: {rx}, y: {ry}, z: 0.05}}}}'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description="Teleoperated data recording.")
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
    parser.add_argument(
        "--no-boxes", action="store_true",
        help="Skip box randomization and pose tracking (for arm-only teleop)"
    )
    parser.add_argument(
        "--use-sim-time", action="store_true",
        help="Set use_sim_time on the ROS2 node (required for Gazebo sim)"
    )
    parser.add_argument(
        "--no-collision", action="store_true",
        help="Disable collision checking (self-collision + ground plane)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Dry-run: log commanded joint positions and deltas but never send to robot"
    )
    parser.add_argument(
        "--cameras", action="store_true",
        help="Record wrist + exterior camera frames as MP4 video alongside joints"
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS,
        help=f"Recording FPS (default: {DEFAULT_FPS}; use 30 for vision policies)"
    )
    parser.add_argument(
        "--task", type=str, default="Teleop",
        help="Task description string saved with each frame (default: 'Teleop')"
    )
    parser.add_argument(
        "--dataset-name", type=str, default=None,
        help="Override dataset directory name (default: per-robot preset)"
    )

    args = parser.parse_args()
    fps = args.fps
    max_episode_frames = MAX_EPISODE_SECONDS * fps

    config_cls, dataset_repo_id, root_dir, robot_type = _ROBOT_CONFIGS[args.robot]
    config = config_cls()
    if args.use_sim_time:
        config.use_sim_time = True
    if args.cameras:
        config.cameras = dict(CAMERA_CONFIGS)
    if args.dataset_name:
        dataset_repo_id = f"lerobot/{args.dataset_name}"
        root_dir = Path(f"data/lerobot/{args.dataset_name}")
    iface = config.ros2_interface
    arm_joint_names = iface.arm_joint_names
    gripper_joint_name: Optional[str] = iface.gripper_joint_name or None

    # Full state/action names include gripper when present
    all_joint_names = arm_joint_names + ([gripper_joint_name] if gripper_joint_name else [])
    state_dim = len(all_joint_names)

    if args.overwrite and root_dir.exists():
        logger.info(f"--overwrite: removing existing dataset at {root_dir}")
        shutil.rmtree(root_dir)

    # Build input device
    ground_z = None if args.no_collision else -0.04
    user_input = make_input_device(
        args.input, config,
        urdf_path=args.urdf,
        robot_key=args.robot,
        ground_plane_z=ground_z,
    )

    # Connect robot via backend
    logger.info(f"Connecting to {args.robot}...")
    robot = BackendRobot(config)
    robot.connect()
    time.sleep(2.0)

    # Subscribe to live box pose topics via the robot's ROS node
    use_boxes = not args.no_boxes
    box_tracker = None
    if use_boxes:
        box_tracker = BoxPoseTracker(robot.ros2_interface.robot_node, BOX_POSE_TOPICS)
        time.sleep(1.0)  # let first pose messages arrive

    # Connect input device
    user_input.connect()

    # Publish collision objects to RViz (if collision checking is active)
    if hasattr(user_input, '_collision_checker') and user_input._collision_checker is not None:
        user_input._collision_checker.publish_to_rviz(robot.ros2_interface.robot_node)

    dataset_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": all_joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": all_joint_names,
        },
    }
    if use_boxes:
        dataset_features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (6,),
            "names": ["green_x", "green_y", "green_z", "red_x", "red_y", "red_z"],
        }
    if args.cameras:
        for cam_key, cam_cfg in CAMERA_CONFIGS.items():
            dataset_features[cam_key] = {
                "dtype": "video",
                "shape": (cam_cfg.height, cam_cfg.width, 3),
                "names": ["height", "width", "channels"],
            }

    dataset = LeRobotDataset.create(
        repo_id=dataset_repo_id,
        fps=fps,
        root=root_dir,
        robot_type=robot_type,
        features=dataset_features,
        use_videos=bool(args.cameras),
    )

    saved_episodes = 0

    logger.info(
        f"Starting teleoperated recording on {args.robot}. "
        f"Gripper: {'yes (' + gripper_joint_name + ')' if gripper_joint_name else 'no'}. "
        f"Goal: {args.num_episodes} saved episodes."
    )

    try:
        while saved_episodes < args.num_episodes and not user_input.quit_requested:
            logger.info(f"\n=== Episode {saved_episodes + 1}/{args.num_episodes} ===")

            # 1. Randomize boxes (if enabled)
            if use_boxes:
                randomize_boxes()
                time.sleep(0.5)  # let physics settle + pose messages update

            # 2. Reset robot (backend handles home position + settle automatically)
            if args.debug:
                obs = robot.backend.get_observation()
            else:
                obs = robot.backend.reset()
            actual_home = [obs[f"{j}.pos"] for j in arm_joint_names]
            user_input.reset(actual_home)

            episode_frames: list[dict] = []
            frame_count = 0

            if gripper_joint_name:
                logger.info(
                    "Recording. SpaceMouse: move EE | Left=save | Right/Both=gripper toggle | Ctrl+C=quit"
                )
            else:
                logger.info("Recording. Use input device to move robot. Enter=save, D=discard.")

            while not user_input.episode_end_requested:
                if frame_count >= max_episode_frames:
                    logger.warning(
                        f"Reached max frames ({max_episode_frames}). Auto-saving episode."
                    )
                    break

                step_start = time.perf_counter()

                # Observation — arm joints (+ gripper if configured)
                obs = robot.get_observation()
                arm_state = [obs[f"{j}.pos"] for j in arm_joint_names]
                if gripper_joint_name:
                    gripper_obs = obs.get(f"{gripper_joint_name}.pos", 0.0)
                    current_state = np.array(arm_state + [gripper_obs], dtype=np.float32)
                else:
                    current_state = np.array(arm_state, dtype=np.float32)

                # Live cube positions from Gazebo (if enabled)
                env_state = box_tracker.get_positions() if use_boxes else None

                # Action: arm from input device, gripper from button toggle
                arm_targets = user_input.get_action(arm_state)
                if gripper_joint_name:
                    gripper_target = user_input.gripper_position
                    action_vec = np.array(arm_targets + [gripper_target], dtype=np.float32)
                else:
                    action_vec = np.array(arm_targets, dtype=np.float32)

                # Send action through the backend
                action_dict = {f"{j}.pos": float(action_vec[i]) for i, j in enumerate(arm_joint_names)}
                if gripper_joint_name:
                    action_dict["gripper.pos"] = float(action_vec[-1])

                delta = action_vec - current_state
                if args.debug:
                    logger.info(
                        f"  Frame {frame_count:3d} | "
                        f"cmd: [{', '.join(f'{v:.4f}' for v in action_vec)}] | "
                        f"delta: [{', '.join(f'{v:+.4f}' for v in delta)}]"
                    )
                else:
                    robot.backend.send_action(action_dict)

                if not args.debug and frame_count % 10 == 0:
                    gripper_str = f" | gripper: {action_vec[-1]:.3f}" if gripper_joint_name else ""
                    boxes_str = (
                        f" | boxes: [{env_state[0]:.2f},{env_state[1]:.2f},{env_state[2]:.2f}]"
                        f"[{env_state[3]:.2f},{env_state[4]:.2f},{env_state[5]:.2f}]"
                        if env_state is not None else ""
                    )
                    logger.info(
                        f"  Frame {frame_count:3d} | "
                        f"state: {[f'{v:.3f}' for v in current_state[:3]]}"
                        f"{boxes_str}{gripper_str}"
                    )

                frame = {
                    "observation.state": current_state,
                    "action": action_vec,
                    "task": args.task,
                }
                if env_state is not None:
                    frame["observation.environment_state"] = env_state
                if args.cameras:
                    for cam_key in CAMERA_CONFIGS:
                        img = obs.get(cam_key)
                        if img is None:
                            logger.warning(f"Missing camera frame for {cam_key} at frame {frame_count}")
                            continue
                        frame[cam_key] = img
                episode_frames.append(frame)
                frame_count += 1

                dt = time.perf_counter() - step_start
                sleep_time = 1.0 / fps - dt
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
        if box_tracker is not None:
            box_tracker.destroy()
        user_input.disconnect()
        robot.disconnect()
        logger.info(f"Done. {saved_episodes} episodes saved to {root_dir}")


if __name__ == "__main__":
    main()
