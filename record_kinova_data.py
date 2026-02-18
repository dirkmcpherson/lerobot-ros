
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

# ROS wrapper imports
from lerobot_robot_ros.config import ROS2Config, ROS2InterfaceConfig, ActionType, GripperActionType
from lerobot_robot_ros.robot import ROS2Robot

# Configure logging
init_logging()
logger = logging.getLogger(__name__)

# --- Task Definition ---
# Home: MoveIt named "home" for Gen3 7DOF
HOME_POSITION = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]

# Target: arm nearly fully extended, pointing down and to the left.
# joint_1 rotated ~90° left, shoulder/elbow configured to point arm downward.
# NOTE: Verify these angles match the desired pose on the physical robot before
# running full data collection. Adjust as needed.
TARGET_POSITION = [1.57, 2.0, 3.14, -0.5, 0.0, 0.0, 1.57]

# --- Dataset Config ---
DATASET_REPO_ID = "lerobot/kinova_gen3_reach"
ROOT_DIR = Path("data/lerobot/kinova_gen3_reach")
FPS = 10
NUM_EPISODES = 50
EPISODE_LENGTH_FRAMES = 100  # 10 seconds at 10 FPS
HOME_SETTLE_SEC = 6.0        # Time to wait after commanding home position
ROBOT_TYPE = "kinova_gen3"


@dataclass
class KinovaGen3Config(ROS2Config):
    action_type: ActionType = ActionType.JOINT_TRAJECTORY

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            arm_joint_names=[
                "joint_1", "joint_2", "joint_3", "joint_4",
                "joint_5", "joint_6", "joint_7"
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


def reset_to_home(robot: ROS2Robot, joint_names: list[str]) -> None:
    """Command the robot to move to the home position and wait for it to settle."""
    logger.info(f"Resetting to home: {HOME_POSITION}")
    robot.ros2_interface.send_joint_position_command(
        HOME_POSITION, unnormalize=False, time_from_start_sec=5.0
    )
    time.sleep(HOME_SETTLE_SEC)
    obs = robot.get_observation()
    current = [obs[f"{j}.pos"] for j in joint_names]
    logger.info(f"At home (actual): {[f'{v:.3f}' for v in current]}")


def main():
    logger.info("Initializing Robot...")
    config = KinovaGen3Config()
    robot = ROS2Robot(config)
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
        # Kept for future goal-conditioned policy training.
        # For this fixed task the target is always TARGET_POSITION.
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
            reset_to_home(robot, joint_names)

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

                # Send at 10 FPS cadence (0.1 s per step → controller reaches each
                # waypoint before the next one arrives)
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
