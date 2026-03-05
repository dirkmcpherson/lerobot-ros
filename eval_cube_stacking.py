"""Evaluate a trained DiffusionPolicy on the cube stacking task.

Reads live cube poses from Gazebo (via ROS bridge) and feeds them as
observation.environment_state alongside the robot's joint state.

Usage:
    python eval_cube_stacking.py [--checkpoint PATH] [--duration SEC]
"""

import argparse
import logging
import time

import torch
from pathlib import Path

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.utils import init_logging

from lerobot_backends import BackendRobot
from lerobot_backends.ros2.config import KinovaGen3LiteConfig

init_logging()
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("outputs/train/cube_stacking/checkpoints/last/pretrained_model")
FPS = 10

# Same box pose topics as recording script
BOX_POSE_TOPICS = {
    "task_box_green": "/model/task_box_green/pose",
    "task_box_red": "/model/task_box_red/pose",
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

    def _pose_callback(self, name: str, msg) -> None:
        self._poses[name] = [msg.position.x, msg.position.y, msg.position.z]

    def get_positions(self) -> list[float]:
        """Return flat list: [green_x, green_y, green_z, red_x, red_y, red_z]."""
        values = []
        for name in ["task_box_green", "task_box_red"]:
            values.extend(self._poses[name])
        return values

    def destroy(self):
        for sub in self._subs:
            sub.destroy()
        self._subs.clear()


def main():
    parser = argparse.ArgumentParser(description="Evaluate cube stacking policy.")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Path to pretrained_model directory"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Evaluation duration in seconds (default: 30)"
    )
    args = parser.parse_args()

    # Load policy + processors
    logger.info(f"Loading policy from {args.checkpoint}")
    try:
        policy = DiffusionPolicy.from_pretrained(args.checkpoint)
    except Exception as e:
        logger.error(f"Failed to load policy: {e}")
        return

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint, "policy_preprocessor.json"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint, "policy_postprocessor.json"
    )
    policy.eval()

    # Connect robot
    logger.info("Connecting to Gen3 Lite...")
    config = KinovaGen3LiteConfig()
    robot = BackendRobot(config=config)
    robot.connect()
    time.sleep(2.0)

    iface = config.ros2_interface
    arm_joint_names = iface.arm_joint_names
    gripper_joint_name = iface.gripper_joint_name
    all_joint_names = arm_joint_names + ([gripper_joint_name] if gripper_joint_name else [])

    # Subscribe to box poses
    box_tracker = BoxPoseTracker(robot.ros2_interface.robot_node, BOX_POSE_TOPICS)
    time.sleep(1.0)

    try:
        # Reset to home
        robot.backend.reset()
        policy.reset()

        logger.info(f"Running policy for {args.duration}s...")
        start_time = time.time()

        while time.time() - start_time < args.duration:
            step_start = time.perf_counter()

            # Read robot state
            raw_obs = robot.get_observation()
            state_values = [raw_obs.get(f"{j}.pos", 0.0) for j in all_joint_names]
            state_tensor = torch.tensor(state_values, dtype=torch.float32)

            # Read live cube positions
            env_state = box_tracker.get_positions()
            env_tensor = torch.tensor(env_state, dtype=torch.float32)

            # Build observation dict (unbatched — preprocessor adds batch dim)
            obs_dict = {
                "observation.state": state_tensor,
                "observation.environment_state": env_tensor,
            }
            obs_normalized = preprocessor(obs_dict)

            with torch.no_grad():
                action_normalized = policy.select_action(obs_normalized)

            action_unnorm = postprocessor({"action": action_normalized})
            action_np = action_unnorm["action"].numpy().squeeze(0)

            logger.info(
                f"Action: {[f'{v:.3f}' for v in action_np[:3]]}... | "
                f"Boxes: G[{env_state[0]:.2f},{env_state[1]:.2f},{env_state[2]:.2f}] "
                f"R[{env_state[3]:.2f},{env_state[4]:.2f},{env_state[5]:.2f}]"
            )

            # Send action — arm joints + gripper
            action_dict = {}
            for i, name in enumerate(arm_joint_names):
                action_dict[f"{name}.pos"] = float(action_np[i])
            if gripper_joint_name and len(action_np) > len(arm_joint_names):
                action_dict["gripper.pos"] = float(action_np[len(arm_joint_names)])
            robot.send_action(action_dict)

            dt = time.perf_counter() - step_start
            sleep_time = 1.0 / FPS - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted.")
    finally:
        box_tracker.destroy()
        robot.disconnect()
        logger.info("Done.")


if __name__ == "__main__":
    main()
