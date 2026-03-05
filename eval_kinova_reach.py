import time
import logging
import torch
import numpy as np
from pathlib import Path

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.utils import init_logging

from lerobot_backends import BackendRobot
from lerobot_backends.ros2.config import KinovaGen3Config

init_logging()
logger = logging.getLogger(__name__)

# Must match record_kinova_data.py exactly
TARGET_POSITION = [1.57, 2.0, 3.14, -0.5, 0.0, 0.0, 1.57]

CHECKPOINT_PATH = Path("outputs/train/kinova_reach/checkpoints/last/pretrained_model")
EVAL_DURATION_SEC = 30.0
FPS = 10


def main():
    logger.info(f"Loading policy from {CHECKPOINT_PATH}")
    try:
        policy = DiffusionPolicy.from_pretrained(CHECKPOINT_PATH)
    except Exception as e:
        logger.error(f"Failed to load policy: {e}")
        return

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        CHECKPOINT_PATH, "policy_preprocessor.json"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        CHECKPOINT_PATH, "policy_postprocessor.json"
    )

    policy.eval()

    logger.info("Initializing Kinova Gen3 Robot...")
    robot = BackendRobot(config=KinovaGen3Config())
    robot.connect()
    time.sleep(2.0)

    joint_names = [f"joint_{i+1}" for i in range(7)]
    target_tensor = torch.tensor(TARGET_POSITION, dtype=torch.float32)

    logger.info(f"Target: {TARGET_POSITION}")

    try:
        # Reset to home via backend (handles home position + settle automatically)
        robot.backend.reset()

        policy.reset()
        logger.info(f"Running policy for {EVAL_DURATION_SEC}s...")

        start_time = time.time()
        while time.time() - start_time < EVAL_DURATION_SEC:
            step_start = time.perf_counter()

            raw_obs = robot.get_observation()

            state_tensor = torch.tensor(
                [raw_obs.get(f"{name}.pos", 0.0) for name in joint_names],
                dtype=torch.float32,
            )
            obs_dict = {
                "observation.state": state_tensor,
                "observation.environment_state": target_tensor,
            }
            obs_normalized = preprocessor(obs_dict)

            with torch.no_grad():
                action_normalized = policy.select_action(obs_normalized)

            action_unnorm = postprocessor({"action": action_normalized})
            action_np = action_unnorm["action"].numpy().squeeze(0)  # (1, 7) -> (7,)

            logger.info(f"Action: {[f'{v:.3f}' for v in action_np]}")

            action_dict = {f"{name}.pos": float(val) for name, val in zip(joint_names, action_np)}
            robot.send_action(action_dict)

            dt = time.perf_counter() - step_start
            sleep_time = 1.0 / FPS - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted.")
    finally:
        robot.disconnect()
        logger.info("Done.")


if __name__ == "__main__":
    main()
