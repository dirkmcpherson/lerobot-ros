"""Evaluate a trained reach policy in ManiSkill.

Loads a DiffusionPolicy checkpoint, runs it in the ManiSkill Panda env,
and reports how close the arm gets to the target pose.

Usage:
    python eval_maniskill.py
    python eval_maniskill.py --checkpoint outputs/train/maniskill_reach/checkpoints/last/pretrained_model
    python eval_maniskill.py --render --num-episodes 5
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.utils import init_logging

from lerobot_backends import BackendRobot
from lerobot_backends.maniskill.config import ManiSkillPickCubeConfig

init_logging()
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("outputs/train/maniskill_reach/checkpoints/last/pretrained_model")

# Must match record_maniskill_data.py
TARGET_POSITION = np.array([0.5, 0.8, 0.3, -1.2, 0.2, 1.8, 0.4], dtype=np.float32)
NUM_JOINTS = 7
EPISODE_LENGTH = 100
FPS = 20


def main():
    parser = argparse.ArgumentParser(description="Evaluate ManiSkill reach policy.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="Show cv2 render window")
    args = parser.parse_args()

    # Load policy + processors
    logger.info(f"Loading policy from {args.checkpoint}")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint, "policy_preprocessor.json"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        args.checkpoint, "policy_postprocessor.json"
    )
    policy.eval()

    render_mode = "rgb_array" if args.render else None
    config = ManiSkillPickCubeConfig(obs_mode="state", render_mode=render_mode)
    robot = BackendRobot(config)
    robot.connect()

    cv2_window = None
    if args.render:
        import cv2
        cv2_window = "ManiSkill Eval"
        cv2.namedWindow(cv2_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cv2_window, 640, 480)

    errors = []

    try:
        for ep in range(args.num_episodes):
            obs = robot.backend.reset()
            policy.reset()

            initial_pos = None

            target = TARGET_POSITION.copy()
            target_tensor = torch.tensor(target, dtype=torch.float32)

            for step in range(EPISODE_LENGTH):
                # Render
                if cv2_window is not None:
                    import cv2
                    frame = robot.backend._env.render()
                    if hasattr(frame, "cpu"):
                        frame = frame.cpu().numpy()
                    frame = np.asarray(frame).squeeze()
                    if frame.ndim == 3:
                        cv2.imshow(cv2_window, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

                state = [obs[f"joint_{i+1}.pos"] for i in range(NUM_JOINTS)]
                if initial_pos is None: initial_pos = state

                state_tensor = torch.tensor(state, dtype=torch.float32)

                obs_dict = {
                    "observation.state": state_tensor,
                    "observation.environment_state": target_tensor,
                }
                obs_normalized = preprocessor(obs_dict)

                with torch.no_grad():
                    action_normalized = policy.select_action(obs_normalized)

                action_unnorm = postprocessor({"action": action_normalized})
                action_np = action_unnorm["action"].numpy().squeeze(0)

                action_dict = {f"joint_{j+1}.pos": float(action_np[j]) for j in range(NUM_JOINTS)}
                if config.gripper_joint_name:
                    action_dict["gripper.pos"] = 1.0
                robot.backend.send_action(action_dict)
                obs = robot.backend.get_observation()

            final = np.array(
                [obs[f"joint_{i+1}.pos"] for i in range(NUM_JOINTS)], dtype=np.float32
            )
            error = np.linalg.norm(final - target)
            errors.append(error)

            logger.info(
                f"Episode {ep + 1}/{args.num_episodes} | "
                f"final_error={error:.4f} | "
                f"initial={[f'{v:.3f}' for v in initial_pos[:3]]}... | "
                f"final={[f'{v:.3f}' for v in final[:3]]}... | "
                f"target={[f'{v:.3f}' for v in target[:3]]}..."
            )

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted.")
    finally:
        if cv2_window is not None:
            import cv2
            cv2.destroyAllWindows()
        robot.disconnect()

    if errors:
        logger.info(
            f"\nResults over {len(errors)} episodes: "
            f"mean_error={np.mean(errors):.4f} | "
            f"std={np.std(errors):.4f} | "
            f"min={np.min(errors):.4f} | "
            f"max={np.max(errors):.4f}"
        )


if __name__ == "__main__":
    main()
