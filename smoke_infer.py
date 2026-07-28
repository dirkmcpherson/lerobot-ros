#!/usr/bin/env python3
"""Load the trained checkpoint and run one inference step — CPU, no ROS, no arm.

Final third of the pure-software pipeline smoke test. This is the deployment/eval
round-trip that `eval_kinova_reach.py` performs on the real robot, but with a
synthetic observation instead of a live `get_observation()` so it needs no hardware.

It exercises exactly the version-fragile surface the real eval depends on:

    DiffusionPolicy.from_pretrained(ckpt)
    PolicyProcessorPipeline.from_pretrained(ckpt, "policy_preprocessor.json")   # normalize obs
    policy.select_action(obs)                                                    # DP inference
    PolicyProcessorPipeline.from_pretrained(ckpt, "policy_postprocessor.json")  # unnormalize action

Success = a finite 7-dim action comes out for a gen3_lite-shaped observation.
"""
import logging
from pathlib import Path

import lerobot_py314_compat  # noqa: F401  (patches draccus for Py3.14 before lerobot config load)
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.utils import init_logging

init_logging()
logger = logging.getLogger(__name__)

CKPT = Path("outputs/train/gen3_lite_smoke/checkpoints/last/pretrained_model")
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]

# A synthetic gen3_lite observation: at home, with an arbitrary target pose.
HOME = [0.0, -0.52, 1.22, -1.22, -0.52, 0.0, 0.0]
TARGET = [0.5, 0.2, 0.9, -0.9, 0.1, 0.4, 0.42]


def main():
    logger.info(f"Loading policy from {CKPT}")
    policy = DiffusionPolicy.from_pretrained(CKPT)
    preprocessor = PolicyProcessorPipeline.from_pretrained(CKPT, "policy_preprocessor.json")
    postprocessor = PolicyProcessorPipeline.from_pretrained(CKPT, "policy_postprocessor.json")
    policy.eval()
    policy.reset()

    obs = {
        "observation.state": torch.tensor(HOME, dtype=torch.float32),
        "observation.environment_state": torch.tensor(TARGET, dtype=torch.float32),
    }

    obs_n = preprocessor(obs)
    with torch.no_grad():
        action_n = policy.select_action(obs_n)
    action = postprocessor({"action": action_n})["action"].numpy().squeeze(0)

    logger.info("Inference round-trip OK.")
    logger.info(f"  action ({len(action)}-dim): {[f'{v:.3f}' for v in action]}")
    assert action.shape == (len(JOINT_NAMES),), f"unexpected action shape {action.shape}"
    assert bool(torch.isfinite(torch.tensor(action)).all()), "non-finite action"
    logger.info("  shape + finiteness checks passed.")


if __name__ == "__main__":
    main()
