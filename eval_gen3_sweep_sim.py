"""Closed-loop evaluation of a sweep-task policy in Gazebo (gen3-kinesthetic-teaching-ros2).

Per episode: docker/sweep_demo.py --phase start resets the cubes and moves the arm to a
random start pose (seeds disjoint from recording), then the policy runs at 10 Hz for
--duration seconds. Score = cubes swept off the table, out of 15.

    python eval_gen3_sweep_sim.py --episodes 10
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.utils import init_logging

from gen3_sweep_sim import ARM_JOINTS, FPS, NUM_CUBES, SweepSimTracker, arm_state, sweep_demo
from lerobot_backends import BackendRobot
from lerobot_backends.ros2.config import KinovaGen3SweepSimConfig

init_logging()
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path("outputs/train/gen3_sweep_sim/checkpoints/last/pretrained_model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--duration", type=float, default=35.0, help="seconds per rollout")
    ap.add_argument("--seed", type=int, default=1000, help="start-pose seed of the first episode")
    args = ap.parse_args()

    policy = DiffusionPolicy.from_pretrained(args.checkpoint)
    policy.eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(args.checkpoint, "policy_preprocessor.json")
    postprocessor = PolicyProcessorPipeline.from_pretrained(args.checkpoint, "policy_postprocessor.json")

    robot = BackendRobot(KinovaGen3SweepSimConfig())
    robot.connect()
    tracker = SweepSimTracker(robot.ros2_interface.robot_node)
    tracker.wait_ready()

    swept = []
    try:
        for ep in range(args.episodes):
            reset = sweep_demo("--phase", "start", "--seed", args.seed + ep)
            if reset.wait() != 0:
                logger.error(f"episode {ep}: reset failed")
                continue
            time.sleep(1.0)
            policy.reset()
            print(f"ROLLOUT_START {ep} seed={args.seed + ep}", flush=True)
            t0, next_t, slow = time.monotonic(), time.perf_counter(), 0
            while time.monotonic() - t0 < args.duration:
                obs = robot.get_observation()
                batch = preprocessor({
                    "observation.state": torch.from_numpy(arm_state(obs)),
                    "observation.environment_state": torch.from_numpy(tracker.cube_vector()),
                })
                with torch.no_grad():
                    action = postprocessor({"action": policy.select_action(batch)})["action"]
                action = action.squeeze(0).cpu().numpy()
                robot.send_action({f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, action)})
                next_t += 1.0 / FPS
                remaining = next_t - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    slow += 1
            time.sleep(1.5)   # let the last cubes land
            on_table = tracker.cubes_on_table
            swept.append(NUM_CUBES - on_table if on_table is not None else 0)
            logger.info(
                f"episode {ep}: {swept[-1]}/{NUM_CUBES} cubes swept"
                + (f" ({slow} control periods overran)" if slow else "")
            )
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        robot.disconnect()
    if swept:
        full = sum(s == NUM_CUBES for s in swept)
        logger.info(
            f"{len(swept)} episodes: mean {np.mean(swept):.1f}/{NUM_CUBES} cubes swept, "
            f"{full}/{len(swept)} complete sweeps; per episode {swept}"
        )


if __name__ == "__main__":
    main()
