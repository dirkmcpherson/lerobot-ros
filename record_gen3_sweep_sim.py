"""Record scripted sweep demonstrations from the Gazebo sweep task into a LeRobotDataset.

Needs the sim up first (gen3-kinesthetic-teaching-ros2):
    ros2 launch gen3_kinesthetic_teaching_sim gen3_sim_sweep_task.launch.py headless:=true

Runs that repo's docker/sweep_demo.py --phase collect as the demonstrator and samples
at 10 Hz between its EPISODE_START / EPISODE_END markers. Episodes that do not end
with every cube off the table are discarded.

    python record_gen3_sweep_sim.py --episodes 60
"""

import argparse
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

from gen3_sweep_sim import FPS, TASK, SweepSimTracker, arm_state, dataset_features, sweep_demo
from lerobot_backends import BackendRobot
from lerobot_backends.ros2.config import KinovaGen3SweepSimConfig

init_logging()
logger = logging.getLogger(__name__)


class Sampler:
    """Samples (state, cubes, controller reference) at FPS on its own thread."""

    def __init__(self, robot, tracker):
        self._robot, self._tracker = robot, tracker
        self._stop = threading.Event()
        self._thread = None
        self.rows = []

    def start(self):
        self.rows, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self.rows

    def _run(self):
        period = 1.0 / FPS
        next_t = time.perf_counter()
        while not self._stop.is_set():
            obs = self._robot.get_observation()
            self.rows.append(
                (arm_state(obs), self._tracker.cube_vector(), self._tracker.reference_vector())
            )
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo-id", default="lerobot/gen3_sweep_sim")
    ap.add_argument("--root", type=Path, default=Path("data/lerobot/gen3_sweep_sim"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.root)

    robot = BackendRobot(KinovaGen3SweepSimConfig())
    robot.connect()
    tracker = SweepSimTracker(robot.ros2_interface.robot_node)
    tracker.wait_ready()

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        root=args.root,
        robot_type="kinova_gen3",
        features=dataset_features(),
        use_videos=False,
    )
    sampler = Sampler(robot, tracker)
    kept = dropped = 0

    demo = sweep_demo(
        "--phase", "collect", "--episodes", args.episodes, "--seed", args.seed, "--handshake",
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    try:
        for line in demo.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"[demo] {line}")
            if line.startswith("EPISODE_START"):
                sampler.start()
            elif line.startswith("EPISODE_END"):
                rows = sampler.stop()
                success, message = tracker.status()
                if success and len(rows) > 1:
                    refs = np.stack([r[2] for r in rows])
                    # action[t] = where the controller is told to be one frame later
                    actions = np.concatenate([refs[1:], refs[-1:]])
                    for (state, cubes, _ref), action in zip(rows, actions):
                        dataset.add_frame({
                            "observation.state": state,
                            "observation.environment_state": cubes,
                            "action": action,
                            "task": TASK,
                        })
                    dataset.save_episode()
                    kept += 1
                    logger.info(f"kept episode ({len(rows)} frames); {kept} kept, {dropped} dropped")
                else:
                    dropped += 1
                    logger.warning(f"dropped episode: {message}; {kept} kept, {dropped} dropped")
                demo.stdin.write("\n")
                demo.stdin.flush()
        demo.wait()
    except KeyboardInterrupt:
        logger.info("interrupted")
        demo.terminate()
    finally:
        dataset.finalize()
        robot.disconnect()
        logger.info(f"{kept} episodes saved to {args.root} ({dropped} dropped)")


if __name__ == "__main__":
    main()
