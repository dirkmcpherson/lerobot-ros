"""Shared pieces for the Gen3 sweep task in Gazebo (gen3-kinesthetic-teaching-ros2).

The sim side is that repo's ``gen3_sim_sweep_task.launch.py`` (``native/sim_up.sh``
starts it headless); the scripted demonstrator / episode reset is its
``docker/sweep_demo.py``, which needs the system python + ROS, so it is always
run as a subprocess through that repo's ``native/env.sh``.

The teaching repo is found as the parent of this checkout (this repo is its
``lerobot-ros`` submodule), else ``$GEN3_TEACHING_REPO``, else
``~/workspace/gen3-kinesthetic-teaching-ros2``.

Features (fixed by the recorder, read back by eval):
  observation.state              (7)  arm joint positions, raw controller values
  observation.environment_state  (45) xyz of cube_1..cube_15, world frame (sim oracle)
  action                         (7)  joint_trajectory_controller reference one frame ahead
"""

import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int32
from std_srvs.srv import Trigger

def _teaching_repo() -> Path:
    if "GEN3_TEACHING_REPO" in os.environ:
        return Path(os.environ["GEN3_TEACHING_REPO"])
    parent = Path(__file__).resolve().parent.parent
    if (parent / "docker/sweep_demo.py").exists():
        return parent
    return Path.home() / "workspace/gen3-kinesthetic-teaching-ros2"


TEACHING_REPO = _teaching_repo()
SWEEP_DEMO = TEACHING_REPO / "docker/sweep_demo.py"
HOST_ENV = TEACHING_REPO / "native/env.sh"

FPS = 10
NUM_CUBES = 15
TASK = "Sweep the cubes off the table"
ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]


def dataset_features() -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": ARM_JOINTS},
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (3 * NUM_CUBES,),
            "names": [f"cube_{i}_{a}" for i in range(1, NUM_CUBES + 1) for a in "xyz"],
        },
        "action": {"dtype": "float32", "shape": (7,), "names": ARM_JOINTS},
    }


def sweep_demo(*args: str, **popen_kwargs) -> subprocess.Popen:
    """Start docker/sweep_demo.py under the host ROS env (system python, conda off)."""
    quoted = " ".join(str(a) for a in args)
    cmd = f". {HOST_ENV} && exec python3 -u {SWEEP_DEMO} {quoted}"
    return subprocess.Popen(["bash", "-c", cmd], text=True, **popen_kwargs)


class SweepSimTracker:
    """Cube poses, controller reference, cube count and the task status service.

    Attaches to an existing node (the BackendRobot's ``ros2_interface.robot_node``,
    which is already spun by its executor thread).
    """

    def __init__(self, node):
        self._node = node
        self._lock = threading.Lock()
        self.cubes: np.ndarray | None = None
        self.reference: dict[str, float] | None = None
        self.cubes_on_table: int | None = None
        node.create_subscription(PoseArray, "/sweep_task_monitor/cube_poses", self._on_cubes, 10)
        node.create_subscription(
            JointTrajectoryControllerState,
            "/joint_trajectory_controller/controller_state",
            self._on_controller,
            10,
        )
        node.create_subscription(Int32, "/sweep_task_monitor/cubes_on_table", self._on_count, 10)
        self._status = node.create_client(Trigger, "/sweep_task_monitor/status")

    def _on_cubes(self, msg: PoseArray) -> None:
        arr = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses], dtype=np.float32)
        with self._lock:
            self.cubes = arr.reshape(-1)

    def _on_controller(self, msg: JointTrajectoryControllerState) -> None:
        if msg.reference.positions:
            with self._lock:
                self.reference = dict(zip(msg.joint_names, msg.reference.positions))

    def _on_count(self, msg: Int32) -> None:
        self.cubes_on_table = msg.data

    def wait_ready(self, timeout: float = 20.0) -> None:
        t0 = time.monotonic()
        while self.cubes is None or self.reference is None:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    "No /sweep_task_monitor/cube_poses or controller_state: is the sweep sim up?"
                )
            time.sleep(0.1)

    def cube_vector(self) -> np.ndarray:
        with self._lock:
            return self.cubes.copy()

    def reference_vector(self) -> np.ndarray:
        with self._lock:
            return np.array([self.reference[j] for j in ARM_JOINTS], dtype=np.float32)

    def status(self, timeout: float = 10.0) -> tuple[bool, str]:
        if not self._status.wait_for_service(timeout_sec=5.0):
            return False, "status service unavailable"
        done = threading.Event()
        future = self._status.call_async(Trigger.Request())
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            return False, "status call timed out"
        res = future.result()
        return res.success, res.message


def arm_state(obs: dict) -> np.ndarray:
    return np.array([obs[f"{j}.pos"] for j in ARM_JOINTS], dtype=np.float32)
