"""Run Gen3LiteHil on the real ROS2Backend against fake_driver.py. Usage: check.py <urdf>
(start fake_driver.py first; see README.md)."""

import logging
import sys
import time

import numpy as np

from lerobot_backends.ros2.backend import ROS2Backend
from lerobot_robot_ros import Gen3LiteHil, Gen3LiteHilConfig, StaleObservationError

PERIOD = 0.1


def tool(obs) -> np.ndarray:
    return np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]])


def main():
    logging.basicConfig(level=logging.INFO)
    cfg = Gen3LiteHilConfig(guard_enabled=False, home_on_connect=False, urdf_path=sys.argv[1], use_gripper=True)
    robot = Gen3LiteHil(cfg)
    assert isinstance(robot.backend, ROS2Backend)
    robot.connect()
    obs = robot.get_observation()
    start = tool(obs)
    print("obs keys", list(obs), "tool", start.round(4), "ages", robot.observation_ages())

    step_times = []
    for i in range(20):  # 20 cm asked for in +x; the tool box ends 13 cm away
        t0 = time.perf_counter()
        robot.send_action({"delta_x": 1.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 0 if i > 10 else 1})
        obs = robot.get_observation()
        step_times.append(time.perf_counter() - t0)
        time.sleep(max(PERIOD - (time.perf_counter() - t0), 0))
    moved = tool(obs) - start
    print(
        "moved", moved.round(4), "gripper", round(obs["gripper.pos"], 3),
        "step path ms mean/max", round(np.mean(step_times) * 1e3, 2), round(np.max(step_times) * 1e3, 2),
    )
    assert abs(moved[0] - 0.13) < 0.002, "should stop at the tool box face x = 0.43"
    assert abs(obs["gripper.pos"] - 0.85) < 1e-3, "gripper should have closed"

    time.sleep(0.6)
    print("watchdog trips", robot.watchdog_trips)
    assert robot.watchdog_trips == 1

    print("waiting for the fake driver to stop publishing...")
    deadline = time.time() + 30
    while True:
        if time.time() > deadline:
            raise SystemExit("joint state never went stale")
        try:
            robot.get_observation()
            time.sleep(0.2)
        except StaleObservationError as e:
            print("STALE detected:", e)
            break
    try:
        robot.send_action({"delta_x": 1.0, "delta_y": 0.0, "delta_z": 0.0})
    except StaleObservationError:
        print("command refused on stale joint state")
    else:
        raise SystemExit("a command was sent on stale joint state")

    robot.disconnect()
    robot.disconnect()
    print("ROS CHECK PASSED")


if __name__ == "__main__":
    main()
