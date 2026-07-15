#!/usr/bin/env python3
"""End-to-end smoke test of THIS repo's lerobot backend against a REAL Kinova Gen3 Lite.

Unlike the CLI/`real_eef_move_z.py` checks (which bypass the project code), this drives
the actual application path:

    BackendRobot(KinovaGen3LiteConfig()) -> connect() / get_observation() / send_action()

It exercises, on real hardware:
  1. connect + feature descriptors + observation round-trip
  2. the gripper through the backend (ParallelGripperCommand path)
  3. a single small, bounded wrist-joint move (joint_6 += 0.10 rad) through the
     JOINT_TRAJECTORY path, then returns it

Safety:
  - joint targets are absolute radians; the backend clamps them to the config's
    min/max joint limits before sending (built-in net)
  - only joint_6 changes; the other five are held at their live current values
  - action_time_from_start_sec is overridden to 2.0 s for a gentle move
  - gripper is NOT commanded fully closed; open target stays within the (now
    loosened) URDF limit

Prereqs: driver up (`gen3_lite.launch.py robot_ip:=...`), arm homed, run inside the
`lerobot-ros-314` conda env with ROS + kortex workspace sourced.
"""
import logging
import time

from lerobot.utils.utils import init_logging

from lerobot_backends import BackendRobot
from lerobot_backends.ros2.config import KinovaGen3LiteConfig

MOVE_JOINT = "joint_6"      # single wrist joint to nudge
MOVE_DELTA = 0.10           # rad (~5.7 deg)
SETTLE = 3.0                # s to wait after a joint command (traj is 2.0 s)
GRIP_SETTLE = 2.5           # s to wait after a gripper command
GRIP_PARTIAL_CLOSE = 0.5    # rad
GRIP_OPENISH = 0.1          # rad (stays clear of the URDF lower limit)


def main():
    init_logging()
    log = logging.getLogger("backend_test")

    config = KinovaGen3LiteConfig()
    config.action_time_from_start_sec = 2.0  # gentle
    joints = config.ros2_interface.arm_joint_names
    gripper = config.ros2_interface.gripper_joint_name

    robot = BackendRobot(config)
    log.info("Connecting to the real Gen3 Lite through the lerobot backend...")
    robot.connect()
    time.sleep(2.0)

    try:
        log.info(f"observation_features: {robot.observation_features}")
        log.info(f"action_features:      {robot.action_features}")

        def read():
            obs = robot.get_observation()
            arm = {j: obs[f"{j}.pos"] for j in joints}
            grip = obs[f"{gripper}.pos"]
            return obs, arm, grip

        _, arm0, grip0 = read()
        log.info("Initial state:")
        for j in joints:
            log.info(f"    {j}: {arm0[j]:+.4f}")
        log.info(f"    {gripper}: {grip0:+.4f}")

        def hold_action(grip_val, arm=None):
            arm = arm or arm0
            act = {f"{j}.pos": arm[j] for j in joints}
            act["gripper.pos"] = grip_val
            return act

        # ---- Phase 1: gripper through the backend --------------------------
        log.info(f"[1/2] GRIPPER via backend -> partial close ({GRIP_PARTIAL_CLOSE})")
        robot.send_action(hold_action(GRIP_PARTIAL_CLOSE))
        time.sleep(GRIP_SETTLE)
        _, _, g = read()
        log.info(f"      gripper now: {g:+.4f}")

        log.info(f"[1/2] GRIPPER via backend -> open-ish ({GRIP_OPENISH})")
        robot.send_action(hold_action(GRIP_OPENISH))
        time.sleep(GRIP_SETTLE)
        _, _, g = read()
        log.info(f"      gripper now: {g:+.4f}")

        # ---- Phase 2: single wrist-joint move through the backend ----------
        _, arm1, _ = read()
        start = arm1[MOVE_JOINT]
        target = dict(arm1)
        target[MOVE_JOINT] = start + MOVE_DELTA
        log.info(f"[2/2] JOINT via backend -> {MOVE_JOINT}: {start:+.4f} -> {target[MOVE_JOINT]:+.4f} (+{MOVE_DELTA})")
        act = {f"{j}.pos": target[j] for j in joints}
        act["gripper.pos"] = GRIP_OPENISH
        robot.send_action(act)
        time.sleep(SETTLE)
        _, arm2, _ = read()
        log.info(f"      {MOVE_JOINT} now: {arm2[MOVE_JOINT]:+.4f}  (moved {arm2[MOVE_JOINT]-start:+.4f})")

        log.info(f"[2/2] JOINT via backend -> return {MOVE_JOINT} to {start:+.4f}")
        act = {f"{j}.pos": arm1[j] for j in joints}
        act["gripper.pos"] = GRIP_OPENISH
        robot.send_action(act)
        time.sleep(SETTLE)
        _, arm3, _ = read()
        log.info(f"      {MOVE_JOINT} now: {arm3[MOVE_JOINT]:+.4f}  (residual {arm3[MOVE_JOINT]-start:+.4f})")

        log.info("DONE — backend connect/observe/gripper/joint round-trip complete.")

    finally:
        robot.disconnect()
        log.info("Disconnected.")


if __name__ == "__main__":
    main()
