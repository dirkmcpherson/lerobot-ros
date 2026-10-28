"""lerobot-record for the Gen3 Lite, homing between episodes.

Same command-line arguments as ``lerobot-record``. Stock lerobot-record gives the robot
no hook between episodes, so this wraps its ``record_loop``: the reset period (the call
made without a dataset) starts with ``robot.go_home()``, which opens the gripper and
drives slowly to ``--robot.home_position`` through the workspace guard. The SpaceMouse
teleop re-syncs to the homed pose on its next call (it sees the gap in calls). The rest
of ``--dataset.reset_time_s`` is teleoperated as usual, e.g. to put the cup back.

    python record_gen3_lite.py --robot.type=gen3_lite_ros --teleop.type=spacemouse_joint ...
"""

import logging

from lerobot.scripts import lerobot_record

_record_loop = lerobot_record.record_loop


def _record_loop_homing(*args, **kwargs):
    robot = kwargs.get("robot", args[0] if args else None)
    if kwargs.get("dataset") is None and hasattr(robot, "go_home"):
        logging.info("Episode done: opening the gripper and homing before the reset period.")
        robot.go_home()
    return _record_loop(*args, **kwargs)


lerobot_record.record_loop = _record_loop_homing

if __name__ == "__main__":
    lerobot_record.main()
