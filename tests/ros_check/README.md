# ROS-side check of the Gen3 Lite HIL adapter (no arm)

`pytest` covers the adapter on a fake backend. This check runs the same adapter on the
real `ROS2Backend` (rclpy, `/joint_states`, the FollowJointTrajectory and GripperCommand
actions) against `fake_driver.py`, a stand-in for the kortex driver that jumps to each
goal. It needs a ROS 2 Jazzy environment with `control_msgs`, and nothing else on the
ROS graph publishing `/joint_states`.

```bash
. gen3_lite_env.sh                                  # with the kortex driver NOT running
python tests/ros_check/fake_driver.py 22 &          # publishes for 22 s, then stops
python tests/ros_check/check.py tests/fixtures/gen3_lite.urdf
```

Expected last line: `ROS CHECK PASSED` (moves 13 cm to the tool-box face, one watchdog
hold, stale joint state detected and the next command refused).

On a machine without ROS, the same thing in a container (2026-11-18, image
`gen3-smoke:jazzy`, repo and venv mounted read-only at their host paths):

```bash
docker run --rm --network none -v $PWD:$PWD:ro -e HOME=/tmp gen3-smoke:jazzy bash -lc "
  source /opt/ros/jazzy/setup.bash; PY=$PWD/.venv-hilserl/bin/python
  \$PY $PWD/tests/ros_check/fake_driver.py 22 & sleep 2
  cd /tmp && \$PY $PWD/tests/ros_check/check.py $PWD/tests/fixtures/gen3_lite.urdf"
```
