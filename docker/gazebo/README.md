# Gen3 Lite in headless Gazebo, in a container

For testing the HIL-SERL adapter against the real ros2_control controllers on a machine
with Docker but no ROS install and no GPU. ROS 2 Jazzy, Gazebo Harmonic,
`gz_ros2_control`, the stock kortex controllers: `joint_trajectory_controller`,
`gen3_lite_2f_gripper_controller` (`position_controllers/GripperActionController`),
`joint_state_broadcaster`, and `twist_controller` (inactive).

```bash
# once (about 1 GB on top of ros:jazzy; --build-arg BASE=<local Jazzy image> reuses layers)
docker build --network host -t lerobot-ros-gazebo:jazzy docker/gazebo

docker/gazebo/run.sh up        # container + simulated arm; prints the controller list
docker/gazebo/run.sh exec 'python -m hilserl_gen3_lite.smoke_test --robot.type=gen3_lite_hil \
    --robot.use_sim_time=true --run.confirm_motion=true'
docker/gazebo/run.sh down
```

Targeted checks that need real controllers (sustained motion into the tool-box face,
descent to the box floor, the gripper action, a paused simulation, a no-go box, the
frozen orientation and a set-up turn):

```bash
docker/gazebo/run.sh exec 'python docker/gazebo/sim_checks.py'
```

Hard-kill test: start `python docker/gazebo/sim_checks.py --runaway` inside the container,
`kill -9` it after a few seconds, and watch `/joint_states`. Nothing in a killed process
can stop the arm; the trajectory controller finishes its last goal (due 0.2 s out) and
holds.

`run.sh exec '<command>'` runs anything else the same way. The full loop used for the
results in `docs/GEN3_LITE_HILSERL.md` (state-only reach, scripted operator; add
`--policy.device=cpu` on a machine without a GPU):

```bash
docker/gazebo/run.sh exec 'python -m hilserl_gen3_lite.record --config_path=config/hilserl/gazebo_env.json --run.confirm_motion=true'
docker/gazebo/run.sh exec 'python -m hilserl_gen3_lite.learner --config_path=config/hilserl/gazebo_train.json --policy.device=cpu --run.dir=runs/gazebo_reach' &
docker/gazebo/run.sh exec 'python -m hilserl_gen3_lite.actor --config_path=config/hilserl/gazebo_train.json --policy.device=cpu --run.dir=runs/gazebo_reach --run.confirm_motion=true --run.eval_interval_steps=500 --run.eval_episodes=5 --run.announce=false'
```

How it is put together:

- **No network.** The container runs with `--network none`. Nothing in it can reach a
  physical arm, and the host cannot see its ROS graph. `--run.confirm_motion=true` is
  safe here for that reason.
- **Python comes from the host.** The repo is mounted read-only at its host path, so the
  host virtualenv (`.venv-hilserl`, Python 3.12, the same as the image's) runs inside
  against the image's `rclpy`. Set `VENV=` for another one. `runs/` and `data/` are
  mounted writable; the container runs as your user, so the files are yours.
- **Model fix.** `ros-jazzy-kortex-description` 0.2.6 does not expand the Gen3 Lite
  (`Invalid parameter "gripper"`: `kortex_robot.xacro` passes an argument the Lite's
  `load_arm` macro does not declare). The Dockerfile declares it.
- **Launch.** `gen3_lite_sim.launch.py` is a slim, headless equivalent of
  `kortex_bringup`'s `kortex_sim_control.launch.py robot_type:=gen3_lite dof:=6
  gripper:=gen3_lite_2f`: same xacro, same controllers yaml, same physics engine
  (bullet-featherstone), `gz sim -s`, no RViz. It needs `kortex_description` only.
- The arm spawns at the all-zero pose (straight up), 0.3 m above the ground plane.
  `gen3_lite_hil` homes from there with its guarded joint move on connect.

Measured on a ThinkPad T490 (4 cores, no GPU), 2026-11-18: real-time factor 1.02 with the
arm idle, `/joint_states` at about 920 Hz.

Differences from the real arm worth remembering: no gravity sag or friction to speak of,
position-commanded joints follow their trajectory almost exactly, there is no camera in
the world, and the controllers yaml is the packaged one (not the streaming variant in
`config/`).
