# Handoff: test Gen3 Lite HIL-SERL in Gazebo on the GPU machine

Written 2026-11-18 for whoever picks this up next (a coding agent, most likely) on a
machine with an RTX 3080 and Gazebo. Read this first, then
[GEN3_LITE_HILSERL.md](GEN3_LITE_HILSERL.md) for the design, the limits and the logs.

## Where things stand

Stock LeRobot HIL-SERL (lerobot 0.6.1) is wired to the Kinova Gen3 Lite through this
repo's ROS 2 stack: an end-effector delta robot class with its safety limits, takeover
flags and provenance tags on every stored transition, entry points that wrap the
unmodified upstream learner and actor, and CSV run logs. All of it was written on a
laptop with no ROS install, no GPU and no arm.

What that leaves untested, and what this pass is for:

| Never exercised | Why it matters |
|---|---|
| The real `joint_trajectory_controller` (a new one-point goal every 100 ms, each preempting the last) | The laptop check used a stand-in driver that jumps to each goal instantly. Tracking lag, goal rejection and stop-go motion are all unknown. |
| CUDA | Every run so far was CPU. The replay buffer's `torch.compile` path has never run at all (the laptop has no C++ compiler). |
| The mesh-based workspace guard from the installed `kortex_description` | Tested only against a URDF copied into `tests/fixtures/`. |
| A run long enough to show learning | The longest run was 600 steps on a fake arm. |

**Goal of this pass:** find out whether the system works against real ros2_control
controllers in Gazebo and on the GPU, and report what you measured, so the operator's
first session on the physical arm is not spent finding plumbing bugs. Fix what is broken
in the adapter or the glue; do not tune the learner.

## Rules

1. **Simulation only. Never command the physical arm.** The real driver and Gazebo expose
   the same topic and action names, so a command sent while both are up reaches both.
   Use a ROS domain of your own in every terminal (`export ROS_DOMAIN_ID=42`), and before
   any command that moves the robot confirm all three:

   ```bash
   ros2 topic list | grep -x /clock                      # present: a simulator is publishing time
   ros2 param get /controller_manager use_sim_time       # True
   pgrep -af robot_ip                                    # nothing: no real-arm launch running
   ```

   If you cannot establish that the physical arm is unreachable from your domain, stop
   and ask.
2. `gen3_lite_hil` refuses to move without `--run.confirm_motion=true`, in Gazebo too.
   Passing it is fine once rule 1 is satisfied. Do not weaken that gate.
3. Out of scope, as in the original brief: the SAC losses, the batch sampling ratios, the
   network architecture, the lerobot version, the ROS package layout, controller gains.
4. Do not change the default limits in `Gen3LiteHilConfig`. Anything Gazebo needs
   loosened goes in `config/hilserl/gazebo_*.json` or on the command line.
5. A fix to the adapter or the glue comes with a test (`tests/`, pytest, no hardware).
6. Do not commit or push unless asked. Report failures with their output; a stage that
   fails is a result, not something to work around quietly.

## What I do not know about that machine

Check these before anything else; several of the commands below assume them.

- **ROS distro.** This branch (`ros2_ubuntu24_04`) targets Jazzy on Ubuntu 24.04 with
  Python 3.12. If the machine is Ubuntu 26.04 / Lyrical (Python 3.14, branch
  `ros2_ubuntu26_04`), stop and report: lerobot 0.6.1 with the `hilserl` extra has not
  been tried on 3.14.
- **Kortex workspace.** `gen3_lite_env.sh` expects `/opt/ros/jazzy`, a built
  `~/workspace/ros2_kortex_ws` (override with `KORTEX_WS`) and a conda env
  `lerobot-ros-06` (override with `LEROBOT_ENV`). `docs/GEN3_LITE_DAGGER.md` describes
  that setup and a Gazebo smoke test already done with it.
- **NVIDIA driver.** `docs/GEN3_LITE_DAGGER.md` notes that pip pulls a CUDA torch that
  needs driver 580 or newer, and that on one earlier occasion the kernel had no NVIDIA
  module loaded. `nvidia-smi` and `torch.cuda.is_available()` settle it.
- **3080 memory** (10 or 12 GB). Irrelevant for the state-only Gazebo run; it matters for
  the real config (batch 256, 128x128 images).
- **A display.** Without one, Gazebo has to run headless (`-s`); the DAgger doc did that
  with a scratch copy of `kortex_sim_control.launch.py`.

## Stages

Do them in order. Each has a pass criterion; stop and report at the first failure you
cannot fix within the rules.

### A. Environment

```bash
. gen3_lite_env.sh && export ROS_DOMAIN_ID=42
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import rclpy, torch, lerobot; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"
pip install "lerobot[hilserl,dataset,training]==0.6.1" pin pytest      # if not already there
pip install -e . -e lerobot_robot_ros -e lerobot_teleoperator_devices
python -m pytest
```

Pass: lerobot 0.6.1, CUDA available, 64 tests pass. One test is skipped unless the
kortex meshes are found; point `KORTEX_DESCRIPTION_PARENT` at the directory containing
`kortex_description/` (the source tree, or the installed `share/`) so it runs.

### B. Fake arm on the GPU

No ROS involved. This is the first time CUDA and `torch.compile` are exercised, so do
not set `TORCH_COMPILE_DISABLE`.

```bash
python -m hilserl_gen3_lite.record  --config_path=config/hilserl/dryrun_env.json
python -m hilserl_gen3_lite.learner --config_path=config/hilserl/dryrun_train.json \
    --policy.device=cuda --run.dir=runs/dryrun_gpu &
python -m hilserl_gen3_lite.actor   --config_path=config/hilserl/dryrun_train.json \
    --policy.device=cuda --run.dir=runs/dryrun_gpu \
    --run.eval_interval_steps=150 --run.eval_episodes=3 --run.announce=false
python -m hilserl_gen3_lite.summarize_run runs/dryrun_gpu
```

The actor exits after 600 steps; the learner never exits on its own, so send it SIGINT
afterwards. Pass: `episodes.csv`, `train.csv`, `eval.csv`, `run_meta_*.json` exist and
`train.csv` has rows. Evaluation success reads 0 in this config because its scripted
operator takes over in every episode, which counts an evaluation episode as failed; look
at the `success` column of the training episodes instead. On the laptop CPU this run
reached policy-only successes in six steps by episode 35.

### C. The adapter over ROS, without Gazebo

```bash
python tests/ros_check/fake_driver.py 22 &
python tests/ros_check/check.py tests/fixtures/gen3_lite.urdf
```

Nothing else may be publishing `/joint_states` in your domain. Pass: last line
`ROS CHECK PASSED`. This passed in a Jazzy container on the laptop; it is here to catch
an environment difference before Gazebo adds its own.

### D. Gazebo: bring-up and the smoke test

Terminal 1 (leave running):

```bash
. gen3_lite_env.sh && export ROS_DOMAIN_ID=42
ros2 launch kortex_bringup kortex_sim_control.launch.py \
    use_sim_time:=true launch_rviz:=false robot_type:=gen3_lite dof:=6 gripper:=gen3_lite_2f
```

Terminal 2, after the rule 1 checks and `ros2 control list_controllers` shows
`joint_trajectory_controller` and the gripper controller active:

```bash
mkdir -p runs
python -m hilserl_gen3_lite.smoke_test --robot.type=gen3_lite_hil \
    --robot.use_sim_time=true --run.confirm_motion=true 2>&1 | tee runs/gazebo_smoke.txt
```

It homes with the guarded joint move, goes to the reset pose (tool at 0.30, 0.16,
0.22 m), idles 3 s, drives 3 cm out and back on each axis printing commanded against
measured tool position per step, checks the watchdog, and prints `SMOKE TEST PASSED` or
`FAILED` with counters. Pass: `PASSED`. Whatever the verdict, keep the full output: the
per-step error column is the first real measurement of tracking lag.

How to read failures:

| Symptom | Likely cause | What to do |
|---|---|---|
| `StaleObservationError: Joint state is N ms old` | The age is wall-clock time since the last `/joint_states` message; Gazebo below real time stretches it past 200 ms | Check the real-time factor. For simulation only, add `--robot.max_obs_age_s=1.0`. Report the factor and the ages printed by the idle loop. |
| Arm does not move, no error | Controller rejected the goals, or `joint_trajectory_controller` is not active | `ros2 control list_controllers`; the driver terminal's log. `docs/LYRICAL_KINOVA.md` §7.1 describes a joint-limit case that deactivates it. |
| `Move stopped: the workspace guard blocked the straight joint path` | Homing or the move to the reset pose crosses the table plane, the guard box or a self-collision margin | Note the start pose and the guard's reason. The sim spawns the base about 5 cm above the ground; the guard's plane is at the base's bottom face. |
| `FAIL: tracking error above 8 mm` after settling | Real lag or steady-state error | Report the numbers. The tolerance was a guess; do not just raise it. |
| Error column grows during a leg and `lead_clipped` steps appear (`clipped commands` counter) | Lag reaching the 3 cm lead limit at 0.1 m/s | Report. Each goal is due 0.2 s out (`action_time_from_start_sec`), so about 2 cm of lag was expected. |
| `IK rejected a step` | Target unreachable with the held orientation | Report the pose. It should not happen inside the default box near the reset pose. |
| `FAIL: loop period deviates` | The step path is slow: the guard's mesh check alone is about 21 ms (`docs/GEN3_LITE_DAGGER.md`) | Report mean and worst period. |
| Guard construction fails at connect | `gen3_lite_gen3_lite_2f.xacro` does not expand in this kortex checkout (it failed in one checkout on the laptop) | `--robot.urdf_path=<a generated URDF>`; report which kortex commit. |

Also record, once: `ros2 topic hz /joint_states`, and whether
`ros2 topic echo --once /joint_states` shows non-zero velocities while the arm moves.

### E. Gazebo: the full loop, unattended

There is no camera in the kortex Gazebo world and nobody to hold a gamepad, so
`config/hilserl/gazebo_env.json` and `gazebo_train.json` use a state-only policy (the
10-number state, no image encoder) and the scripted operator (`--teleop.type=scripted_ee`).
It drives the tool towards a target 7.5 cm from the reset pose, takes over on steps 10–39
of each of the first 40 episodes, then keeps its hands off, and reports success when the
tool is within 1.5 cm of the target. Everything else matches the real-arm config.

```bash
python -m hilserl_gen3_lite.record --config_path=config/hilserl/gazebo_env.json --run.confirm_motion=true

python -m hilserl_gen3_lite.learner --config_path=config/hilserl/gazebo_train.json \
    --run.dir=runs/gazebo_reach_s0                                   # terminal A

python -m hilserl_gen3_lite.actor --config_path=config/hilserl/gazebo_train.json \
    --run.dir=runs/gazebo_reach_s0 --run.confirm_motion=true \
    --run.eval_interval_steps=500 --run.eval_episodes=5 --run.announce=false   # terminal B

python -m hilserl_gen3_lite.summarize_run runs/gazebo_reach_s0
```

4000 environment steps, roughly 10–15 minutes. Stop the learner with SIGINT when the
actor exits. A run directory cannot be reused by the learner; use a new name per run.

Pass, plumbing: the run completes; no `StaleObservationError`, no IK rejections; the
actor log has no `Policy FPS ... below required` warnings beyond the first steps;
`train.csv` shows both buffers growing and all three batch shares non-zero.

Pass, behaviour: in the early episodes the takeover finishes the reach (success 1,
`takeovers` 1); later, episodes succeed with `intervention_steps` 0; evaluation success
rises across blocks. **This is not guaranteed.** The state-only reach has not been tuned,
and 500 steps on the fake arm were not enough for it. If it has not learned by 4000
steps, try `--policy.online_steps=15000` once before concluding anything, and report the
curves either way. Do not start adjusting learning rates, the discount or the
temperature: that is the learner, and it is out of scope.

Look through the actor's log (`runs/gazebo_reach_s0/actor_*/logs/`) for counts of
`Watchdog:`, `Workspace guard blocked`, `IK rejected` and `clamped`. One watchdog hold
per episode end is expected and harmless: the actor pauses to push transitions and load
weights, the 0.3 s watchdog fires, the environment then resets.

### F. Not in this pass

Images in simulation (needs a camera added to the Gazebo world), the reward classifier,
`crop_dataset_roi`, wandb, resuming a learner run, and anything on the physical arm.
Say so if one of them turns out to block a stage above.

## Things I would look at first if something is off

- **Stop-go motion.** Each command is a trajectory point that ends at rest 0.2 s out.
  The backend has an opt-in smoother (`stream_velocities`, with
  `config/gen3_lite_ros2_controllers_streaming.yaml`). Leave it off for the baseline;
  describe the motion you see.
- **Stored action against executed motion.** The buffer stores the action the policy or
  human asked for; the adapter may have shortened it at a limit. That matches upstream,
  whose bounds step also clips after the action is chosen. A policy pressed against the
  box will show up as `clipped commands` climbing.
- **Episode length** is `control_time_s x fps - 1` steps (upstream's time limit counts
  the reset observation). The scripted operator's `episode_steps` must equal it, or its
  takeover window drifts from episode to episode.
- **Shutdown.** The upstream actor does not disconnect the robot when it exits; the
  watchdog's hold is what stops the arm. Confirm in Gazebo that the arm is stationary
  after the actor ends, including after Ctrl+C mid-episode.
- **Reset** opens the gripper on every homing move (inherited from `Gen3LiteRos.go_home`).
  Harmless for a reach.
- **Upstream quirks already worked around** are listed in GEN3_LITE_HILSERL.md under
  "What the upstream code could not do as-is": the motor-bus environment, the config
  serialisation crash, and the replay buffer dropping `is_intervention`.

## What to hand back

1. Machine facts: OS, ROS distro, GPU and driver, torch version, kortex commit.
2. Per stage A–E: passed, failed or skipped, with the command and the relevant output.
3. From D: the smoke-test output, `/joint_states` rate, Gazebo's real-time factor, and
   the tracking error after settling and at its worst during a leg.
4. From E: `summary.json`, `curves.png`, the two rise-time numbers, and the warning counts.
5. Every code or config change you made, each with its reason and its test.
6. What you verified by running it, separately from what you inferred.
7. Anything that suggests the default limits are wrong for the real arm. Flag it; do not
   change the defaults.

## Map

| Path | What |
|---|---|
| `docs/GEN3_LITE_HILSERL.md` | Design, limits, log formats, the real-arm run book, verified / not verified |
| `lerobot_robot_ros/lerobot_robot_ros/gen3_lite_hil.py`, `ee_delta.py`, `config_gen3_lite_hil.py` | The robot class, the step limiter and kinematics, the limits |
| `lerobot_robot_ros/lerobot_robot_ros/gen3_lite_ros.py`, `lerobot_backends/ros2/` | The joint-space path underneath: step cap, workspace guard, ROS interface |
| `hilserl_gen3_lite/env.py` | Gym environment and processor pipelines |
| `hilserl_gen3_lite/{actor,learner,record}.py` | Entry points wrapping upstream; `compat.py` is the one in-memory upstream fix |
| `hilserl_gen3_lite/smoke_test.py`, `summarize_run.py` | Stage D and the run summary |
| `config/hilserl/` | `gazebo_*` (this pass), `dryrun_*` (fake arm), `*_gen3_lite_reach` (real arm; cameras are placeholders) |
| `tests/`, `tests/ros_check/` | 64 hardware-free tests; the ROS check of stage C |
| `docs/GEN3_LITE_DAGGER.md`, `docs/LYRICAL_KINOVA.md` | Earlier bring-up notes, including a Gazebo run of the joint-space robot |
