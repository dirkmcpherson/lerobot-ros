# Gen3 Lite HIL-SERL: stock baseline (lerobot 0.6.1, ROS 2 Jazzy)

Unmodified LeRobot HIL-SERL (SAC, two replay buffers, 50/50 batches) driving the Kinova
Gen3 Lite through this repo's ROS 2 stack, with run logging good enough to measure rise
time. This file is the recon note, the per-phase report and the operator's run README.

**Status on 2026-11-18.** Everything below was built and tested on a laptop with no arm
and no GPU. The adapter, takeover flag, provenance tags, logging and the full learner +
actor loop run against a fake arm and, in a container, against a simulated Gen3 Lite in
Gazebo with the real ros2_control controllers ([Gazebo results](#gazebo-results)).
Nothing has run on the real arm or on a GPU: the hardware smoke test, the demos and the
baseline run are the operator's next steps ([Run book](#run-book)), and
[GEN3_LITE_HILSERL_HANDOFF.md](GEN3_LITE_HILSERL_HANDOFF.md) is the plan for the GPU
machine. The exact split is under [Verified and not verified](#verified-and-not-verified).

---

## Phase 0: recon

| # | Question | Answer |
|---|---|---|
| 1 | LeRobot | pip dependency, `lerobot==0.6.1` (`docs/GEN3_LITE_DAGGER.md`; packages pin `>=0.4.0,<0.7.0`). It contains the HIL-SERL modules in the recent layout: `lerobot/rl/{actor,learner,buffer,gym_manipulator,crop_dataset_roi}.py`, `lerobot/rl/algorithms/sac/`, `lerobot/rl/data_sources/data_mixer.py`. Differences from the brief's table: the SAC policy is `lerobot/policies/gaussian_actor/` with the losses in `lerobot/rl/algorithms/sac/sac_algorithm.py`; the reward classifier is `lerobot/rewards/classifier/`; needs the `hilserl` extra (`grpcio`, `gym-hil`, `placo`). |
| 2 | ROS 2 | Jazzy on this branch (`ros2_ubuntu24_04`), Kinova `ros2_kortex` built in `~/workspace/ros2_kortex_ws`. Controllers: `joint_state_broadcaster`, `joint_trajectory_controller` (active), `twist_controller` (`picknik_twist_controller`, loaded), `gen3_lite_2f_gripper_controller`, `fault_controller`. A twist interface exists for the lite (`/twist_controller/commands`, `geometry_msgs/Twist`) but it is in the **tool** frame and needs a controller switch away from `joint_trajectory_controller` (`docs/LYRICAL_KINOVA.md` §7.3). |
| 3 | Interfaces | Joint state: `/joint_states` (`sensor_msgs/JointState`, joints 1–6 + `right_finger_bottom_joint`). Arm command: action `/joint_trajectory_controller/follow_joint_trajectory` (`control_msgs/FollowJointTrajectory`), one point per 10 Hz tick, due 0.2 s out, each goal preempting the last. Gripper: action `/gen3_lite_2f_gripper_controller/gripper_cmd` (`control_msgs/GripperCommand` on Jazzy), 0.0 open .. 0.85 closed. End-effector pose: not a topic; computed from joints with pinocchio and the kortex URDF (`tool_frame` in `base_link`). Cameras: none configured in the repo; they go through lerobot camera classes (OpenCV / RealSense), not ROS topics. Rates of `/joint_states` on the arm: not measured here. |
| 4 | Existing glue | `--robot.type=gen3_lite_ros` (`lerobot_robot_ros`): absolute joint targets, per-joint step cap, workspace guard (table plane, tool box, self-collision from the URDF meshes), guarded homing. `--teleop.type=spacemouse_joint`. A `gamepad_6dof` teleop for velocity robots. All reused: the HIL robot subclasses `Gen3LiteRos`. lerobot ships a `gamepad` teleop with exactly the delta x/y/z + gripper actions and takeover events HIL-SERL needs, so no new gamepad class was written. |
| 5 | Environment | Yes, one interpreter: conda env `lerobot-ros-06` (Python 3.12) + sourced Jazzy (`gen3_lite_env.sh`). Checked here in the `gen3-smoke:jazzy` container: `rclpy` and lerobot 0.6.1 / torch import together and the adapter runs on the real `ROS2Backend`. |
| 6 | Tests | The repo had no test suite or CI; sub-packages carry a ruff config (line length 110). Added `tests/` (pytest, `python -m pytest`), ruff-clean under that config. No `CLAUDE.md`. |

No stop condition applied: the HIL-SERL modules are present, `rclpy` and lerobot share
an interpreter, and the existing joint path runs at 10 Hz.

### Command path: Option B (inverse kinematics, joint position targets)

Chosen over the twist interface because:

- It is the path already verified on this arm: joint targets at 10 Hz through
  FollowJointTrajectory, with the workspace guard checking every command. The guard
  works on joint configurations (link meshes against the table plane and each other),
  which a twist command would bypass.
- The twist controller is in the tool frame and needs a controller switch
  (`SINGLE_LEVEL_SERVOING`); homing and the guard both live on the joint controller.
- It is what upstream HIL-SERL does (end-effector delta, IK, joint targets).

IK is pinocchio damped least squares (the library the repo's SpaceMouse teleop and guard
already use), not upstream's placo solver, and it runs inside the robot class rather
than in the processor pipeline, so the limits cannot be configured away.

### What the upstream code could not do as-is

1. `gym_manipulator.RobotEnv` reads `robot.bus.motors` and resets through
   `robot.bus.sync_read/sync_write` (Feetech/Dynamixel bus). The Gen3 Lite has no bus.
   `hilserl_gen3_lite/env.py` supplies an environment and processor builder with the
   same interface; the entry points swap them in. The takeover steps
   (`AddTeleopActionAsComplimentaryDataStep`, `AddTeleopEventsAsInfoStep`,
   `InterventionActionProcessorStep`) and everything in the learner are upstream's.
2. `cfg.to_dict()` fails for any `gym_manipulator` env config whose `processor.reset`
   is set (`ResetConfig.fixed_reset_joint_positions: Any | None` cannot be encoded by the
   pinned draccus 0.11.6). The learner calls it on start-up. Fixed in memory at start-up
   by `hilserl_gen3_lite/compat.py` (narrows that one field's type).
3. The replay buffer keeps only the `complementary_info` keys of the first transition it
   sees, and `concatenate_batch_transitions` leaves a key that exists on one side only
   at half length. With demos recorded by stock `gym_manipulator` (only
   `discrete_penalty`), `is_intervention` is dropped from the offline buffer and is
   misaligned in the mixed batch. No buffer change was needed: the demo recorder here
   writes the same columns the actor sends, and the learner wrapper refuses to sample if
   the two buffers' keys differ.

4. The learner builds the offline (demo) buffer through `make_dataset`, which returns
   images as `uint8` in [0, 255]; `ReplayBuffer` stores them as floats without rescaling,
   while the actor sends images in [0, 1]. Demo images were 255 times brighter than
   online ones in every mixed batch. `compat.py` wraps the buffer's dataset conversion so
   they come out in [0, 1]. (Found and fixed by a second agent after the first pass; the
   takeover test now loads the demos the way the learner does, so it would catch this.)

5. lerobot's `GripperPenaltyProcessorStep` reads the last action number as a gripper
   joint target (its SO-100 pipeline converts to joint targets first). Here that number
   is the command itself (0 close, 1 stay, 2 open); fed to the upstream step, "stay"
   reads as a close command and every step with the gripper open is penalised.
   `env.py` has `GripperTogglePenaltyStep` instead: the environment reports when a
   command changes what the gripper is asked to do, and only that step is penalised.

No file of the installed lerobot package is edited. Runtime substitutions, all made by
the entry points in `hilserl_gen3_lite/`:

| Upstream name | Replaced by | Why |
|---|---|---|
| `lerobot.rl.actor.make_robot_env`, `make_processors` | `hilserl_gen3_lite.env` | item 1 |
| `lerobot.rl.actor.push_transitions_to_transport_queue` | wrapper: tag + log the episode, then the original | provenance, episode log |
| `lerobot.rl.actor.act_with_policy` | wrapper: open the run log, then the original | logging |
| `lerobot.rl.learner.OnlineOfflineMixer` | subclass that only observes the batches it yields and checks buffer keys | item 3, batch shares |
| `lerobot.rl.learner.RLTrainer` | subclass: `training_step` unchanged, then a CSV row | train log |
| `ResetConfig.fixed_reset_joint_positions` type | `list[float] \| None` | item 2 |
| `ReplayBuffer._lerobotdataset_to_transitions` | wrapper: `uint8` images to float in [0, 1] | item 4 |

### Files

Added:

| Path | What |
|---|---|
| `lerobot_robot_ros/lerobot_robot_ros/ee_delta.py` | `limit_step` (action clip, step cap, speed limit, lead limit, tool box) and `ArmKinematics` (pinocchio FK / IK). No ROS imports. |
| `.../config_gen3_lite_hil.py`, `.../gen3_lite_hil.py` | `--robot.type=gen3_lite_hil`: end-effector delta robot, watchdog, staleness. |
| `.../fake_backend.py` | `FakeArmBackend` and `--robot.type=gen3_lite_hil_fake` (no ROS, no arm; synthetic camera). |
| `lerobot_teleoperator_devices/.../scripted_ee.py`, `config_scripted_ee.py` | `--teleop.type=scripted_ee`: stand-in operator for tests, dry runs and unattended Gazebo runs. |
| `hilserl_gen3_lite/env.py` | Gym environment and processor pipelines. |
| `hilserl_gen3_lite/provenance.py` | `source`, `takeover_onset`, `episode_index`, `step_index`. |
| `hilserl_gen3_lite/{actor,learner,record}.py` | Entry points wrapping upstream. |
| `hilserl_gen3_lite/{run_logging,eval_schedule,settings,compat,summarize_run}.py` | CSV logs, evaluation blocks, `--run.*` arguments, the draccus fix, the run summary. |
| `hilserl_gen3_lite/smoke_test.py` | Supervised hardware smoke test. |
| `hilserl_gen3_lite/workspace.py` | Interactive setup: drive the arm to teach the tool box, the no-go boxes, the gripper orientation and the start pose; try them live; save into the task's config files. |
| `config/hilserl/*.json` | `env_*.json` (workspace setup, record) and `train_*.json` (learner + actor) for two real-arm tasks, `gen3_lite_reach` (gripper unused) and `gen3_lite_cup` (pick and place); `dryrun_*.json` and `dryrun_cup_*.json` (the same on the fake arm), `gazebo_*.json` (Gazebo, state-only, scripted operator). |
| `docker/gazebo/` | Image, headless launch file and `sim_checks.py` for a simulated Gen3 Lite with the real controllers; see its README. |
| `tests/` | 92 tests, URDF fixture (`tests/fixtures/gen3_lite.urdf`, from `kortex_description`), `tests/ros_check/`. |

Touched (small, backward compatible):

| Path | Change |
|---|---|
| `lerobot_backends/ros2/__init__.py` | `ROS2Backend` imported lazily, so configs import without `rclpy`. |
| `lerobot_backends/robot.py` | `BackendRobot(config, backend=None)`: a backend can be injected. |
| `lerobot_backends/ros2/backend.py` | `joint_velocities()`, `joint_state_age`, `camera_frame_ages()`. |
| `lerobot_backends/ros2/workspace_guard.py`, `config_gen3_lite_ros.py` | optional `package_dirs` / `mesh_package_dirs` for the meshes. |
| `lerobot_robot_ros/.../gen3_lite_ros.py` | `send_joint_action()` split out of `send_action()`; homing uses it. Behaviour unchanged. |
| `pyproject.toml`, `.gitignore`, package `__init__.py`s | new package, pytest path, `runs/`. |

---

## Phase 1: the robot adapter (`gen3_lite_hil`)

**Actions.** `delta_x`, `delta_y`, `delta_z` in [-1, 1]; one unit is `ee_step_size_m`
(1 cm) of tool travel in the base frame. Roll, pitch and yaw are frozen: at the
orientation of the start pose (`reset_joint_positions`) when one is set, the same in
every episode; otherwise at whatever the tool has when control (re)starts. The policy
cannot turn the tool; the orientation is chosen once, in the workspace setup. Kinova
home's orientation, which the default start pose keeps, is the gripper level and
pointing 60 degrees to the left of straight ahead. Optional `gripper` (0 close, 1 stay,
2 open), off by default.

**Observations.** `joint_1.pos` .. `joint_6.pos`, `gripper.pos`, `ee.x/y/z`, one image per
camera; joint velocities behind `observe_joint_velocities` (off: the driver has reported
zeros, see the streaming controllers yaml). The environment packs the scalars into a
10-number `observation.state`. `robot.observation_ages()` gives each source's age.

**Limits, enforced in `Gen3LiteHil.send_action` on every command** (defaults):

| Limit | Default | Field |
|---|---|---|
| Action clip | each axis to [-1, 1]; NaN/inf refused | |
| Step length | 1 cm per command | `max_ee_step_m` |
| Speed | 0.1 m/s: a step is at most speed x time since the last command | `max_ee_speed_m_s` |
| Lead | target at most 3 cm ahead of the measured tool | `max_ee_lead_m` |
| Tool box | x 0.15–0.43, y 0.03–0.29, z 0.04–0.40 m (base frame); commands are clipped to it | `ee_bounds_min/max` |
| IK check | a step whose solution misses by > 4 mm or turns the tool > 0.03 rad from the frozen orientation, or moves any joint more than 0.2 rad from the previous target (a wrist about to reconfigure), is not sent | `ik_*_tolerance`, `ik_max_joint_step_rad` |
| Joint step | no joint commanded more than 0.1 rad from its measured position; the whole joint step is scaled to fit, so the command stays on the intended line | `max_relative_target` |
| Workspace guard | table plane, guard box 0.13–0.45 / 0.01–0.31 / 0.02–0.50, self-collision (existing); a blocked command holds the last accepted target | `guard_enabled`, `plane_z`, `workspace_min/max` |
| No-go boxes | boxes in the base frame around anything rigid in reach; no moving link, fingers included, may come within 2 cm; checked by the workspace guard with the collision meshes, a violating command holds | `no_go_boxes`, `no_go_margin` |
| Watchdog | hold at the measured pose after 0.3 s (3 periods) without an action | `watchdog_periods` |
| Stale state | joint state older than 200 ms: no command, `StaleObservationError`; stale camera frame: error | `max_obs_age_s` |
| Disconnect | commands a hold first | |

The tool box must lie inside the guard box (checked when the config is built). Where the
arm is outside the tool box (Kinova home is, slightly), a zero action does not move it,
motion back in is allowed, motion further out is not.

For a position-controlled arm the "zero command" is a hold: the measured joint positions
sent once as the goal. On stale joint state the measured pose is unknown, so nothing is
sent and the controller keeps its last goal, at most one step away.

**Tests** (`python -m pytest tests/test_ee_delta.py tests/test_gen3_lite_hil.py`): clipping,
step and speed caps, all six box faces, a random policy pushed into the box corners,
staleness of joint state and camera, watchdog (trips once, stays quiet while actions
arrive, re-arms), connect/disconnect twice, hold on disconnect, gripper commands, IK
rejection, homing, and the mesh-based workspace guard stopping the tool above a table
plane under the new action path.

**Smoke test for the operator** (moves the arm; see the run book):

```bash
python -m hilserl_gen3_lite.smoke_test --robot.type=gen3_lite_hil --run.confirm_motion=true
```

The joint-step cap matters more than it looks. Each goal is due 0.2 s out, so at
0.1 m/s the arm runs about 2 cm behind its target, and a joint that needs 0.04 rad per
step is then already 0.08–0.12 rad from its measured position: the cap binds on roughly a
third of all commands in a training run. Two versions of this code handled that badly and
were fixed after being seen in Gazebo: taking the capped pose as the new reference bent
the path (5 mm of z over 13 cm of x), and clamping joints one by one produced a mixed
pose up to 2 cm off the line (the workspace guard stopped it). The reference now stays on
the intended line and the whole joint step is scaled.

**Acceptance.** Unit tests: pass. In Gazebo the smoke test passes ([results](#gazebo-results)).
"Operator reports the smoke test moved the arm as
commanded" and "10 Hz loop without stale-data errors on the arm": open, operator's step.
On the fake arm and over real `rclpy` with a stand-in driver the step path takes 3–4 ms.

## Phase 2: teleop and the intervention flag

**Teleop.** lerobot's `--teleop.type=gamepad` (pygame), unchanged:

| Control | Effect |
|---|---|
| Hold RB (upper right, pygame button 5) | take over; release hands control back |
| Left stick | x / y |
| Right stick, vertical | z |
| Buttons 6 / 7 | close / open the gripper while held (it then stays); only with `use_gripper` |
| Y (button 3) | success: reward 1, episode ends |
| B (button 1) | failure: episode ends |
| A (button 0) | re-record (demo recording) |

While RB is held the human action is executed and stored (upstream
`InterventionActionProcessorStep`).

**Stored fields.** Every transition's `complementary_info` has, as floats:
`is_intervention`, `source` (0 demo, 1 intervention, 2 policy), `takeover_onset` (1 on the
first human step of each takeover), `episode_index`, `step_index`, plus upstream's
`discrete_penalty`. The actor's transitions are tagged per episode just before they are
sent; demos are written with the same columns (`source = demo`, `is_intervention` = RB
held, no onsets). Demo and online episodes are numbered separately: pair
`episode_index` with `source`.

**Does the buffer return them on sample?** Yes, unchanged, provided both buffers were
started with the same keys (item 3 above). No pass-through patch was needed.

**Test** (`tests/test_takeover_provenance.py`): two episodes with one takeover each and
two demos are recorded through the real upstream pipeline steps on the fake arm, loaded
into upstream `ReplayBuffer`s the way the learner does, and sampled through upstream
`OnlineOfflineMixer`. Checked: the flags and tags per step; the stored action is the
human's during the takeover and the policy's otherwise; the arm moved accordingly; and in
the sampled batch every provenance field covers the whole batch and agrees row by row
with the action. **Acceptance: met** (on the fake arm).

## Phase 3: baseline run

Prepared: configs, scripts, logging. Not done: the run itself (needs the arm, the GPU
and the operator). The same commands were run end to end on the fake arm with a scripted
operator (`config/hilserl/dryrun_*.json`, CPU): 5 demos, then 600 environment steps with
upstream learner and actor exchanging transitions and weights, producing every log file
below. That shows the plumbing works; it says nothing about learning on the real task.

### Logs

Everything lands in the directory given as `--run.dir` (the same for learner and actor):

| File | Row per | Columns |
|---|---|---|
| `episodes.csv` | episode | `episode`, `env_steps` (cumulative, all episodes), `train_env_steps` (excluding evaluation), `wall_time_s`, `success`, `episode_reward`, `length`, `intervention_steps`, `intervention_fraction`, `takeovers`, `is_eval` |
| `train.csv` | training log step (`log_freq`) | `optimization_step`, `wall_time_s`, `loss_critic`, `mean_q`, `loss_actor`, `loss_temperature`, `temperature`, `loss_discrete_critic`, grad norms, `online_buffer_size`, `offline_buffer_size`, `batch_frac_demo`, `batch_frac_intervention`, `batch_frac_policy`, `batch_frac_is_intervention` |
| `eval.csv` | evaluation block | `block`, `env_steps_at_start`, `env_steps_at_end`, `episodes`, `successes`, `success_rate`, `episodes_with_takeover`, `mean_length` |
| `rise_time.json` | (rewritten after each block) | `env_steps_to_50pct`, `env_steps_to_90pct`, `final_eval_success_rate` |
| `run_meta_learner.json`, `run_meta_actor.json` | | full config, git commit and dirty flag, seed, lerobot version, run settings |
| `learner/`, `actor_<time>/` | | lerobot's own output dirs: text logs, checkpoints, the buffers saved as datasets |

`python -m hilserl_gen3_lite.summarize_run runs/<name>` prints the two rise-time numbers
and the final success rate, and writes `summary.json` and `curves.png`.

Definitions worth knowing before comparing runs:

- **Evaluation block.** Every `--run.eval_interval_steps` environment steps (default
  2000) the next `--run.eval_episodes` episodes (default 10) are evaluation episodes: the
  actor says "evaluation, hands off". Takeover still works, for safety; an evaluation
  episode with any takeover counts as a failure. The policy still samples its actions,
  as in the original HIL-SERL evaluation script; the transitions still go to the
  learner, as upstream's actor has no other mode.
- **Rise time** is measured in `env_steps` at the start of the block, counting every step
  the robot took, evaluation included.
- **`mean_q`** is the critic ensemble's mean prediction on the logged batch (one extra
  no-grad forward pass per log step).
- **Batch shares.** `online_ratio` 0.5: half the batch from the online buffer (policy
  steps and interventions), half from the offline buffer (demos and interventions). So
  `batch_frac_policy` is below 0.5 whenever the operator has intervened.
- An episode lasts `control_time_s x fps - 1` steps at most (upstream's time limit also
  counts the reset observation).

If wandb is wanted as well, set `wandb.enable` in the train config: the same extra
numbers reach it through lerobot's own logging.

---

## Run book

This run book is written for the gripper-less reach (`*_gen3_lite_reach.json`). The
first real session is a pick and place with a cup instead: its configs are
`*_gen3_lite_cup.json` and its step-by-step plan, including the interactive workspace
setup, is [GEN3_LITE_HILSERL_HANDOFF.md](GEN3_LITE_HILSERL_HANDOFF.md). The
commands below are the same with the cup configs substituted.

The operator runs every command in this section that moves the arm. Each one prints its
limits first and refuses to start on the real arm without `--run.confirm_motion=true`.

### 0. Once

```bash
. gen3_lite_env.sh
pip install "lerobot[hilserl,dataset,training]==0.6.1" pin pytest
pip install -e . -e lerobot_robot_ros -e lerobot_teleoperator_devices
python -m pytest                      # 92 tests, no hardware; ~60 s
```

Rehearse the whole pipeline without the arm (CPU is enough; set `TORCH_COMPILE_DISABLE=1`
on a machine without a C++ compiler):

```bash
python -m hilserl_gen3_lite.record  --config_path=config/hilserl/dryrun_env.json
python -m hilserl_gen3_lite.learner --config_path=config/hilserl/dryrun_train.json --run.dir=runs/dryrun &
python -m hilserl_gen3_lite.actor   --config_path=config/hilserl/dryrun_train.json --run.dir=runs/dryrun \
    --run.eval_interval_steps=150 --run.eval_episodes=3 --run.announce=false
python -m hilserl_gen3_lite.summarize_run runs/dryrun
```

(In the dry run the scripted operator also takes over during evaluation episodes, so
evaluation success reads 0; the training-episode success column is the one to look at.)

### 1. Smoke test (arm moves)

Driver up (`ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.1.10
launch_rviz:=false`), arm homed, table clear, hand on the emergency stop.

```bash
python -m hilserl_gen3_lite.smoke_test --robot.type=gen3_lite_hil --run.confirm_motion=true
```

It homes (if not there), moves to the reset pose (tool at 0.30, 0.16, 0.22 m), idles 3 s
checking the 10 Hz loop and staleness, drives 3 cm out and back along +x, -x, +y, -y,
+z, -z printing commanded against measured tool position, checks the watchdog, and ends
with `SMOKE TEST PASSED` or `FAILED`. Please send back the printed output either way.

### 2. Workspace, cameras, task

The tool box, the no-go boxes, the gripper orientation and the start pose are taught
interactively (arm moves) and saved into both config files of the task:

```bash
python -m hilserl_gen3_lite.workspace --config_path=config/hilserl/env_gen3_lite_reach.json \
    --run.confirm_motion=true
```

Drive with the gamepad, type `help` for the commands (`w` sweep the tool box, `n NAME`
mark a no-go box, `yaw` / `pitch` / `roll` turn the gripper, `r` set the start pose,
`show`, `save`). Everything taught is enforced at once, so it can be tried before saving.

Still edited by hand, the same way in both files, under `env.robot`:

- `cameras`: a **placeholder** (one OpenCV camera, index 0, 640x480). Replace with the
  real cameras. Each camera `name` also needs an `observation.images.<name>` entry in
  `env.features`, `env.features_map`, `policy.input_features` and `policy.dataset_stats`.
- If the state or action size changes (gripper on, velocities on), update
  `observation.state` / `action` shapes and `policy.dataset_stats`; `tests/test_configs.py`
  checks the files agree with the robot.

### 3. Demos (arm moves)

```bash
python -m hilserl_gen3_lite.record --config_path=config/hilserl/env_gen3_lite_reach.json \
    --run.confirm_motion=true
```

15 episodes by default (`dataset.num_episodes_to_record`), into `data/gen3_lite_reach_demos`.
Hold RB and drive; press Y when the task is done.

### 4. Crop (optional for the first run)

```bash
python -m lerobot.rl.crop_dataset_roi --repo-id local/gen3_lite_reach_demos --root data/gen3_lite_reach_demos
```

Upstream's tool: draw the region of interest per camera, it prints `crop_params_dict` and
writes a cropped copy of the dataset. Put the printed dict into
`env.processor.image_preprocessing.crop_params_dict` in both configs, and point
`dataset` in the train config at the cropped copy. Not exercised here (it needs a
display). Without it, full frames are resized to 128x128.

### 5. Reward

The configs use the **operator success key** (Y): reward 1 and episode end on the step it
is pressed, 0 otherwise. Nothing to train. The classifier route is upstream's
(`lerobot/rewards/classifier`, then `env.processor.reward_classifier.pretrained_path`);
the processor step is wired but untested here.

### 6. Train (arm moves)

Terminal A, learner (no motion):

```bash
python -m hilserl_gen3_lite.learner --config_path=config/hilserl/train_gen3_lite_reach.json \
    --run.dir=runs/reach_baseline_s0
```

Terminal B, actor, once the learner prints `gRPC server started`:

```bash
python -m hilserl_gen3_lite.actor --config_path=config/hilserl/train_gen3_lite_reach.json \
    --run.dir=runs/reach_baseline_s0 --run.confirm_motion=true \
    --run.eval_interval_steps=2000 --run.eval_episodes=10
```

Stop with Ctrl+C in the actor, then the learner. Then:

```bash
python -m hilserl_gen3_lite.summarize_run runs/reach_baseline_s0
```

For another seed, change `seed` in the train config and use a new `--run.dir`. The
learner refuses a run directory it has already used. If only the actor is restarted
within a run, its counters continue from `actor_state.json`.

### Operator notes (from the HIL-SERL paper and follow-ups)

- Intervene often early, then less as the policy improves.
- Give short, specific corrections and let the robot explore otherwise.
- Avoid driving whole episodes to success by hand: that inflates the critic's value
  estimates and destabilises training.
- During an evaluation block keep hands off; take over only for safety (the episode then
  counts as a failure).

### Safety

- No torque sensing is assumed: the arm will not detect a collision. The software limits
  above are the protection, together with the emergency stop.
- The limits are enforced in `Gen3LiteHil`, not only in the JSON; the JSON can tighten
  or move them. Controller gains, firmware and Kinova protection zones are untouched.
- If a log shows the tool outside `ee_bounds` by more than tracking error, or a step
  longer than `max_ee_step_m`, stop and report it before anything else.

---

## Verified and not verified

Verified here (2026-11-18, laptop, no arm):

- 92 unit tests pass (`python -m pytest`), including the mesh-based guard under the new
  action path and the Phase 2 takeover test through upstream pipeline, buffer and mixer.
- Forward kinematics of the URDF fixture puts Kinova home at (0.439, 0.193, 0.445) m,
  against (0.439, 0.193, 0.443) measured on the real arm (`docs/GEN3_LITE_DAGGER.md`).
- In Gazebo with the real controllers: the smoke test, 16 targeted checks and three
  full learner + actor runs ([Gazebo results](#gazebo-results)).
- The gripper path (pick and place): 4-number actions, the discrete gripper head and the
  toggle penalty, with a scripted operator closing the gripper at one point and opening
  it at another. Fake arm with images: 24 of 24 episodes; Gazebo, state-only, with the
  real `GripperActionController` (1.4 s to close): 28 of 28 episodes, exactly two
  penalised steps per demo, no guard blocks or IK rejections. Both runs were short and
  driven almost entirely by the scripted operator: they show the plumbing, not learning.
- The adapter on the real `ROS2Backend` over `rclpy` (Jazzy container) against a stand-in
  driver: observations, 20 commands at 10 Hz, stop at the tool-box face, gripper action,
  one watchdog hold, and stale joint state detected on the first 0.2 s poll after
  `/joint_states` stopped, with the next command refused (`tests/ros_check/`).
- Learner + actor + demos end to end on the fake arm, CPU, 600 steps; all log files
  written; the smoke-test script rehearsed on the fake arm.
- In the collision model, the straight joint path from Kinova home to the reset pose is
  clear, and all eight corners of the default tool box are reachable with the held
  orientation except the two lowest ones nearest the base, where the guard stops the
  wrist 1–3 mm above the table plane (the guard doing its job). From 150 random poses
  in the tool box, the straight joint path back to the reset pose (what the environment
  does between episodes) was clear of the guard every time.
- The policy and SAC algorithm build from `train_gen3_lite_reach.json` (CPU, with the
  `helper2424/resnet10` encoder downloaded) and produce a 3-number action from a
  128x128 image and the 10-number state.

Not verified:

- Anything on the real arm: tracking error with real dynamics and gravity, the 10 Hz
  loop with real cameras, real `/joint_states` age against the 200 ms limit, the gamepad
  in hand.
- Anything on a GPU, and `torch.compile` in the replay buffer (no compiler here).
- Any run with images against real controllers: the Gazebo runs are state-only.
- Grasping a physical object, and learning a pick and place at all.
- The real configs beyond parsing, consistency checks and building the policy: cameras
  are placeholders, GPU memory is unknown (batch 256, 128x128 images, `utd_ratio` 2 are
  upstream's example values), and the pretrained encoder `helper2424/resnet10` is fetched
  from the Hugging Face hub on first use (it ships its own model code).
- `crop_dataset_roi`, the reward classifier, wandb, resuming a learner run.
- Whether the arm's `/joint_states` carries real velocities (Gazebo's does).
- `kortex_bringup`'s own launch files and the operator's kortex workspace: the container
  uses the apt `kortex_description` with a one-line fix and a slim launch file.

### Gazebo results

2026-11-18, `docker/gazebo/` on a ThinkPad T490 (4 cores, no GPU, everything on CPU):
Gazebo Harmonic, `gz_ros2_control`, the stock `joint_trajectory_controller` and
`position_controllers/GripperActionController`, real-time factor about 1.0.

**Smoke test** (`hilserl_gen3_lite.smoke_test`, final code): passed. Loop period 100.2 ms
with 0.7 ms worst deviation; joint state 0.3–0.6 ms old; guarded homing from the all-zero
spawn pose and the move to the reset pose went through; lag 18–21 mm on the third step of
each 3 cm leg, 0.3–1.8 mm error half a second after stopping; back within 0.7 mm of the
start after six legs; watchdog hold on cue.

**Targeted checks** (`docker/gazebo/sim_checks.py`): all 16 passed, and the 9 added since (below).

| Check | Result |
|---|---|
| 20 unit steps in +x, box face 13 cm away | stopped at x = 0.4300 with no overshoot, y and z held; lag 23.5 mm on average, 29.9 mm at most (the 3 cm lead limit acting) |
| Tool speed during that move | commanded at 0.1 m/s; measured up to 0.119 m/s while the arm caught up |
| 30 unit steps in -z | stopped at the box floor, z = 0.0400; measured pose never violated the workspace guard |
| Gripper action | finger joint 0.850 after close, 0.000 after open |
| Simulation paused mid-motion | next command refused 236–245 ms after the pause (limit 200 ms, polled every 20 ms); `get_observation` raised; control recovered after resume |
| `kill -9` of the commanding process mid-motion | arm stationary when next observed, about a second later |
| No-go box 11 cm from the tool | the arm stopped 8 cm short of its face (a fingertip is 6 cm ahead of the tool point; margin 2 cm), no link ever inside the margin, backing away not blocked |
| Frozen orientation through a 12 cm diagonal move and back | 0.10 degrees of deviation at worst while moving, 0.002 at rest |
| A 20 degree yaw set-up turn | carried out on the spot (tool moved under 0.1 mm), and the new orientation then held |
| The interactive workspace tool, with typed commands | tool box swept, gripper turned, start pose set, no-go box added, both config files saved and loadable |

**Full loop** (`config/hilserl/gazebo_*.json`: state-only policy, scripted operator taking
over in the first 40 episodes, 4000 steps, evaluation of 5 episodes every 500 steps). Same
seed three times, once per version of the adapter:

| Run | Adapter | Steps to 50 % | Steps to 90 % | Final | Guard blocks | IK rejections | Actor exit |
|---|---|---|---|---|---|---|---|
| `gazebo_reach_s0` | as first written | 2004 | 2504 | 5/5 | 0 | 0 | not checked |
| `gazebo_reach_s0b` | reference kept on the line, joints still clamped one by one | 2014 | 2504 | 5/5 | 4 (tool commanded up to 2 cm outside its box) | 4 | hung; killed |
| `gazebo_reach_s0c` | final: whole joint step scaled, IK jump check, env closed on exit | 2002 | 2508 | 5/5 | 0 | 3 | clean, code 0 |

In the final run the operator's takeovers finished all 35 early training episodes, the
policy alone then succeeded in every evaluation episode from 2508 steps on, in 12 to 14
steps each. Limit counters for its 3995 commands: per-joint cap 1465 (37 %), step-length
or speed cap 3145 (mostly diagonal actions: a unit action on two or three axes is longer
than 1 cm and is scaled back), no guard blocks, no stale stops, no watchdog trips, 3 IK
rejections (the arm held for that step). The actor logged 9 `Policy FPS below required`
warnings (7–9 Hz) with Gazebo, the learner and the actor sharing four cores; the learner
made 15–20 updates per second.

**Defects this found**, all fixed with tests: the reference drifting off the commanded
line when the per-joint cap shortened a step; joints clamped one by one giving a pose off
the line; the actor process hanging at exit because nothing disconnected the robot.

**What it does not show.** Gazebo's position-controlled joints follow their trajectory
almost exactly and there is no gravity sag, friction or backlash to speak of, so lag and
settling error on the real arm will differ. The three runs share a seed and a scripted
operator; they show the loop learns the reach, not how fast it would with a person.

## Open questions for the operator

Answered on 2026-11-18: the first task is a pick and place with a plastic cup; one
camera, head-on to the arm; no rotation in the action space (x, y, z and the gripper;
orientation fixed by the reset pose); reward from the success key.

- [ ] Camera device and resolution (the config has a placeholder). Head-on is fine for a
      cup that starts on a fixed mark.
- [ ] What needs a no-go box: the camera and its mount, and anything else rigid in reach.
- [ ] The grasp orientation, chosen in the workspace setup. The default is the gripper
      level and 60 degrees to the left of straight ahead; at the default position it can
      turn about 26 degrees further towards straight ahead, or point straight down.
- [ ] Cup start and place marks, and the exact success criterion.
- [ ] Does this unit report joint torque? (Assumed no.)
- [ ] Which GPU and how much memory, and how much RAM? The cup config caps the buffers
      at 14 GB.
- [x] Twist interface for the lite: exists (`twist_controller`), tool frame; not used.

## Hook points kept open for the planned changes

| Planned change | Where it goes | What is already there |
|---|---|---|
| 1. Imitation term on interventions | `SACAlgorithm._compute_loss_actor(batch)` in `lerobot/rl/algorithms/sac/sac_algorithm.py`; batches are dicts. Note upstream's final update calls `_prepare_forward_batch(..., include_complementary_info=False)` before the actor loss, so that flag has to change with it. | `is_intervention` in every sampled batch |
| 2. Separate demo / correction shares | `OnlineOfflineMixer.sample` / `get_iterator` (`lerobot/rl/data_sources/data_mixer.py`): the single place batches are built; and `learner.process_transitions` for routing. The learner wrapper already substitutes the mixer class. | `source` on every transition, in both buffers |
| 3. Lower value target before takeovers | `SACAlgorithm._compute_loss_critic(batch)`, `td_target` | `takeover_onset`, `episode_index`, `step_index` |
| 4. Model-generated critic batch share | critic batch assembly in `SACAlgorithm.update` | nothing extra needed |
