# Running the Kinova Gen3 Lite on ROS 2 Lyrical Luth (Ubuntu 26.04)

This document describes how this project was brought up on **ROS 2 Lyrical Luth /
Ubuntu 26.04** against a **Kinova Gen3 Lite** — in both Gazebo simulation and on
the physical arm — what had to change from the upstream (Jazzy) setup, **why**,
and **how** each piece was made to work.

The upstream `README.md` targets ROS 2 **Jazzy** + Python **3.12** + MoveIt. None
of that combination is usable on Lyrical as-is; the sections below are the delta.

> **TL;DR** — The arm runs. Joint state, joint-trajectory control, the twist
> (Cartesian-velocity) controller, and the gripper all work on real hardware, and
> the project's own lerobot backend (`BackendRobot`) is verified end-to-end against
> the physical arm (§7.4), reliably across repeated runs. The cost was a
> Python-3.14 conda environment and seven patches (env, gripper action type,
> controller type, spawner `--param-file`, MoveIt lazy-import, joint-trajectory
> action interface, and a draccus/Py3.14 config-loading shim), plus a one-line URDF
> gripper-limit loosening. MoveIt is **not available on Lyrical**, so the
> MoveIt-Servo / `CARTESIAN_VELOCITY` path is unavailable — but the Gen3 Lite
> pipeline does not need it. The offline learning half (record → DP train →
> checkpoint → inference) is validated in software on this box (§9); real *training*
> needs a GPU machine (~2.8 s/step on CPU), while collection and deployment run here.

---

## 1. The Problem: why Jazzy instructions don't work on Lyrical

Lyrical Luth (12th ROS 2 release, May 2026) ships on Ubuntu 26.04, whose **system
Python is 3.14**. Every apt ROS binary — including `rclpy` — is built **only for
cpython-3.14**.

This project must import `rclpy` **and** `torch`/`lerobot` **in the same
interpreter** (the backend is a single Python process talking to both ROS and the
policy). That forces a hard choice:

- The pinned stack (`torch 2.7.1`, `lerobot 0.4.3`, Python 3.12) **cannot run on
  3.14** — `torch < 2.8` has no cp314 wheels. Full stop.
- Cross-distro middleware (RoboStack Jazzy/Kilted at Python 3.12 bridged to the
  Lyrical robot over DDS) is fragile, and `control_msgs` actions — i.e. the
  gripper — are the documented cross-distro breakage case.

**Decision: match the environment to Lyrical's Python 3.14** and load the system
`rclpy` natively. A conda-forge Python 3.14 is ABI-compatible with the system
cpython-3.14 `.so` files, so once `/opt/ros/lyrical/setup.bash` is sourced the
system `rclpy` imports cleanly alongside a modern `torch`.

---

## 2. What actually runs

| Layer | Component | Notes |
|-------|-----------|-------|
| OS | Ubuntu 26.04 | stock |
| Middleware | ROS 2 **Lyrical Luth**, system **Python 3.14**, `rmw_fastrtps_cpp` | stock |
| Kinova SDK | `kortex_api` **2.8.0** | **precompiled** Kinova binary (`linux_x86-64_gcc_5.4`), downloaded from Kinova artifactory — links fine on 26.04 via libstdc++ ABI backward-compat |
| ROS driver | `kortex_driver`, `kortex_bringup`, `kortex_description` | **built from source** in `~/workspace/ros2_kortex_ws` (no apt binaries exist for Lyrical) |
| Control | `joint_state_broadcaster`, `joint_trajectory_controller`, `twist_controller` (picknik), `gen3_lite_2f_gripper_controller`, `fault_controller` | all activate on real hardware |
| Python env | conda **`lerobot-ros-314`** (Py 3.14): `rclpy` + `torch 2.10.0` + `lerobot 0.5.0` + `pinocchio` + `numpy 2.3.5` | one process; **CPU only** (no GPU on this box) |
| App | `lerobot_backends/ros2` (`ROS2Interface`) | patched (§4); sim-validated; **verified end-to-end on real hardware** (§7.4). Offline DP train/deploy pipeline validated in software (§9) |

### On the "precompiled vs. source" question
- **Kinova's SDK is precompiled and works** — `kortex_api` is a thin CMake shim
  that downloads a prebuilt Kinova binary. We never compile the SDK.
- **The ROS 2 wrapper has no binary release** for Lyrical (`ros2_kortex` is a
  source-first project; `apt-cache search ros-lyrical | grep kortex` is empty), so
  the driver/bringup/description packages — plus a vendored copy of the
  surrounding control/sim stack (`ros2_control`, `ros2_controllers`,
  `gz_ros2_control`, `ros_gz`, `picknik_controllers`, `ros2_robotiq_gripper`,
  `control_msgs`, `serial`) — are built from source in the workspace.

---

## 3. Environment setup (`lerobot-ros-314`)

A pinned, reproducible spec of the working env lives at
[`environment-lyrical-314.yml`](../environment-lyrical-314.yml) (128 PyPI packages,
generated from the live env). **It deliberately excludes all ROS packages** —
`rclpy`, `tf2_*`, `control_msgs`, etc. are provided by *sourcing* system Lyrical,
not by pip, and are not on PyPI. Recreate with:

```bash
conda env create -f environment-lyrical-314.yml
conda activate lerobot-ros-314
pip install -e lerobot_robot_ros lerobot_teleoperator_devices   # local packages
```

From scratch (how it was originally built), against conda-forge only (avoids the
Anaconda commercial-ToS default channels):

```bash
conda create -y -n lerobot-ros-314 --override-channels -c conda-forge python=3.14
conda activate lerobot-ros-314
# ML stack with cp314 wheels
pip install --no-deps torch numpy==2.3.5 lerobot pin eigenpy coal ...
```

Key version facts (verified):

- `torch 2.10.0` (cp314; CPU — `torch.cuda.is_available()` is `False`),
  `lerobot 0.5.0`, `numpy 2.3.5`, `pinocchio` (cp314 wheel).
- `lerobot` upstream pins `numpy < 2.3` (stale); 3.14 needs `numpy >= 2.3.2`, so
  it is installed to tolerate `2.3.5`. All deep `lerobot` APIs the project uses
  still exist on 0.5.0.
- `pygame` has **no** cp314 wheel (affects only keyboard teleop, not core use).

### Activating for a session

```bash
conda activate lerobot-ros-314
source /opt/ros/lyrical/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

> The old Python-3.12 `lerobot-ros` env still exists as a fallback but **cannot
> import `rclpy`** on this machine (wrong cpython ABI). System `python3` is 3.13
> (miniconda base) and also cannot load Lyrical `rclpy` — always use the 3.14 env.

---

## 4. The patches (the "Lyrical tax")

Five changes were required. Nothing here works stock.

### 4.1 Gripper action type — `GripperCommand` → `ParallelGripperCommand`

**Why.** Lyrical removed `position_controllers/GripperActionController`. The Gen3
Lite gripper is now driven by
`parallel_gripper_action_controller/GripperActionController`, whose action is
**`control_msgs/action/ParallelGripperCommand`**. That goal is a
`sensor_msgs/JointState` (`command.name[]`, `command.position[]`) — a completely
different layout from the classic `GripperCommand` (scalar `command.position`).
The backend previously sent `GripperCommand`, so the gripper server never matched.

**How.**
- `lerobot_backends/ros2/config.py` — new enum member
  `GripperActionType.PARALLEL_ACTION`; `KinovaGen3LiteConfig` now uses it.
- `lerobot_backends/ros2/ros_interface.py` — imports `ParallelGripperCommand`;
  in `connect()` the parallel path builds a `ParallelGripperCommand.Goal` and sets
  `command.name = [gripper_joint_name]` once; `send_gripper_command()` sets
  `command.position = [goal]` (a list, per `JointState`) instead of a scalar.
  The classic `GripperCommand` path is left intact for other robots.

Controller config (`kortex_description/.../gen3_lite/6dof/config/ros2_controllers.yaml`):

```yaml
gen3_lite_2f_gripper_controller:
  # Lyrical: position_controllers/GripperActionController was removed.
  type: parallel_gripper_action_controller/GripperActionController
```

Both the sim launch **and** the real launch resolve this same file, so the change
covers both.

### 4.2 Controller spawner `--param-file`

**Why.** On Lyrical, `ros2_control_node` (real) and `gz_ros2_control` (sim) no
longer forward per-controller parameter sections from the plugin `<parameters>`
YAML — only the `controller_manager:` section is read. Controllers came up with
empty `joints`/`joint` params and failed to configure.

**How.** Every controller spawner is given the controllers YAML explicitly:

```python
Node(package="controller_manager", executable="spawner",
     arguments=[controller, "-c", "/controller_manager",
                "--param-file", robot_controllers])
```

Patched in `kortex_sim_control.launch.py` (sim) and `kortex_control.launch.py`
(real — used by `gen3_lite.launch.py`). Modified `ros2_kortex` files are backed up
under `forked_cortex/`.

### 4.3 MoveIt-Servo lazy import

**Why.** `ros_interface.py` imported `MoveIt2Servo` (needs `moveit_msgs`) at module
top level. MoveIt is not installed on Lyrical (§5), so the whole backend failed to
import even though the Gen3 Lite path never uses Servo.

**How.** The import was moved inside the `CARTESIAN_VELOCITY` branch of
`connect()`. Only the Annin AR4 cartesian-velocity path pays the MoveIt cost; the
Gen3 Lite (`JOINT_TRAJECTORY`) backend imports cleanly without it.

### 4.4 Python 3.14 environment

See §3 — this is a patch in the sense that the documented Python-3.12 install is
not usable on Lyrical.

### 4.5 Joint trajectory via the ACTION interface, not the command topic

**Why.** The backend published `JointTrajectory` messages to the JTC **command
topic** (`/joint_trajectory_controller/joint_trajectory`). On Lyrical this is
**unreliable** — one-shot trajectories are silently dropped (the joint doesn't
move, no error). It was intermittent: the same code moved the arm once and then
stopped, which made it look like a servoing-mode or joint-limit problem. The
tell: commanding the identical trajectory through the JTC **action**
(`follow_joint_trajectory`) returned `error_code=0 SUCCESSFUL` and moved the arm
every single time.

**How.** `ros_interface.py` gained a `FollowJointTrajectory` action-client path for
`JOINT_TRAJECTORY`, selected by `ROS2InterfaceConfig.use_trajectory_action`
(default **True**). The action name is derived from `arm_topic`'s controller
namespace (`…/follow_joint_trajectory`). Goals are sent fire-and-forget
(`send_goal_async`); a new goal preempts the previous, which is the right behavior
for streaming waypoints. The old topic publisher remains available via
`use_trajectory_action=False`. Verified: the end-to-end backend test moved
`joint_6` identically across **3/3** consecutive runs (previously it succeeded
once then silently stopped).

### 4.6 draccus config loading on Python 3.14 (`lerobot_py314_compat.py`)

**Why.** lerobot 0.5.0 loads every policy/train config through
`draccus.parse(cls, config_file, args=[])`, which builds an `argparse` parser even
when reading purely from a file. draccus 0.8.0 passes a field's *type annotation*
straight to `argparse.add_argument(type=...)`. For `Optional[X]` / `X | None`
fields (e.g. `root: str | None`, `input_features: Dict[str, PolicyFeature] | None`)
that annotation is a **non-callable union object**. On Python ≤ 3.13 `argparse`
only invokes `type` when it actually parses a string, so with `args=[]` it was
never called and this silently worked. **Python 3.14's `add_argument` validates
callability eagerly**, so it raises `TypeError: X | None is not callable` at
parser-construction time — breaking both the `lerobot-train` CLI *and*
`Policy.from_pretrained` (i.e. loading a checkpoint for eval/deployment).

**How.** [`lerobot_py314_compat.py`](../lerobot_py314_compat.py) wraps
`draccus.utils.canonicalize_union` so a non-callable (Optional/union) result is
replaced by a callable — the inner type for a simple `Optional[X]`, or a harmless
`str` placeholder otherwise. draccus only *invokes* this `type` when parsing a CLI
*string* argument; the file-loading path (`args=[]`) never calls it, so the
placeholder preserves decoding behavior exactly. Import the shim **before** any
lerobot config load; the real deploy script `eval_kinova_reach.py` already does.

This fixes **config-file loading** (checkpoints → eval/deploy). The
`lerobot-train` **CLI** has a *second*, deeper draccus/3.14 bug in the union/choice
wrapper for polymorphic config fields (`env: EnvConfig | None`,
`policy: PreTrainedConfig | None` → `AttributeError: 'EnvConfig' has no attribute
'__args__'`) that the shim does not cover. The clean workaround is to **drive
training from Python**, not the CLI: build a `TrainPipelineConfig` in code and call
`train.__wrapped__(cfg)` (the `@parser.wrap()` decorator only adds CLI parsing; the
wrapped function is the actual trainer). See [`smoke_train.py`](../smoke_train.py).

---

## 5. What is currently impossible / blocked

**MoveIt is not available on Lyrical.** `moveit_msgs`, `moveit_servo`,
`moveit_ros_move_group`, and `moveit_core` are all absent — no binaries are
published for this brand-new distro, and a source build is a major effort. This
blocks:

- motion planning / collision-aware moves;
- `moveit_servo` base-frame Cartesian velocity (the Annin AR4 `CARTESIAN_VELOCITY`
  path).

**This does not block the Gen3 Lite pipeline**, which uses `JOINT_TRAJECTORY` +
the gripper action, and gets Cartesian motion directly from the Kinova
`twist_controller`.

Other constraints:

- **No GPU** — `torch` is CPU-only; inference is fine but DP training is ~2.8 s/step
  (§9), so real training must run on a GPU box. Collection and deployment stay here.
- **Pinned deps are permanently impossible on 3.14** (`torch < 2.8` has no cp314
  wheels). `lerobot 0.5.0` + `torch 2.10` is the supported combination here.

---

## 6. Bring-up — Simulation

```bash
# gen3_lite 6-DOF + gripper in Gazebo
ros2 launch kortex_bringup kortex_sim_control.launch.py \
  use_sim_time:=true launch_rviz:=false robot_type:=gen3_lite dof:=6 gripper:=gen3_lite_2f
```

**Gotcha.** Fully kill stale `gz sim` / `robot_state_publisher` / `parameter_bridge`
/ `ros2_control_node` / `controller_manager` processes (and `rm /dev/shm/fast*`)
before relaunching — leftover nodes cause duplicate-node contention and spurious
controller load/activate failures. The GPU-less box also starts Gazebo slowly;
prefer headless (`-s`) and give spawners time.

---

## 7. Bring-up — Real hardware

Physical Gen3 Lite reachable at `192.168.1.10` (here via a USB-ethernet adapter on
`192.168.1.22/24`).

```bash
ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.1.10 launch_rviz:=false
```

The driver connects, streams `/joint_states` (joints 1–6 + `right_finger_bottom_joint`),
and brings up all controllers. **No motion happens on bring-up** — controllers come
up holding the current pose.

### 7.1 HOME THE ARM FIRST — the servoing-mode trap

`joint_trajectory_controller` (JTC) will **fail to activate** if any joint is
outside its URDF soft limit. We hit `joint_2` at `-2.667 rad` vs. URDF
`[-2.61, 2.36]` — 3° past the limit — and JTC deactivated on the spot.

This has a **non-obvious side effect on the gripper**. When JTC starts it switches
the arm to `LOW_LEVEL_SERVOING`; its failed deactivation does **not** switch back.
In `LOW_LEVEL_SERVOING`, the driver only transmits commands to the robot inside
`sendJointCommands()`, which runs **only while a joint controller is active**. So
with JTC dead, the gripper action is accepted, the position is written to
`base_command_`, but `base_cyclic_.Refresh()` is never called — **the gripper
command never reaches the robot** and the action returns `stalled: true,
reached_goal: false` with no motion.

**Fix: home the arm** (Kinova web app at `http://192.168.1.10`, or the controller)
so all joints are within limits, then relaunch. JTC then activates, the arm runs in
`LOW_LEVEL_SERVOING` with a joint controller live, and gripper commands transmit
every cycle.

> Diagnostic shortcut without homing: activating `twist_controller` switches the arm
> to `SINGLE_LEVEL_SERVOING`, where the gripper uses `base_.SendGripperCommand()`
> which **is** always transmitted (zero twist = no arm motion). Useful to prove the
> gripper hardware independently, but homing is the real fix.

### 7.2 Gripper test

```bash
# open (0.0) .. closed (0.85 rad); goal is a JointState
ros2 action send_goal /gen3_lite_2f_gripper_controller/gripper_cmd \
  control_msgs/action/ParallelGripperCommand \
  "{command: {name: [right_finger_bottom_joint], position: [0.5]}}"
```

**URDF gripper lower limit was a hair too tight (now fixed).** On a full open the
real gripper overshoots to ≈ `-0.009 rad`, below the original URDF limit `[0, 0.85]`.
The joint limiter caught this and **deactivated the gripper controller**. Fix: the
lower limit was loosened `0.0 → -0.02` in
`kortex_description/grippers/gen3_lite_2f/urdf/gen3_lite_2f_macro.xacro` (the active
`right_finger_bottom_joint` block; backed up to `forked_cortex/`). The install tree
is symlinked to source, so a driver relaunch picks it up with no rebuild.

### 7.3 Cartesian EEF motion — mind the TOOL frame

The Kinova twist interface is in the **TOOL frame**
(`CARTESIAN_REFERENCE_FRAME_TOOL`), **not** the base frame. A raw `linear.z` twist
moves along the gripper axis, not world-down. To move a true world-Z delta, run a
closed-loop servo: read `base_link → tool_frame` each cycle, compute the desired
**world** velocity (hold x/y, drive z), and rotate it into the tool frame before
publishing to `/twist_controller/commands` (`geometry_msgs/Twist`). This requires
switching `joint_trajectory_controller` → `twist_controller`
(`SINGLE_LEVEL_SERVOING`) for the move, then switching back.

A reference implementation (world-Z servo with safety aborts on lateral drift,
wrong-direction, and timeout) lives at
[`real_eef_move_z.py`](../real_eef_move_z.py). Run it inside the `lerobot-ros-314`
env with the driver up and the arm homed; it switches JTC → twist, moves down then
up, and always restores JTC on exit. Verified: EEF moved ±46 mm and returned to
within 0.2 mm of start, no drift.

### 7.4 Verified on real hardware

- Driver connects; `/joint_states` streaming.
- Gripper actuation via `ParallelGripperCommand`.
- Cartesian EEF ±0.05 m via the closed-loop twist servo.
- JTC activates and holds after homing.
- **Full lerobot backend round-trip** — `BackendRobot(KinovaGen3LiteConfig())`
  `connect()` → `get_observation()` → `send_action()`, driving both the gripper
  (partial-close → open) and a bounded `joint_6` trajectory (+0.10 rad, returned to
  within 0.3 mrad). This is the project's *own* application path, not the CLI, and
  it runs **reliably across repeated invocations** now that joints go through the
  `FollowJointTrajectory` action (§4.5) instead of the flaky command topic.
  Script: [`real_kinova_backend_test.py`](../real_kinova_backend_test.py). Observed:

  ```
  gripper 0.5 close -> 0.5003 ;  0.1 open -> 0.0999
  joint_6 +0.10 -> moved +0.0997 ;  return -> residual -0.0003 rad
  ```

### 7.5 Status / still untested on hardware

The project's backend path is now **verified end-to-end on the physical arm** (§7.4),
and the offline learning pipeline (record-format → DP train → checkpoint → inference)
is **verified in software on this box** (§9). What remains untested on hardware (not
blocked — just not yet run):

- cameras in the observation dict (none wired to this box yet);
- a full `record_*` data-collection / policy `eval_*` loop at control rate on the arm;
- multi-joint trajectories beyond the single-joint smoke test.

The one thing that is *impractical* rather than untested: **training a real policy on
this box** — CPU-only DP is ~2.8 s/step (§9), so a 100 k-step run is ~3 days. Real
training belongs on a GPU machine; collection and deployment stay here.

---

## 8. Safety notes for real-hardware work

- Always warn a human and have manual override / E-stop ready before any command
  that moves the arm.
- Keep Cartesian speeds small (the reference servo caps at 0.025 m/s) with hard
  aborts on lateral drift, wrong-direction motion, and timeout; always publish
  zero-twist on exit.
- Before relaunching the driver, stop the previous one so it releases robot control
  (otherwise homing from the web app fights the running session).

---

## 9. Offline policy pipeline — software smoke test (no arm, no ROS, no camera)

The real-hardware work above proves the **ROS ↔ arm** boundary. The *other* half of
the project — record a LeRobot dataset, train a Diffusion Policy, load the checkpoint
and run inference — had never been exercised on **this** stack (lerobot 0.5.0, Python
3.14, CPU-only torch). Because that half needs no robot, it is validated in pure
software here so real-hardware time isn't spent debugging library plumbing. There is
no point running the full loop end-to-end on the arm until we can actually train a
policy — which needs a GPU box — so this isolates and confirms everything *except* the
training compute.

Three scripts, run in order, all CPU-only and hardware-free:

| Step | Script | Validates |
|------|--------|-----------|
| 1. Synthesize dataset | [`smoke_synth_dataset.py`](../smoke_synth_dataset.py) | `LeRobotDataset.create/add_frame/save_episode` writes a valid **v3.0** dataset on lerobot 0.5.0 (state-only, `use_videos=False`) |
| 2. Train DP on CPU | [`smoke_train.py`](../smoke_train.py) | DiffusionPolicy trains on the **state + `environment_state`** (no-image) branch; checkpoint + processor artifacts are written; gives the CPU throughput number |
| 3. Inference round-trip | [`smoke_infer.py`](../smoke_infer.py) | `DiffusionPolicy.from_pretrained` + `PolicyProcessorPipeline` pre/post round-trip (the exact path `eval_kinova_reach.py` uses) yields a finite action |

The synthetic dataset deliberately matches the **real Gen3 Lite feature signature**
(7-dim `observation.state` = 6 joints + gripper, 7-dim `observation.environment_state`
target, 7-dim `action`) so the schema and trained-policy config transfer directly to a
real `record_*` → train run later; only the *data* is synthetic (a trivial reach).

```bash
conda activate lerobot-ros-314     # Python 3.14
python smoke_synth_dataset.py       # -> data/lerobot/kinova_gen3_lite_smoke (900 frames, v3.0)
python smoke_train.py 200 16        # 200 steps, batch 16, CPU -> outputs/train/gen3_lite_smoke
python smoke_infer.py               # loads checkpoint, one select_action, prints a 7-dim action
```

### What it established
- **Dataset format works** on 0.5.0 — writes and reloads a v3.0 dataset cleanly.
- **State-only DP is trainable** — DiffusionPolicy accepts `state + environment_state`
  with no camera (its `validate_features` requires "at least one image *or*
  `environment_state`"). Loss descended (1.20 → 1.10 in 20 steps) and a checkpoint was
  written. So a **camera is needed for a *meaningful* task, not to close the loop.**
- **Deployment round-trip works** — checkpoint → processor → `select_action` → action,
  the same path the real eval uses (once the Py3.14 draccus shim from §4.6 is imported).
- **CPU training throughput: ~2.8 s/step** at batch 16 for the default DP (≈249 M
  params). That is the go/no-go number: **100 k steps ≈ 78 h (~3 days)** on this box →
  **real training must run on a GPU machine.** Collection and deployment stay here.

### Gotchas surfaced (all fixed or documented)
- `Policy.from_pretrained` and the `lerobot-train` CLI both hit the **Python-3.14
  draccus** bug — see §4.6. The shim fixes checkpoint loading; training is driven from
  Python (`train.__wrapped__(cfg)`) rather than the CLI.
- DP windowing (`horizon=16`, `n_obs_steps=2`, `drop_n_last_frames=7`) means episodes
  must be well over ~23 frames or a dataset yields **zero** training samples; the
  synthetic episodes are 60 frames for margin.
