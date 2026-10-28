# Gen3 Lite: demos → Diffusion Policy → DAgger corrections (lerobot 0.6.1, Jazzy)

Everything runs through lerobot's own CLIs (`lerobot-record`, `lerobot-train`,
`lerobot-rollout --strategy.type=dagger`, `lerobot-edit-dataset`). This repo supplies
three plugins that lerobot auto-loads once they're installed:

| `--*.type=` | Package | What it is |
|---|---|---|
| `--robot.type=gen3_lite_ros` | `lerobot_robot_ros` | `BackendRobot` + `KinovaGen3LiteConfig`. Actions: 6 absolute joint positions (rad) + `gripper.pos` (0.0 open .. 0.85 closed). Per-step arm cap `max_relative_target=0.1` rad. |
| `--teleop.type=spacemouse_joint` | `lerobot_teleoperator_devices` | SpaceMouse → damped-least-squares IK (pinocchio) → absolute joint targets, same keys as the robot. Reads `/joint_states` itself; re-syncs pose and gripper after a 0.5 s gap in calls (= a handover). `--teleop.device=scripted` replays a slow circle with no device attached. |
| `--policy.type=state_diffusion` | `lerobot_policy_state_diffusion` | lerobot's Diffusion Policy with its "needs an image or environment state" check removed, so it trains on joint state alone. |

## Setup (once)

```bash
conda create -y -n lerobot-ros-06 python=3.12
conda install -y -n lerobot-ros-06 -c conda-forge libstdcxx-ng
. gen3_lite_env.sh   # ROS Jazzy + ros2_kortex_ws + this env, clean of other workspaces
pip install "lerobot[dataset,training,diffusion]==0.6.1" pin pyspacemouse pynput
pip install -e . -e lerobot_robot_ros -e lerobot_teleoperator_devices -e lerobot_policy_state_diffusion
```

`pip install` pulls the newest CUDA torch (2.11.0+cu130), which needs NVIDIA driver 580 or newer
loaded. The plugin packages must keep `lerobot_` in their `name =` with **underscores**:
lerobot discovers plugins by distribution name (`lerobot_robot_*`, `lerobot_teleoperator_*`,
`lerobot_policy_*`), and the previous `lerobot-robot-ros` spelling was never found.

`. gen3_lite_env.sh` in **every** terminal (driver, record, train, rollout).

## Real robot, joint state only

**A state-only policy cannot see the cup.** It can only learn a motion that does not
depend on where things are: put the cup in the same marked spot for every demo and
every rollout. Moving the cup needs cameras (`--robot.cameras=...`, then plain
`--policy.type=diffusion`).

Home the arm first (Kinova web app). A joint outside its URDF limit keeps
`joint_trajectory_controller` from activating, which also silently stalls the gripper
(`docs/LYRICAL_KINOVA.md` §7.1).

```bash
# terminal 1: driver
ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.1.10 launch_rviz:=false

# terminal 2: 1) demos. Right SpaceMouse button toggles the gripper. Keyboard: → next
# episode, ← re-record, Esc stop. The reset period is teleoperated too: drive back to start.
lerobot-record --robot.type=gen3_lite_ros --teleop.type=spacemouse_joint \
  --dataset.repo_id=local/gen3_lite_cup --dataset.root=data/gen3_lite_cup \
  --dataset.single_task="pick up the cup" --dataset.fps=10 \
  --dataset.num_episodes=30 --dataset.episode_time_s=40 --dataset.reset_time_s=20 \
  --dataset.push_to_hub=false

# 2) train (GPU)
lerobot-train --dataset.repo_id=local/gen3_lite_cup --dataset.root=data/gen3_lite_cup \
  --policy.type=state_diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 \
  --steps=20000 --batch_size=64 --save_freq=10000 --wandb.enable=false \
  --output_dir=outputs/train/gen3_lite_cup

# 3) DAgger round: the policy runs; Space pauses it (arm holds), Tab starts a correction
# (arm slides to the SpaceMouse's pose = where it already is), Tab again saves that
# correction as an episode, Space resumes the policy, Esc stops.
lerobot-rollout --robot.type=gen3_lite_ros --teleop.type=spacemouse_joint \
  --policy.path=outputs/train/gen3_lite_cup/checkpoints/last/pretrained_model \
  --strategy.type=dagger --strategy.num_episodes=10 \
  --dataset.repo_id=local/rollout_gen3_lite_cup_r1 --dataset.root=data/rollout_gen3_lite_cup_r1 \
  --dataset.single_task="pick up the cup" --dataset.fps=10 --dataset.push_to_hub=false \
  --fps=10

# 4) fold the corrections in (merge needs identical columns; every correction frame has
# intervention=True, so drop it) and retrain on the merged set
lerobot-edit-dataset --repo_id=local/rollout_gen3_lite_cup_r1 --root=data/rollout_gen3_lite_cup_r1 \
  --new_repo_id=local/rollout_gen3_lite_cup_r1_noflag --new_root=data/rollout_gen3_lite_cup_r1_noflag \
  --operation.type=remove_feature --operation.feature_names="['intervention']"
lerobot-edit-dataset --new_repo_id=local/gen3_lite_cup_r1 --new_root=data/gen3_lite_cup_r1 \
  --operation.type=merge --operation.repo_ids="['local/gen3_lite_cup','local/rollout_gen3_lite_cup_r1_noflag']" \
  --operation.roots="['data/gen3_lite_cup','data/rollout_gen3_lite_cup_r1_noflag']"
```

Notes:
- **Keyboard on Wayland** (the Ubuntu 24.04 default): lerobot won't use a global key
  listener there and falls back to reading keys from the terminal, so keep the rollout's
  terminal focused. Under X11 any window works.
- `lerobot-rollout` stamps a date onto the dataset name (`..._r1_20261028_133600`) and
  requires names that start with `rollout_`; check the actual `--dataset.root` folder
  before merging.
- The SpaceMouse needs read access to its `/dev/hidraw*` (udev rule) for `pyspacemouse`.
- `action_time_from_start_sec` is 0.2 s (`KinovaGen3LiteConfig`): each 10 Hz command is a
  FollowJointTrajectory goal due 0.2 s out, preempted by the next. Recording and rollout
  share it, so the policy learns the same tracking it is deployed with.

## Safety behaviour (shared backend, all lerobot_backends robots)

- **Step cap:** `max_relative_target` caps each arm joint's |goal − measured| per step
  (`BackendRobot.send_action`), for policy and teleop alike. Reads joints only, not cameras.
  The gripper is not capped. Before this change the cap compared `gripper.pos` against
  a missing key (0.0) and would have pinned the gripper near open. The teleop applies
  the same 0.1 rad cap itself, so recorded actions equal what was sent.
- **Stale state:** `get_observation()` raises once the newest `/joint_states` is older
  than `max_state_age_sec` (1.0 s). The joint-state callback no longer raises, which
  used to kill the executor thread silently and freeze observations.
- **Cameras:** a failed read repeats the last good frame instead of dropping the key,
  which made `dataset.add_frame` reject the whole episode.
- **Gripper:** the goal is re-sent only when it changes (or every 1 s), not every tick.

## Workspace guard (`gen3_lite_ros`, on by default)

`lerobot_backends/ros2/workspace_guard.py`, run inside `Gen3LiteRos.send_action` on the
final (step-capped, joint-limit-clamped) target, so it covers SpaceMouse, policy and
reset commands alike. Uses the URDF collision meshes (pinocchio + hpp-fcl), about 21 ms
per check.
- **Table plane** at `plane_z=0` (bottom face of the arm's base; the real table is lower
  by the mount's height, which no URDF records), every moving link incl. fingers at least
  `plane_margin=0.015` m above it. The base link is exempt.
- **Tool box** `workspace_min=[0.13, 0.01, 0.03]` .. `workspace_max=[0.45, 0.31, 0.50]`
  (base frame): about +-15 cm around the cup-grasp pose (tool at 0.282, 0.162, 0.065),
  widened in x/z so Kinova's home (tool at 0.439, 0.193, 0.443; 0.453 in Gazebo) is inside.
- **Self-collision** with the MoveIt SRDF's disabled pairs, `self_collision_margin=0.005`.
- A violating command is replaced by the last accepted arm target (holding the measured
  pose instead would let gravity sag compound). Moves that reduce an existing violation
  are allowed, so the arm can always be driven out. `lerobot-record` still saves the
  teleop's action, not the held one: derive actions from measured states where that matters.
- Checked on the real arm (no commands sent): the taught grasp pose and all 448 frames of
  a live SpaceMouse session pass; a 0.05 rad step toward the table from the grasp pose is
  blocked (fingertip would reach the plane); synthetic reach-down / fold / swing-out poses
  are all flagged.

**Homing** (`home_on_connect=True`): on connect the robot drives to `home_position`
(Kinova's home, measured on the real arm: `[0, -0.282, 1.315, -0.003, -1.047, 0]`, gripper
horizontal pointing +x) at `home_speed=0.3` rad/s, every waypoint through the guard, gripper
open. If the guard blocks the straight joint path it stops with an error instead of detouring.
Gazebo (2026-10-28, sim spawned base 5 cm above ground): from the all-zero pose (tool at
z 1.0, outside the box) homed in 4.4 s with nothing blocked; from the grasp pose, a joint
path toward the table stopped at the step that would put the right fingertip 0.6 cm above
the plane; a joint_1 swing stopped with the tool at x 0.139 against the 0.13 wall.

## Smoke test in Gazebo (2026-10-28, CPU only)

GPU driver not loaded at the time (kernel 7.0.0-38 had no NVIDIA module), so Gazebo ran
headless (`-s`, a scratch copy of `kortex_sim_control.launch.py`) and training ran on CPU.
`robot_type:=gen3_lite dof:=6 gripper:=gen3_lite_2f use_sim_time:=true`, plus
`--robot.use_sim_time=true --teleop.use_sim_time=true --teleop.device=scripted`.

| Step | Result |
|---|---|
| `lerobot-record`, 5 × 12 s at 10 Hz | 599 frames, max \|action − state\| exactly 0.1 rad. An earlier 3-episode run, before the teleop applied the cap itself: arm moved up to 1.4 rad per joint per episode, next state within 0.011–0.022 rad (mean) of each action, and 1 of 360 frames had a 0.57 rad action (wrist singularity at the all-zero start pose: joints 4/6 aligned). |
| `lerobot-train --policy.type=state_diffusion`, `down_dims=[64,128,256]`, 300 steps, batch 16 | 27 s on CPU, loss 1.10 → 0.58. Checkpoint loads in rollout. |
| `lerobot-rollout --strategy.type=dagger`, 2 corrections (keys via `xdotool` on Xvfb) | Pause → correction → save → resume worked twice. Teleop re-synced at each handover. 2 episodes × 50 frames, all `intervention=True`, max \|action − state\| 0.072 rad, gripper stayed open (matched the sim's open gripper). Arm returned to its start pose at exit. |
| merge (5 demos + 2 corrections) → retrain 100 steps | 7 episodes, trains. |

Seen in the smoke test, expected on the real arm too:
- The barely-trained policy hit the step cap constantly (177 clamp warnings in ~40 s).
- The autonomous loop ran at 4–6 Hz instead of 10: Diffusion Policy inference on CPU, with
  Gazebo competing for cores. Correction phases held 10 Hz. Use the GPU.
- The handover tick blocks for 1 s while lerobot slides the arm to the teleop target.
