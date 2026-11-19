# Gen3 Lite HIL-SERL in Genesis: cylinder to shelf

A Genesis scene behind the same robot class the real arm uses (`Gen3LiteHil`: end-effector
deltas, limits, IK with frozen orientation, workspace guard, watchdog, staleness checks),
so the HIL-SERL learner and actor run unchanged. Added 2026-11-18.

## What is simulated

- **Arm:** genesis_pickaplace's Gen3 Lite URDF (`lerobot_robot_ros/assets/gen3_lite_2f_genesis.urdf`,
  meshes from the installed `kortex_description`). Joints 1–6 match the kortex URDF the robot
  class uses for kinematics. Gravity compensation, PD position control with
  genesis_pickaplace's gains. Base on the ground plane, which is the table (`plane_z = 0`).
- **Driver behaviour:** the simulator runs in its own thread in real time (dt 0.01 s, CPU
  backend). Each arm command is interpolated from the previous goal over 0.2 s, like the
  real `joint_trajectory_controller` goal. Joint state and camera frames are snapshots with
  ages, so the stale-observation and watchdog logic act as on the arm.
- **Gripper:** the robot class's `0.0 open .. 0.85 closed` maps onto the URDF's finger
  driver joint, 0.96 rad (fingertips 10.5 cm apart) .. −0.09 rad (closed), with the other
  three finger joints set through the URDF's mimic relations. On a 6 cm cylinder the
  gripper stalls near 0.48.
- **Task:** an orange cylinder (r 3 cm, h 10 cm, 85 g) starts at (0.32, 0.08) ± 1 cm; a blue
  shelf 8 cm tall is centred at (0.32, 0.24). Success: the cylinder resting upright on the
  shelf with the gripper open.
- **Camera:** one head-on camera at (0.95, 0.16, 0.35) looking at (0.30, 0.16, 0.10),
  rendered 128×128 at 20 Hz.
- **Grasp:** top-down. The reset pose `[0.323, -0.356, 1.508, 1.571, 1.278, -0.201]` puts the
  tool at (0.30, 0.15, 0.20) pointing straight down; that orientation is frozen for the task.

## What the policy sees

Only what the real robot reports: six joints, the gripper, the tool position (the 10-number
state) and the camera image. The cylinder and shelf positions are never in the observation.

## The demonstrator

`--teleop.type=scripted_pick_place`: a waypoint planner over the simulator's privileged
state (cylinder pose, fingertip positions): approach above the cylinder, descend, close,
lift, carry over the shelf, lower, open. It chooses its phase from the world state on every
step, so it can take over at any point of an episode, and it retries a missed grasp. Every
x, y, z command gets Gaussian noise (`noise_std`, 0.3 action units = 3 mm), which the
planner corrects on the next step; the gripper command is noiseless. Takeover windows work
like `scripted_ee` (`takeovers`, `max_takeover_episodes`).

## Running it

```bash
LEROBOT_ENV=lerobot-ros-genesis . gen3_lite_env.sh     # genesis-world 1.4.7 lives in this env
python -m hilserl_gen3_lite.record  --config_path=config/hilserl/genesis_cup_env.json --run.confirm_motion=true
python -m hilserl_gen3_lite.learner --config_path=config/hilserl/genesis_cup_train.json --run.dir=runs/genesis_cup_s0 &
python -m hilserl_gen3_lite.actor   --config_path=config/hilserl/genesis_cup_train.json --run.dir=runs/genesis_cup_s0 \
    --run.confirm_motion=true --policy.online_steps=40000 --run.eval_interval_steps=5000 --run.eval_episodes=10 --run.announce=false
python -m hilserl_gen3_lite.summarize_run runs/genesis_cup_s0
```

No ROS is involved; `--run.confirm_motion=true` is still required by the entry points.
Before starting a learner, check nothing else holds its port (`ss -ltnp | grep 50051`): a
leftover learner shares the port silently and the actor connects to it.

The configs are the real-arm cup configs (`*_gen3_lite_cup.json`) with the robot, camera,
operator, reset pose, tool box (x 0.24–0.40, y 0.04–0.28, z 0.05–0.24 m), 15 s episodes and
1 s resets changed. Network, SAC settings, batch mix and buffer sizes are untouched.

## Known differences from the arm

- Lag is smaller: the PD arm settles faster than the real arm's ~2 cm at 0.1 m/s.
- Contact is Genesis's: grasps are firmer and more repeatable than on a real plastic cup.
- The scripted operator does not know about evaluation blocks; during the takeover phase it
  also takes over in evaluation episodes, which then count as failures.

## Results: first run, 2026-11-18 (`runs/genesis_cup_s0`)

RTX 3080 (learner), Genesis 1.4.7 on CPU (actor), 20 planner demos, takeovers on steps
10–109 of the first 100 episodes and none after, 40 000 env steps in 82 minutes.

| Env steps | What happened |
|---|---|
| 0 – 13k | Takeover phase: the planner finished the task in most episodes (86 of 105 succeeded). |
| 13k – 27k | Hands off: 0 successes; the policy reached down towards the cylinder, often within 1–2 cm of the grasp point, but closed the gripper on nothing and pressed against the tool box walls. |
| 27k – 40k | Policy-only successes appear: evaluation 7/10 at 30k and 8/10 at 35k (no takeovers); 21 of the last 30 training episodes succeeded; mean length fell to ~85 steps. |

Rise times: 50 % evaluation success at 30 004 env steps, 90 % not reached. 74 policy-only
successes in total. Counters at exit: guard blocks 0, stale stops 0, IK rejections 152
(the large-joint-jump check, 0.21–0.27 rad for a 1 cm step: a near-singular wrist region
inside the box for this top-down orientation), box-clipped commands 10 265 (the policy
pushes against the box walls), one watchdog hold per episode end.

The evaluation blocks at 5k and 10k read 0 partly because the scripted operator took over
in them (see above). One seed only; not tuned.

Full results, including a 12-episode review of the final checkpoint: [GEN3_LITE_GENESIS_RESULTS.md](GEN3_LITE_GENESIS_RESULTS.md).
