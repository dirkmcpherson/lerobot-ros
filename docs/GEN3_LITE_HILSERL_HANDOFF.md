# Handoff: first real-arm session, pick and place a plastic cup

Written 2026-11-18 for the agent working with the operator at the robot machine. Read
this first, then [GEN3_LITE_HILSERL.md](GEN3_LITE_HILSERL.md) for the design, the limits,
the log formats and what has been verified where.

## The plan

The operator has decided to skip a simulation pass on the robot machine and learn
directly on the arm:

- **Task:** pick up a plastic cup and place it on a marked spot.
- **Observations:** one camera, head-on to the arm, plus proprioception (six joint
  angles, the gripper joint, the tool position).
- **Actions:** tool x, y, z deltas and the gripper (close / stay / open). Roll, pitch and
  yaw of the tool are frozen for the whole task, at an orientation chosen during setup.
- **Protection:** a tool box the commands are clipped to, plus no-go boxes around
  anything rigid in reach (the camera, fixtures), taught interactively in step 3.
- **Reward:** the operator's success key on the gamepad.
- **Learner:** stock LeRobot 0.6.1 HIL-SERL, unchanged.

The goal of the session is one complete baseline run with its curves and rise-time
numbers, and an honest account of everything that went wrong on the way.

## Who does what

**The operator runs every command that moves the arm, with a hand on the emergency
stop. You never do.** You prepare, check, edit configs, run what does not move the arm,
watch logs and analyse results.

| Moves the arm (operator only) | Does not move the arm (you may run) |
|---|---|
| `hilserl_gen3_lite.smoke_test` | `python -m pytest` |
| `hilserl_gen3_lite.workspace` | anything with a `dryrun_*` config (fake arm) |
| `hilserl_gen3_lite.record` | `hilserl_gen3_lite.learner` (it only talks to the actor) |
| `hilserl_gen3_lite.actor` | `hilserl_gen3_lite.summarize_run`, `lerobot-find-cameras`, `lerobot.rl.crop_dataset_roi` |
| | reading topics, e.g. `ros2 topic hz /joint_states` |

The four commands on the left refuse to start on the real arm without
`--run.confirm_motion=true` and print their limits first. Hand the operator the exact
command; do not pass that flag yourself. Launching the kortex driver does not move the
arm, but leave that to the operator too. Other rules:

- Do not change the SAC losses, the batch sampling ratios, the network architecture, the
  lerobot version, controller gains or Kinova's protection zones.
- Do not loosen the default limits in `Gen3LiteHilConfig`. Task-specific bounds go in
  the JSON configs.
- If a log suggests the arm went past a limit (the tool outside `ee_bounds` by more than
  a couple of centimetres of lag, or a commanded step longer than 1 cm), stop and tell
  the operator before anything else.
- A fix to the adapter or the glue comes with a test. Do not commit or push unless asked.

## Where the system stands

| Verified | Where |
|---|---|
| 92 hardware-free tests | laptop |
| Smoke test, 16 targeted checks, three full learner + actor runs against the real `joint_trajectory_controller` and gripper controller (state-only reach) | Gazebo in a container, laptop, CPU |
| The gripper path end to end: 4-number actions, the discrete gripper head, the toggle penalty, a scripted pick and place between two points | fake arm with images (24 episodes) and Gazebo state-only (28 of 28 episodes), laptop, CPU |
| Frozen orientation (0.1 degrees of deviation while moving), a no-go box (arm stops with every link outside the margin), and the interactive workspace tool end to end | Gazebo, laptop, CPU |
| The policy and SAC build from the cup config | laptop, CPU |

Never exercised, so expect the first surprises here:

| Never exercised | Why it matters |
|---|---|
| The physical arm with this code | Gravity, friction and real tracking lag. In Gazebo lag was about 2 cm at 0.1 m/s and 1 mm after settling. |
| A real camera | Frame rate, latency, exposure; the 200 ms stale-frame limit. |
| The gamepad in a hand | Button numbers differ between controllers; see step 3. |
| CUDA, and `torch.compile` in the replay buffer | Every run so far was on CPU, on a machine with no C++ compiler. |
| Images against real controllers | Images have only run with the fake arm. One image bug (demo images 255 times too bright) was found late; it is fixed and tested. |
| Grasping a physical object | Gazebo had no cup. The scripted "pick and place" closed and opened the gripper at two points in the air. |
| Learning this task | No run has learned a pick and place; the Gazebo gripper run was 1500 steps with the scripted operator driving nearly all of them. |

## What changed for this task

The baseline so far was a reach with the gripper unused. Pick and place switches the
gripper on, which needed one real fix:

- **Gripper penalty.** lerobot's `GripperPenaltyProcessorStep` reads the last action
  number as a gripper joint target; here it is the command itself (0 close, 1 stay,
  2 open). Used as-is it would have penalised every "stay" step taken with the gripper
  open. `hilserl_gen3_lite/env.py` has its own step: the environment reports when a
  command changes what the gripper is asked to do, and that step gets `gripper_penalty`
  (-0.02). Holding the close button while the fingers travel is one toggle, not ten (in
  Gazebo the fingers took 1.4 s to close; each demo had exactly two penalised steps).
  The penalty goes where upstream puts it: `complementary_info["discrete_penalty"]`,
  added to the reward of the discrete gripper critic only.
- **Configs.** `config/hilserl/env_gen3_lite_cup.json` (workspace setup, record) and
  `train_gen3_lite_cup.json` (learner, actor): gripper on in the robot, the teleop and the
  processor; `num_discrete_actions: 3`; 20 s episodes; 8 s between episodes to put the cup
  back; 20 demos; replay buffers capped at 50 000 online and 20 000 offline transitions
  (about 200 kB each with one 128x128 image, so up to 14 GB of RAM); a checkpoint every
  20 000 optimisation steps (each one writes both buffers out as video datasets, which
  stalls the learner for a while).
- **Frozen orientation.** Every command holds one tool orientation: the start pose's.
  It used to be whatever the tool had when control resumed; it is now the same exact
  orientation in every episode, and a step that would turn the tool more than 0.03 rad
  is not sent.
- **No-go boxes.** `env.robot.no_go_boxes`: axis-aligned boxes in the base frame that no
  moving link, fingers included, may come within 2 cm of (`no_go_margin`). They are
  checked by the workspace guard with the collision meshes, next to the table plane and
  self-collision; a command that would violate one is replaced by a hold.
- **`hilserl_gen3_lite.workspace`**: the interactive setup for all of the above (step 3).
  The operator drives the arm, teaches the tool box by sweeping it, the no-go boxes by
  marking points, turns the gripper to the orientation the grasp needs, tries the walls
  out live, and saves into both config files.
- **`dryrun_cup_*.json`**: the same pipeline on the fake arm with a scripted operator.

lerobot's convention, which the configs follow: the `action` feature has shape `[3]` (the
continuous x, y, z), the gripper is a separate discrete head, and stored actions carry
its index as a fourth number.

## The session

### 0. Before the arm moves (you)

```bash
. gen3_lite_env.sh
pip install "lerobot[hilserl,dataset,training]==0.6.1" pin pytest
pip install -e . -e lerobot_robot_ros -e lerobot_teleoperator_devices
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import rclpy, torch, lerobot; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"
python -m pytest
free -g
```

Then rehearse the cup pipeline on the fake arm, on the GPU, without
`TORCH_COMPILE_DISABLE` (this is the first time CUDA and `torch.compile` run):

```bash
python -m hilserl_gen3_lite.record  --config_path=config/hilserl/dryrun_cup_env.json
python -m hilserl_gen3_lite.learner --config_path=config/hilserl/dryrun_cup_train.json \
    --policy.device=cuda --run.dir=runs/dryrun_cup_gpu &
python -m hilserl_gen3_lite.actor   --config_path=config/hilserl/dryrun_cup_train.json \
    --policy.device=cuda --run.dir=runs/dryrun_cup_gpu --run.eval_enabled=false --run.announce=false
python -m hilserl_gen3_lite.summarize_run runs/dryrun_cup_gpu
```

The actor exits by itself after 600 steps; stop the learner with Ctrl+C. Pass: tests
green, CUDA available, `runs/dryrun_cup_gpu/train.csv` has rows with the
`loss_discrete_critic` column filled in, and the training episodes succeed (the scripted
operator drives most of each one). If the machine is Ubuntu 26.04 / Lyrical rather than
Jazzy, or the kortex workspace is missing, stop and report: this branch assumes the setup
in `docs/GEN3_LITE_DAGGER.md`.

### 1. Camera (you, with the operator placing it)

```bash
lerobot-find-cameras opencv
```

Put the device and resolution under `env.robot.cameras.front` in **both** cup configs.
The name `front` is already wired through the features and the policy; keep it. The
entry there now is a placeholder (index 0, 640x480, 30 fps).

A head-on camera is fine for this task. The policy gets the arm's own position in all
three axes from proprioception, so it does not need the image to know how far it has
reached. What the image has to supply is what proprioception cannot: where the cup is
and whether it is in the gripper. With the cup starting on a fixed mark, a head-on view
shows its left-right and up-down position well, and the remaining uncertainty (a
centimetre or so of depth each time it is put back) lies along the camera axis. Keep in
mind what that view cannot do, so nobody is surprised later:

- It cannot tell the policy how far away the cup is if the cup's start position is
  varied in depth; apparent size is a weak cue at 128x128. Vary the cup position only
  sideways, or add a second view, when the task gets there.
- If the gripper approaches straight along the camera axis, the cup can hide the
  fingers from the camera just before the grasp.

Practical points: fix exposure, white balance and focus if the camera allows it; do not
move the camera, the lights or the table after the demos are recorded; keep the
operator's hands out of frame or always in the same place. The camera and its mount are
rigid objects near the arm: they get a no-go box in step 3.

### 2. Smoke test (operator)

Driver up, arm homed, table clear.

```bash
python -m hilserl_gen3_lite.smoke_test --robot.type=gen3_lite_hil --run.confirm_motion=true
```

3 cm out and back on each axis around a pose with the tool at (0.30, 0.16, 0.22) m, a
timing check and a watchdog check. Keep the whole output. For comparison, in Gazebo:
loop period 100.2 ms with 0.7 ms worst deviation, lag 18–21 mm on the third step of a
leg, 0.3–1.8 mm error after settling. A `FAIL` on the 8 mm settling tolerance on the real
arm is information about the arm, not necessarily a bug: report the numbers.

Also once, with the driver up: `ros2 topic hz /joint_states`, and whether
`ros2 topic echo --once /joint_states` shows non-zero velocities while the arm moves.

### 3. Workspace setup: orientation, bounds, no-go boxes, start pose (operator)

```bash
python -m hilserl_gen3_lite.workspace --config_path=config/hilserl/env_gen3_lite_cup.json \
    --run.confirm_motion=true
```

Nothing is recorded. The arm homes, moves to the configured start pose, and then only
moves while the takeover button is held, or when a turn command is typed. The operator
drives with the gamepad and types short commands in the terminal; everything taught takes
effect at once, so it can be tried out before it is saved. `help` lists the commands; the
docstring of `hilserl_gen3_lite/workspace.py` describes them.

**First, the gamepad.** lerobot's mapping (pygame numbers; they vary by controller, so
confirm each one before going further; an empty line prints the tool position, the
joints and the gripper):

| Control | pygame | Effect |
|---|---|---|
| Takeover | button 5 (RB on most pads) | the human drives while it is held |
| Left stick | axes 0, 1 | y, x |
| Right stick, vertical | axis 3 | z |
| Close gripper | button 6 | closes while held, then stays closed |
| Open gripper | button 7 | opens while held, then stays open |
| Success | button 3 (Y) | in this tool: marks a point. Later: reward 1, episode ends |
| Failure | button 1 | episode ends |
| Re-record | button 0 | discards the demo being recorded |

**Then, in this order:**

1. **Orientation** (`yaw DEG`, `pitch DEG`, `roll DEG`). The policy cannot turn the tool,
   so the orientation chosen here is the one the whole task is done in. The configured
   start pose has Kinova home's orientation: gripper level, pointing 60 degrees to the
   left of straight ahead (not straight ahead, whatever older notes say). At the
   configured position a level gripper can turn only about 26 degrees further towards
   straight ahead before the elbow joint reaches its limit; pointing straight down
   (`pitch 90`) is reachable. Turns go at 15 degrees per second through every limit, and
   stop with a message where the arm cannot follow. Decide between a side grasp and a
   top-down grasp with the cup in place.
2. **Start pose** (`r`) with the arm where episodes should begin. It warns if a joint is
   close to its limit there; believe it and pick another pose.
3. **Tool box** (`w`, drive, `done`): sweep the tool through everything the task needs:
   the pick spot, the place spot, the top of the carry, the lowest point of the grasp. Do
   the whole task by hand a couple of times while it records. The box is what was swept
   plus 2 cm. A smaller box learns faster and is safer; do not keep slack "just in case".
4. **No-go boxes** (`n NAME`, mark points with the success button, `done`; or
   `n NAME x0 y0 z0 x1 y1 z1` from a tape measure) around the camera and its mount and
   anything else rigid within reach. Bring the tool tip near the extremes of the object,
   not onto it; stop two or three centimetres short. The box is the marked points plus
   3 cm, and every link then stays 2 cm clear of it. The table needs no box (it is the
   guard's plane) and neither does the arm itself.
5. **Try it**: drive at each wall of the tool box (the tool stops there) and towards each
   no-go box (the arm holds, and the log says which link and which box). Do the task by
   hand once more inside the final box, with the real cup: reach, grasp, lift, carry, set
   down, release. `show` prints the settings and writes `runs/workspace.png`, a top and a
   side view; look at it together.
6. **`save`**, which writes `ee_bounds_min/max`, `no_go_boxes` and
   `reset_joint_positions` under `env.robot` in both cup configs and refuses if either
   file would no longer load. Then `quit`.

Afterwards (you): `python -m pytest tests/test_configs.py`, and read the saved values
back to the operator. Things that go wrong here:

- **The box does not contain the grasp.** The guard box's floor is z = 0.02 m above the
  bottom of the arm's base; the earlier DAgger work grasped a cup with the tool at
  z = 0.065 m. If the tool cannot get low enough, or `Workspace guard blocked` names the
  table plane, report where rather than editing the guard.
- **The grasp itself.** Watch the gripper number while holding the cup: it should stall
  partway, where the cup stops the fingers. If a full close crushes the cup, lower
  `env.robot.ros2_interface.gripper_close_position` in both configs.
- **Fingers reach further than the tool point.** With the gripper open a fingertip is
  about 6 cm ahead of the tool position that is printed. The tool box limits the tool
  point; the no-go boxes and the table plane are what keep the fingers off things.

### 4. Decide the task details (operator; write them down)

- Cup start: one marked spot for the first run. A single fixed camera gives the policy
  little to locate a cup that moves around.
- Place target: a second mark, 10 to 20 cm away, inside the box.
- Success, exactly: for example "cup upright on the mark and the gripper open". The
  operator presses the success key at the same moment every time; reward is given on
  that step only.
- Failure or timeout: let the 20 s run out, or press the failure key.

### 5. Demos (operator)

```bash
python -m hilserl_gen3_lite.record --config_path=config/hilserl/env_gen3_lite_cup.json \
    --run.confirm_motion=true
```

20 episodes into `data/gen3_lite_cup_hil_demos`. The arm stays still unless takeover is
held. Between episodes the arm opens the gripper and returns to the reset pose, then
waits out the rest of 8 s for the cup to be put back. If an episode ends with the cup
still in the gripper, it is dropped where the arm is when the reset starts.

Afterwards (you): check the dataset has 20 episodes, each ending with reward 1, and that
`complementary_info.discrete_penalty` is -0.02 on exactly two frames of a clean demo.

### 6. Crop (you prepare, operator draws)

```bash
python -m lerobot.rl.crop_dataset_roi --repo-id local/gen3_lite_cup_hil_demos \
    --root data/gen3_lite_cup_hil_demos
```

Upstream's tool; it needs a display. Put the printed `crop_params_dict` into
`env.processor.image_preprocessing` in both configs and point `dataset` in the train
config at the cropped copy it writes. It has not been run in this project, so check the
cropped dataset still has the `complementary_info.*` columns; if not, skip the crop for
the first run (full frames are resized to 128x128) and report it.

### 7. Train

Terminal A (you may start this one):

```bash
python -m hilserl_gen3_lite.learner --config_path=config/hilserl/train_gen3_lite_cup.json \
    --run.dir=runs/cup_baseline_s0
```

Terminal B (operator), once the learner prints `gRPC server started`:

```bash
python -m hilserl_gen3_lite.actor --config_path=config/hilserl/train_gen3_lite_cup.json \
    --run.dir=runs/cup_baseline_s0 --run.confirm_motion=true \
    --run.eval_interval_steps=5000 --run.eval_episodes=10
```

Stop with Ctrl+C in the actor, then the learner. The actor holds and disconnects the arm
and logs `robot limit counters: {...}` as it exits.

- **Evaluation blocks** cost operator time: 10 episodes of up to 28 s each. Every 5000
  steps that is about a quarter of the session; at the default of 2000 it would be half.
  During a block the operator keeps hands off, takes over only for safety (that episode
  then counts as a failure), and still presses the success key.
- **How long:** unknown. The HIL-SERL paper reports one to two and a half hours for its
  tasks, on different hardware and with wrist cameras. Do not promise a number.
- **Operator notes** (from the paper and its follow-ups): intervene often early, then
  less; give short, specific corrections and let the robot explore otherwise; do not
  drive whole episodes to success by hand, which inflates the critic's values.

While it runs (you): watch `runs/cup_baseline_s0/episodes.csv` and `train.csv`, the
actor's log for `Policy FPS ... below required`, `Workspace guard blocked`,
`IK rejected` and `StaleObservationError`, and the machine's free memory.

### 8. Afterwards (you)

```bash
python -m hilserl_gen3_lite.summarize_run runs/cup_baseline_s0
```

## Things I would look at first if something is off

- **Arm does not move, no error.** `ros2 control list_controllers`: the trajectory
  controller must be active. A joint outside its URDF limit at start-up deactivates it
  (`docs/LYRICAL_KINOVA.md` §7.1); home the arm and relaunch the driver.
- **`StaleObservationError`.** Joint state or a camera frame older than 200 ms. Check the
  rates before raising `max_obs_age_s`; a camera that cannot hold 10 Hz is a problem for
  learning too.
- **The arm lags or the tool path wobbles.** At 0.1 m/s the arm runs about 2 cm behind
  its target, and the per-joint cap (0.1 rad from the measured position) acts on about a
  third of all commands. That was the source of two bugs found in Gazebo; read the
  paragraph under the limits table in GEN3_LITE_HILSERL.md before touching
  `Gen3LiteHil.send_action`.
- **Counters.** `step_clipped_commands` is high by design: any diagonal unit action is
  longer than 1 cm and is scaled back. `box_clipped_commands` climbing means the policy
  is pressing against the tool box. `guard_blocks` means it is pressing against a no-go
  box or the table (the log line names which); a few are normal, a steady stream means
  the tool box overlaps something it should not. `ik_rejections` in numbers mean the box
  reaches somewhere the arm cannot go with the frozen orientation.
- **Gripper never moves on the real arm.** On Jazzy it is `control_msgs/GripperCommand`
  on `/gen3_lite_2f_gripper_controller/gripper_cmd`. `Gripper action server not
  available` in the log means the controller is not up.
- **Images.** `compat.py` rescales demo images for the offline buffer. If learning from
  the demos looks far worse than expected, check the image range in a sampled batch from
  both buffers before anything else.
- **Upstream quirk, not ours to fix.** With `utd_ratio: 2`, upstream's final update of
  each step drops `complementary_info`, so the gripper penalty reaches the discrete
  critic in half of its updates.
- **Memory and disk.** Both buffers live in RAM; each checkpoint also writes them to disk
  as video.

## What to hand back

1. Machine facts: OS, ROS distro, GPU and driver, RAM, torch version, kortex commit, the
   camera model and where it ended up.
2. Each step above: done, failed or skipped, with the command and the relevant output.
3. The smoke-test output, the `/joint_states` rate and whether it carries velocities.
4. The final configs (orientation, bounds, no-go boxes, start pose, camera, crop),
   `runs/workspace.png`, and how success was defined.
5. From the run: `summary.json`, `curves.png`, the two rise-time numbers, the final
   counters line, session length in steps and minutes, and the operator's impression of
   how the takeovers went.
6. Every code or config change, each with its reason and its test.
7. What you verified by running it, separately from what you inferred.
8. Anything suggesting the default limits are wrong for this arm. Flag it; do not change
   the defaults.

## If you want the simulator after all

`docker/gazebo/` runs a simulated Gen3 Lite with the real controllers in a container
with no network (see its README): `docker/gazebo/run.sh up`, then
`docker/gazebo/run.sh exec '<command>'`. `config/hilserl/gazebo_*.json` is the state-only
reach used on the laptop. It is the quickest way to reproduce an adapter problem without
the arm.

## Map

| Path | What |
|---|---|
| `docs/GEN3_LITE_HILSERL.md` | Design, limits, log formats, verified / not verified, Gazebo results |
| `config/hilserl/*_gen3_lite_cup.json` | This task. `*_gen3_lite_reach.json`: the gripper-less reach. `dryrun_*`: fake arm |
| `lerobot_robot_ros/lerobot_robot_ros/gen3_lite_hil.py`, `ee_delta.py`, `config_gen3_lite_hil.py` | The robot class, the step limiter and kinematics, the limits |
| `hilserl_gen3_lite/env.py` | Gym environment, processor pipelines, the gripper penalty step |
| `hilserl_gen3_lite/{workspace,smoke_test,record,actor,learner}.py` | The commands of this session |
| `lerobot_backends/ros2/workspace_guard.py` | Table plane, guard box, no-go boxes, self-collision |
| `hilserl_gen3_lite/compat.py` | The two in-memory fixes to lerobot 0.6.1 |
| `tests/` | 92 hardware-free tests |
