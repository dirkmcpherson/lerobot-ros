# LeRobot ROS

Multi-backend robot interface for the [LeRobot](https://github.com/huggingface/lerobot) framework. Supports ROS 2 hardware robots, ManiSkill simulation, and is extensible to other backends.

Forked from [ycheng517/lerobot-ros](https://github.com/ycheng517/lerobot-ros) and extended with a unified backend architecture.

## Architecture

```
lerobot_backends/
├── backend.py          # RobotBackend Protocol
├── config.py           # BackendRobotConfig base
├── robot.py            # BackendRobot(Robot) — generic wrapper
├── factory.py          # make_backend(config) — selects backend by config type
├── gym_adapter.py      # RobotGymEnv — wraps BackendRobot as gymnasium.Env
├── ros2/
│   ├── config.py       # ROS2BackendConfig, KinovaGen3Config, KinovaGen3LiteConfig, etc.
│   ├── backend.py      # ROS2Backend — wraps ROS2Interface + cameras
│   ├── ros_interface.py
│   └── moveit_servo.py
└── maniskill/
    ├── config.py       # ManiSkillBackendConfig, ManiSkillPickCubeConfig
    └── backend.py      # ManiSkillBackend — wraps gymnasium env
```

Usage: `BackendRobot(config)` — the config type selects the backend automatically.

```python
from lerobot_backends.ros2.config import KinovaGen3LiteConfig
from lerobot_backends.robot import BackendRobot

robot = BackendRobot(KinovaGen3LiteConfig())
robot.connect()
robot.backend.reset()
obs = robot.backend.get_observation()
```

## Supported Robots

| Robot | Config | Backend | Control Modes |
|-------|--------|---------|---------------|
| Kinova Gen3 (7-DOF) | `KinovaGen3Config` | ROS2 | Joint trajectory |
| Kinova Gen3 Lite (6-DOF) | `KinovaGen3LiteConfig` | ROS2 | Joint trajectory + gripper action |
| Annin AR4 | `AnninAR4Config` | ROS2 | Cartesian velocity (MoveIt Servo) |
| SO-101 | `SO101ROSConfig` | ROS2 | Joint trajectory |
| ManiSkill PickCube | `ManiSkillPickCubeConfig` | ManiSkill | Gymnasium env |

## ROS 2 Control Modes

**Arm:**
- `ActionType.JOINT_POSITION` — `position_controllers/JointGroupPositionController`
- `ActionType.JOINT_TRAJECTORY` — `joint_trajectory_controller/JointTrajectoryController`
- `ActionType.CARTESIAN_VELOCITY` — MoveIt Servo

**Gripper:**
- `GripperActionType.TRAJECTORY` — publishes `JointTrajectory` to gripper topic
- `GripperActionType.ACTION` — sends goals to `GripperActionController`

## Scripts

### Data Collection

```bash
# Teleoperated (spacemouse + differential IK)
python record_kinova_data_teleoperated.py --robot gen3_lite --input spacemouse --urdf /tmp/gen3_lite.urdf

# Scripted (random target positions)
python record_kinova_data.py

# ManiSkill simulation
python record_maniskill_data.py
```

### Training

```bash
# Diffusion policy on cube stacking data
bash train_cube_stacking.sh

# Diffusion policy on reach data
bash train_kinova_reach.sh

# Diffusion policy on ManiSkill data
bash train_maniskill.sh
```

### Evaluation

```bash
python eval_cube_stacking.py
python eval_kinova_reach.py
python eval_maniskill.py
```

See [MANISKILL.md](MANISKILL.md) for the full ManiSkill pipeline walkthrough.

### Playback

```bash
python playback_kinova_trajectory.py
```

## Spacemouse Teleoperation

Uses [pinocchio](https://github.com/stack-of-tasks/pinocchio) for differential IK. Requires a URDF file:

```bash
# Generate URDF from xacro
source ~/workspace/ros2_kortex_ws/install/setup.bash
xacro ~/workspace/ros2_kortex_ws/src/ros2_kortex/kortex_description/robots/gen3_lite_gen3_lite_2f.xacro > /tmp/gen3_lite.urdf
```

**Button mapping (with gripper):** Left = save episode, Right = toggle gripper, Both = discard + quit

## Gazebo Simulation

```bash
# Launch Gen3 7-DOF (no gripper — Gazebo gripper is a known issue)
ros2 launch kortex_bringup kortex_sim_control.launch.py \
  use_sim_time:=true launch_rviz:=false robot_type:=gen3 dof:=7

# Launch Gen3 Lite 6-DOF with gripper
ros2 launch kortex_bringup kortex_sim_control.launch.py \
  use_sim_time:=true launch_rviz:=false robot_type:=gen3_lite dof:=6 gripper:=gen3_lite_2f
```

Modified ros2_kortex files are backed up in `forked_cortex/`.

## Gen3 sweep task (gen3-kinesthetic-teaching-ros2)

Diffusion policy on the Gazebo sweep task of
[gen3-kinesthetic-teaching-ros2](https://github.com/AABL-Lab/gen3-kinesthetic-teaching-ros2),
which carries this repo as its `lerobot-ros` submodule (the scripts find the parent repo
automatically). Sim setup is that repo's `native/README.md`. Observations are the 7 arm
joints plus the xyz of all 15 cubes (`observation.environment_state`, a sim-only oracle);
actions are the joint controller's reference one frame ahead, at 10 Hz.

```bash
# in the teaching repo: native/sim_up.sh            (headless sim)
python record_gen3_sweep_sim.py --episodes 50        # scripted demos -> data/lerobot/gen3_sweep_sim
bash train_gen3_sweep_sim.sh                         # 20k steps, ~26 min on an RTX 3080
python eval_gen3_sweep_sim.py --episodes 10          # cubes swept / 15 from unseen start poses
# in the teaching repo: GUI=1 native/sim_up.sh       (GUI on a virtual display)
python video_gen3_sweep_sim.py outputs/gen3_sweep_sim_demo_vs_dp.mp4 --seeds 1000 1001 1002
```

Shared constants, features and ROS plumbing: `gen3_sweep_sim.py`; robot config:
`KinovaGen3SweepSimConfig`.

## Prerequisites

- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)
- [ros2_control](https://control.ros.org/rolling/index.html)
- [MoveIt 2](https://moveit.ai/install-moveit2/binary) (for cartesian velocity control)
- [pinocchio](https://github.com/stack-of-tasks/pinocchio) (for spacemouse IK)

## Install

```bash
conda create -y -n lerobot-ros python=3.12
conda activate lerobot-ros
conda install -c conda-forge libstdcxx-ng -y
source /opt/ros/jazzy/setup.sh

git clone <this-repo>
cd lerobot-ros
pip install -e lerobot_robot_ros lerobot_teleoperator_devices
```

## Adding a New Robot

1. Create a config in `lerobot_backends/ros2/config.py`:

```python
@dataclass
class MyRobotConfig(ROS2BackendConfig):
    action_type: ActionType = ActionType.JOINT_TRAJECTORY
    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            arm_joint_names=["joint_1", "joint_2", ...],
            gripper_joint_name="gripper_joint",
        )
    )
    home_position: list[float] | None = field(
        default_factory=lambda: [0.0, 0.0, ...]
    )
```

2. Use it: `robot = BackendRobot(MyRobotConfig())`

## Adding a New Backend

Implement the `RobotBackend` Protocol from `lerobot_backends/backend.py`:

```python
class RobotBackend(Protocol):
    @property
    def observation_features(self) -> dict[str, type | tuple]: ...
    @property
    def action_features(self) -> dict[str, type]: ...
    @property
    def is_connected(self) -> bool: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def reset(self) -> dict[str, Any]: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]: ...
```

Then register your config type in `lerobot_backends/factory.py`.
