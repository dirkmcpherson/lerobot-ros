# ManiSkill Training Pipeline

Train a diffusion policy on ManiSkill simulation data using the `lerobot_backends` framework.

## Prerequisites

```bash
conda activate lerobot-ros
pip install mani-skill
```

## Pipeline

### 1. Record Data

Collects reach-to-target demonstrations in ManiSkill's Panda environment. Each episode linearly interpolates from the home pose to a randomized target joint pose. The target is stored as `observation.environment_state` so the policy learns to condition on it.

```bash
# Record 50 episodes (takes ~10 seconds)
python record_maniskill_data.py --num-episodes 50

# Record more for better generalization
python record_maniskill_data.py --num-episodes 200 --overwrite

# Visualize while recording
python record_maniskill_data.py --num-episodes 5 --render --overwrite
```

Dataset is saved to `data/lerobot/maniskill_reach/`.

**Dataset features:**
| Feature | Shape | Description |
|---|---|---|
| `observation.state` | (7,) | Panda arm joint positions |
| `observation.environment_state` | (7,) | Target joint pose |
| `action` | (7,) | Commanded joint positions |

### 2. Train

Trains a DiffusionPolicy using `lerobot-train`:

```bash
bash train_maniskill.sh
```

This runs 5000 steps with batch size 64, saving checkpoints every 1000 steps to `outputs/train/maniskill_reach/`.

To customize training, edit `train_maniskill.sh` or pass overrides directly:

```bash
# More steps, smaller batch
lerobot-train \
    --policy.type=diffusion \
    --dataset.repo_id=lerobot/maniskill_reach \
    --output_dir=outputs/train/maniskill_reach \
    --batch_size=32 \
    --steps=10000 \
    ...
```

### 3. Evaluate

Runs the trained policy in ManiSkill and reports joint-space error to target:

```bash
python eval_maniskill.py --num-episodes 10

# With visualization
python eval_maniskill.py --render --num-episodes 5

# Specific checkpoint
python eval_maniskill.py --checkpoint outputs/train/maniskill_reach/checkpoints/005000/pretrained_model
```

## Architecture

The ManiSkill backend (`lerobot_backends/maniskill/`) implements the same `RobotBackend` protocol as the ROS2 backend. Scripts use `BackendRobot(config)` — the config type selects the backend via the factory.

```
record_maniskill_data.py  ──▶  data/lerobot/maniskill_reach/
                                        │
                               train_maniskill.sh (lerobot-train)
                                        │
                               outputs/train/maniskill_reach/
                                        │
                              eval_maniskill.py  ──▶  results
```

### Key files

| File | Purpose |
|---|---|
| `lerobot_backends/maniskill/config.py` | `ManiSkillPickCubeConfig` — env ID, obs mode, joint mapping, env state keys |
| `lerobot_backends/maniskill/backend.py` | `ManiSkillBackend` — wraps gymnasium env as `RobotBackend` |
| `lerobot_backends/gym_adapter.py` | `RobotGymEnv` — wraps any `BackendRobot` back into a gymnasium env |
| `record_maniskill_data.py` | Data collection with linear interpolation + noise |
| `train_maniskill.sh` | Training script |
| `eval_maniskill.py` | Policy evaluation with error reporting |

## Extending to Other Tasks

To use a different ManiSkill environment, create a new config:

```python
from dataclasses import dataclass, field
from lerobot_backends.maniskill.config import ManiSkillBackendConfig

@dataclass
class MyTaskConfig(ManiSkillBackendConfig):
    env_id: str = "PushCube-v1"
    control_mode: str = "pd_joint_pos"
    obs_mode: str = "state"
    env_state_keys: list[str] = field(default_factory=lambda: [
        "tcp_pose", "goal_pos",
    ])
```

Then use it in your recording/eval scripts:

```python
config = MyTaskConfig()
robot = BackendRobot(config)
robot.connect()
obs = robot.backend.reset()
```

The `env_state_keys` field controls which entries from ManiSkill's `obs["extra"]` dict get flattened into the `env_state` observation. Check your env's obs structure with:

```python
raw_obs, info = robot.backend._env.reset()
print(raw_obs["extra"].keys())
```
