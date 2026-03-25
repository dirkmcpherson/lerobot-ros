# DreamZero: 6-DOF Single-Arm Finetuning Guide

## Overview

DreamZero is a 14B parameter World Action Model (WAM) built on a pretrained Wan2.1 image-to-video diffusion backbone. It jointly predicts future video frames and robot actions via flow matching, enabling zero-shot generalization to unseen tasks and few-shot adaptation to new embodiments. It is an NVIDIA GEAR Lab project.

## Why It Works for a 6-DOF Single Arm

1. **DROID (Franka single-arm) is a first-class embodiment.** The paper trains and evaluates DreamZero-DROID on the Franka single-arm robot. Single-arm is core, not an afterthought.

2. **DOF-agnostic architecture.** The action/state dimensions are set entirely by your data config. The `MultiEmbodimentActionEncoder` creates a per-embodiment MLP sized to whatever dimensions your data has. A 6-DOF arm with gripper gives you a 7-dim action space (vs DROID's 9-dim) -- no code changes needed.

3. **Fewer DOF is easier.** The paper notes that higher-DOF robots need more data because the implicit inverse dynamics mapping grows combinatorially. A 6-DOF arm is simpler than 7-DOF, so you may need less data.

4. **Few-shot embodiment adaptation is a headline result.** DreamZero pretrained on AgiBot transfers to a new robot (YAM) with only **30 minutes of play data** via LoRA finetuning.

## Hardware Requirements

### Training

The model is **14B parameters**. LoRA finetuning freezes most weights and uses DeepSpeed ZeRO-2, but you still need multi-GPU. With `per_device_train_batch_size=4` at 320x176 resolution, plan on **4-8x H100/A100 80GB**. ZeRO-3 with CPU offload could reduce this but will be slower.

### Inference

The paper achieves 7Hz on **2x GB200s** with all optimizations. On **2x H100s** without Blackwell-specific optimizations, expect ~3-5s per action chunk -- usable with asynchronous execution (robot executes previous chunk while next one computes).

### Smaller Option

The **Wan2.2-TI2V-5B** backbone (see `docs/WAN22_BACKBONE.md`) is ~3x smaller. The ablation shows 5B gets 21% vs 50% task progress for 14B -- a real drop, but potentially viable for a simple pick-and-place task, and much friendlier on hardware.

## Paper Results (Single-Arm Reference)

- **DROID-Franka seen tasks**: 75% success rate
- **DROID-Franka unseen tasks**: 49% task progress, 22.5% success rate
- **Few-shot adaptation** (AgiBot to YAM, 30 min data): retains zero-shot generalization

---

## Step-by-Step: Finetune and Run "Pick Up the Block"

### Prerequisites

- A 6-DOF arm with a parallel gripper and 2-3 cameras (1 external + 1 wrist minimum)
- A teleoperation setup to record demonstrations (LeRobot v2 format)
- 4-8x H100/A100 80GB GPUs (training), 2x H100+ (inference)
- ~150GB disk for checkpoints

### Step 0: Install Dependencies and Download Checkpoints

```bash
# Install DreamZero
pip install -e .

# Download pretrained DreamZero-AgiBot checkpoint (~45GB)
# This is the base model you'll LoRA-finetune from
huggingface-cli download GEAR-Dreams/DreamZero-AgiBot \
    --local-dir ./checkpoints/DreamZero-AgiBot

# Download Wan2.1 backbone weights (auto-downloaded by training script too)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

# Download tokenizer
huggingface-cli download google/umt5-xxl \
    --local-dir ./checkpoints/umt5-xxl
```

### Step 1: Collect Teleoperation Data

Record 50-100 demonstrations of picking up blocks. More diversity in block position/color matters more than repetition. The paper showed results with as few as 55 trajectories (~30 min).

What to record:
- **Cameras**: 2-3 views as MP4 video files
- **State**: 6 joint positions + 1 gripper position per timestep
- **Actions**: 6 joint position targets + 1 gripper target per timestep
- **Language annotation**: "pick up the block" (or varied: "pick up the red block", "grab the block", etc.)
- **FPS**: 30Hz recommended (match to your robot's control frequency)

#### LeRobot Version Note

**DreamZero requires LeRobot v2.0 format**, but the latest LeRobot (>= 0.4.0) records in **v3.0 format** by default. The key difference is that v2 stores one episode per file while v3 packs many episodes into larger shard files.

**Recommended approach**: Collect data with the latest LeRobot tooling (v3), then convert to v2 before running the GEAR converter:

```bash
# After collecting data with lerobot >= 0.4.0 (outputs v3 format):
python -m lerobot.datasets.v30.convert_dataset_v30_to_v21 --repo-id=<your-dataset>
```

Alternatively, pin `lerobot < 0.4.0` to record directly in v2 format and skip the conversion step.

#### Expected v2 Directory Structure

After conversion (or if recording directly in v2), your dataset should look like:

```
data/myarm/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # columns: observation.state, action, timestamp, ...
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_exterior/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       └── observation.images.cam_wrist/
│           ├── episode_000000.mp4
│           └── ...
└── meta/
    └── info.json                    # must contain: features, total_episodes, fps
```

Each parquet file should have columns for `observation.state` (7-dim: 6 joints + 1 gripper), `action` (7-dim), and `annotation.task` (string).

### Step 2: Convert Dataset to GEAR Format

```bash
python scripts/data/convert_lerobot_to_gear.py \
    --dataset-path ./data/myarm \
    --embodiment-tag myarm \
    --state-keys '{"joint_position": [0, 6], "gripper_position": [6, 7]}' \
    --action-keys '{"joint_position": [0, 6], "gripper_position": [6, 7]}' \
    --relative-action-keys joint_position \
    --task-key annotation.task \
    --force
```

This creates metadata files under `data/myarm/meta/` without modifying your parquet or video files:
- `modality.json` -- maps state/action/video keys with index ranges
- `embodiment.json` -- `{"embodiment_tag": "myarm"}`
- `stats.json` -- per-feature normalization statistics (mean, std, q01, q99)
- `relative_stats_dreamzero.json` -- relative action stats (action minus state)
- `tasks.jsonl` -- unique task descriptions
- `episodes.jsonl` -- per-episode metadata

### Step 3: Register the Embodiment Tag

Add your arm to `groot/vla/data/schema/embodiment_tags.py`:

```python
class EmbodimentTag(Enum):
    ...
    MYARM = "myarm"
    """
    6-DOF single arm with parallel gripper.
    """
```

And add `"myarm"` to the `VALID_EMBODIMENT_TAGS` list in `scripts/data/convert_lerobot_to_gear.py`.

### Step 4: Add Modality Config and Transforms

Edit `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`.

Add the modality config for your arm (adjust camera names to match your `modality.json`):

```yaml
modality_config_myarm:
  video:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    eval_delta_indices: [0]
    modality_keys:
      - video.cam_exterior
      - video.cam_wrist
  state:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:
      - state.joint_position
      - state.gripper_position
  action:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    modality_keys:
      - action.joint_position
      - action.gripper_position
  language:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:
      - annotation.task
```

Add the transform block:

```yaml
transform_myarm:
  _target_: groot.vla.data.transform.ComposedModalityTransform
  transforms:
    # Video
    - <<: *totensor_cfg
      apply_to: ${modality_config_myarm.video.modality_keys}
    - <<: *crop_cfg
      apply_to: ${modality_config_myarm.video.modality_keys}
    - <<: *resize_cfg
      apply_to: ${modality_config_myarm.video.modality_keys}
    - <<: *color_jitter_cfg
      apply_to: ${modality_config_myarm.video.modality_keys}
    - <<: *to_numpy_cfg
      apply_to: ${modality_config_myarm.video.modality_keys}

    # State
    - _target_: groot.vla.data.transform.StateActionToTensor
      apply_to: ${modality_config_myarm.state.modality_keys}
    - _target_: groot.vla.data.transform.StateActionTransform
      apply_to: ${modality_config_myarm.state.modality_keys}
      normalization_modes:
        state.joint_position: q99
        state.gripper_position: q99

    # Action
    - _target_: groot.vla.data.transform.StateActionToTensor
      apply_to: ${modality_config_myarm.action.modality_keys}
    - _target_: groot.vla.data.transform.StateActionTransform
      apply_to: ${modality_config_myarm.action.modality_keys}
      normalization_modes:
        action.joint_position: q99
        action.gripper_position: q99

    # Concat
    - _target_: groot.vla.data.transform.ConcatTransform
      video_concat_order: ${modality_config_myarm.video.modality_keys}
      state_concat_order: ${modality_config_myarm.state.modality_keys}
      action_concat_order: ${modality_config_myarm.action.modality_keys}

    # Model-specific (required, don't change)
    - ${model_specific_transform}
```

Register in the four global maps at the bottom of the same file:

```yaml
modality_configs:
  ...
  myarm: ${modality_config_myarm}

transforms:
  ...
  myarm: ${transform_myarm}

metadata_versions:
  ...
  myarm: '0221'

fps:
  ...
  myarm: 30
```

**Important**: The `modality_keys` values (e.g., `state.joint_position`) must exactly match the keys in your generated `modality.json`.

### Step 5: Create a Dataset YAML

Create `groot/vla/configs/data/dreamzero/myarm_relative.yaml`:

```yaml
# @package _global_

defaults:
  - dreamzero/base_48_wan_fine_aug_relative
  - _self_

max_state_dim: 64
use_global_metadata: false
relative_action: true
relative_action_per_horizon: false
relative_action_keys:
  - joint_position
max_chunk_size: 5
dataset_shard_sampling_rate: 0.1
mixture_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotMixtureDataset.from_mixture_spec
single_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotSubLangSingleActionChunkDatasetDROID

myarm_data_root: ???

train_dataset:
  _target_: ${mixture_dataset_cls}
  _convert_: object
  mixture_spec:
    - dataset_path:
        myarm:
          - ${myarm_data_root}
      dataset_weight: 1.0
      distribute_weights: true

  dataset_class: ${single_dataset_cls}
  all_modality_configs: ${modality_configs}
  all_transforms: ${transforms}
  metadata_versions: ${metadata_versions}
  fps: ${fps}
  dataset_kwargs:
    video_backend: decord
    use_global_metadata: ${use_global_metadata}
    max_chunk_size: ${max_chunk_size}
    relative_action: ${relative_action}
    relative_action_keys: ${relative_action_keys}
    relative_action_per_horizon: ${relative_action_per_horizon}
  mixture_kwargs:
    training: true
    balance_dataset_weights: false
    seed: 42
    shard_sampling_rate: ${dataset_shard_sampling_rate}
```

### Step 6: Create a Training Script

Create `scripts/train/myarm_training.sh`:

```bash
#!/bin/bash
export HYDRA_FULL_ERROR=1

# ============ CONFIGURATION ============
DATA_ROOT=${DATA_ROOT:?"Set DATA_ROOT to your GEAR-converted dataset path"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_myarm_lora"}

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
# =======================================

# Auto-download weights if missing
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [ ! -f "$DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: meta/embodiment.json missing -- run convert_lerobot_to_gear.py first"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/myarm_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=2500 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=5000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    myarm_data_root=$DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=./checkpoints/DreamZero-AgiBot \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
```

Key differences from the template for your 6-DOF arm:
- `num_views=2` (2 cameras instead of 3)
- `max_steps=5000` (small dataset, LoRA converges fast -- the AgiBot script uses 5k too)
- `per_device_train_batch_size=1` (safe for GPU memory)

### Step 7: Launch Training

```bash
DATA_ROOT=./data/myarm bash scripts/train/myarm_training.sh

# Or with overrides:
DATA_ROOT=./data/myarm NUM_GPUS=4 OUTPUT_DIR=./checkpoints/myarm_run1 \
    bash scripts/train/myarm_training.sh
```

Training will save LoRA checkpoints to `OUTPUT_DIR` every 2500 steps. At 5000 steps with a small dataset, this should take a few hours on 8x H100s.

### Step 8: Run Inference -- Pick Up the Block

The inference server uses WebSockets. You'll need to adapt `socket_test_optimized_AR.py` for your arm's observation/action format, or write a simpler inference loop using `GrootSimPolicy` directly.

**Option A: Adapt the WebSocket server**

The server in `socket_test_optimized_AR.py` expects roboarena-format observations. You would modify the `ARDroidRoboarenaPolicy` class to map your arm's observation keys. The key method is the observation converter that maps camera images and joint state into the format the model expects.

**Option B: Direct inference loop (simpler for a lab setup)**

Write a control loop using `GrootSimPolicy` from `groot/vla/model/n1_5/sim_policy.py`:

```python
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema.embodiment_tags import EmbodimentTag
import numpy as np
import torch

# 1. Load model
policy = GrootSimPolicy(
    embodiment_tag=EmbodimentTag.MYARM,
    model_path="./checkpoints/dreamzero_myarm_lora/checkpoint-5000",
)
policy.model.eval()
policy.model = policy.model.to(dtype=torch.bfloat16, device="cuda")

# 2. Control loop
instruction = "pick up the block"
frame_buffer = []

while not done:
    # Get current observation from your robot
    img_exterior = robot.get_camera("exterior")   # (H, W, 3) uint8
    img_wrist = robot.get_camera("wrist")          # (H, W, 3) uint8
    joint_pos = robot.get_joint_positions()         # (6,) float32
    gripper_pos = robot.get_gripper_position()      # (1,) float32

    # Build observation dict matching your modality keys
    obs = {
        "video.cam_exterior": img_exterior,
        "video.cam_wrist": img_wrist,
        "state.joint_position": np.concatenate([joint_pos]),
        "state.gripper_position": gripper_pos,
        "annotation.task": instruction,
    }

    # Accumulate frames (first call: 1 frame, then 4 per chunk)
    frame_buffer.append(obs)

    # Run inference -- returns action chunk of shape (24, 7)
    # [joint_position(6), gripper_position(1)]
    actions = policy.get_action(frame_buffer)

    # Execute action chunk on robot
    for action in actions:
        joint_targets = action[:6]   # 6-DOF joint positions
        gripper_target = action[6]   # gripper open/close
        robot.command_joints(joint_targets, gripper_target)
        robot.wait_for_step()        # wait one control step (1/30s)
```

Note: The actual `GrootSimPolicy` API involves building batch tensors and running the eval transform. The above is a simplified sketch -- refer to `socket_test_optimized_AR.py` lines 87-227 and `eval_utils/run_sim_eval.py` for the precise observation formatting, normalization, and action unnormalization. The key steps are:

1. Images are resized to (180, 320) and normalized to [-1, 1]
2. State is normalized using q99 statistics from `stats.json`
3. Model outputs normalized relative actions
4. Actions are unnormalized and converted back to absolute: `absolute = relative + current_state`

### Step 9: Iterate

If pick-up success is low:
- **Collect more diverse data**: vary block position, color, lighting, table height
- **Train longer**: increase `max_steps` to 10k-20k
- **Add cameras**: a third viewpoint helps with spatial reasoning
- **Check action smoothness**: the model applies Savitzky-Golay filtering on action chunks to suppress high-frequency noise

---

## Pre-Training Checklist

- [ ] `meta/embodiment.json` exists and has `"myarm"` tag
- [ ] `meta/modality.json` has correct state/action/video/annotation keys
- [ ] `meta/stats.json` and `meta/relative_stats_dreamzero.json` exist
- [ ] `meta/tasks.jsonl` and `meta/episodes.jsonl` exist
- [ ] Embodiment tag `myarm` matches keys in `modality_configs` / `transforms` / `metadata_versions` / `fps`
- [ ] YAML `modality_keys` match `modality.json` keys exactly (with `state.`/`action.`/`video.`/`annotation.` prefix)
- [ ] Every state and action key appears in `normalization_modes` in the transform block
- [ ] `relative_action_keys` lists sub-key names that exist in both state and action
- [ ] Wan2.1-I2V-14B-480P and umt5-xxl weights downloaded
- [ ] DreamZero-AgiBot checkpoint downloaded to `./checkpoints/DreamZero-AgiBot`

## Key Considerations

- **Cameras**: 2-3 cameras. System concatenates multi-view frames into a single image. One external + one wrist camera is the minimum.
- **Data format**: Actions should be joint positions. The system uses **relative actions** (action - current_state) normalized to [-1, 1] using q99 percentile clipping.
- **Task instructions**: Model is conditioned on language. Even simple annotations like "pick up the block" work. Varying the phrasing can help.
- **Sub-centimeter precision**: The paper notes limitations on tasks requiring very fine precision (key insertion, fine assembly). Block picking should be well within capability.
