#!/bin/bash
source /home/james/miniconda3/bin/activate lerobot-ros

export HF_LEROBOT_HOME=$(pwd)/data
export HF_HUB_OFFLINE=1

# Dataset features (from record_maniskill_data.py):
#   observation.state:             (7,)  = 7 arm joints
#   observation.environment_state: (7,)  = target joint pose
#   action:                        (7,)  = 7 arm joints

lerobot-train \
    --policy.type=diffusion \
    --dataset.repo_id=lerobot/maniskill_reach \
    --output_dir=outputs/train/maniskill_reach \
    --policy.diffusion_step_embed_dim=128 \
    --policy.num_inference_steps=10 \
    --batch_size=64 \
    --steps=5000 \
    --save_freq=1000 \
    --eval_freq=0 \
    --save_checkpoint=true \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false
