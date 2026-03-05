#!/bin/bash
source /home/james/miniconda3/bin/activate lerobot-ros

# Point lerobot at our local dataset directory
export HF_LEROBOT_HOME=$(pwd)/data
export HF_HUB_OFFLINE=1

# Dataset features:
#   observation.state:             (7,)  = 6 arm joints + 1 gripper
#   observation.environment_state: (6,)  = green xyz + red xyz
#   action:                        (7,)  = 6 arm joints + 1 gripper

lerobot-train \
    --policy.type=diffusion \
    --dataset.repo_id=lerobot/kinova_gen3_lite_teleop \
    --output_dir=outputs/train/cube_stacking \
    --policy.diffusion_step_embed_dim=128 \
    --policy.num_inference_steps=10 \
    --batch_size=8 \
    --steps=5000 \
    --save_freq=1000 \
    --eval_freq=0 \
    --save_checkpoint=true \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false
