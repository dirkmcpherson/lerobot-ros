#!/bin/bash
source /home/james/miniconda3/bin/activate lerobot-ros

# Explicitly set dataset path if needed, or assume it's in ~/.cache/huggingface/lerobot or local data/
# Since we saved to 'data/lerobot/kinova_gen3_reach', we might need to tell lerobot-train where to look 
# OR we push to hub OR we link it.
# lerobot-train handles local datasets if we pass the root.

# For now, let's assume we can point to it.
# The previous script saved to `data/lerobot/kinova_gen3_reach` (relative to current dir).
# lerobot expects `lerobot/kinova_gen3_reach` repo id.
# If we set LEROBOT_HOME to `data`, it might find it.

export HF_LEROBOT_HOME=$(pwd)/data

# Training command
# We use diffusion policy.
# We need to specify fps, policy params, etc.
# We rely on hydra configs usually.
# Let's try to override defaults via command line.

export HF_HUB_OFFLINE=1

lerobot-train \
    --policy.type=diffusion \
    --dataset.repo_id=lerobot/kinova_gen3_reach \
    --output_dir=outputs/train/kinova_reach \
    --policy.diffusion_step_embed_dim=128 \
    --policy.num_inference_steps=10 \
    --batch_size=8 \
    --steps=5000 \
    --save_freq=1000 \
    --eval_freq=1000 \
    --save_checkpoint=true \
    --policy.device=cuda \
    --policy.repo_id=lerobot/kinova_gen3_reach_policy \
    --policy.push_to_hub=false \
    --wandb.enable=false > training.log 2>&1
