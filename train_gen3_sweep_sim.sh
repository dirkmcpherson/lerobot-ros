#!/bin/bash
# Diffusion policy on the Gazebo sweep task (record_gen3_sweep_sim.py).
#   observation.state (7) + observation.environment_state (45 = 15 cubes xyz) -> action (7)
# DDIM with 10 inference steps keeps select_action well under the 10 Hz control period.
# Extra lerobot-train flags pass through, e.g. bash train_gen3_sweep_sim.sh --steps=5000
source /home/james/miniconda3/bin/activate lerobot-ros
cd "$(dirname "$0")"

export HF_LEROBOT_HOME=$(pwd)/data
export HF_HUB_OFFLINE=1

lerobot-train \
    --policy.type=diffusion \
    --dataset.repo_id=lerobot/gen3_sweep_sim \
    --output_dir=outputs/train/gen3_sweep_sim \
    --policy.noise_scheduler_type=DDIM \
    --policy.num_inference_steps=10 \
    --batch_size=64 \
    --steps=20000 \
    --save_freq=10000 \
    --log_freq=500 \
    --eval_freq=0 \
    --save_checkpoint=true \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    "$@"
