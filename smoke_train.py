#!/usr/bin/env python3
"""Train DiffusionPolicy on the synthetic gen3_lite dataset — CPU, no ROS, no arm.

Second half of the pure-software pipeline smoke test. This drives lerobot's REAL
training function (`lerobot.scripts.lerobot_train.train`), not a hand-rolled loop, so
it exercises the same code path a real run would.

Why not the ``lerobot-train`` CLI? On Python 3.14 the draccus argument parser crashes
resolving PEP 604 unions (``TypeError: str | None is not callable``) before training
starts. We sidestep the CLI by building ``TrainPipelineConfig`` in Python and calling
the undecorated ``train.__wrapped__(cfg)`` (the ``@parser.wrap()`` decorator only adds
CLI parsing; the wrapped function is the actual trainer).

Usage:
    python smoke_train.py [STEPS] [BATCH_SIZE]
"""
import shutil
import sys
import time
from pathlib import Path

import lerobot_py314_compat  # noqa: F401  (Py3.14 draccus shim; harmless if unused here)
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.scripts.lerobot_train import train

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 16

OUTPUT_DIR = Path("outputs/train/gen3_lite_smoke")


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="kinova_gen3_lite_smoke",
            root="data/lerobot/kinova_gen3_lite_smoke",
            image_transforms=ImageTransformsConfig(),
            video_backend="pyav",
        ),
        policy=DiffusionConfig(device="cpu", push_to_hub=False),
        output_dir=OUTPUT_DIR,
        job_name="gen3_lite_smoke",
        batch_size=BATCH_SIZE,
        steps=STEPS,
        save_freq=STEPS,   # single checkpoint at the end
        log_freq=max(1, STEPS // 10),
        eval_freq=0,       # no env — skip eval
        num_workers=0,     # CPU: avoid dataloader multiprocessing
        wandb=TrainPipelineConfig.__dataclass_fields__["wandb"].default_factory(),
    )
    cfg.wandb.enable = False

    t0 = time.perf_counter()
    # Bypass the draccus CLI wrapper (broken on Py3.14) — call the trainer directly.
    train.__wrapped__(cfg)
    dt = time.perf_counter() - t0

    print(f"\n=== TRAIN DONE: {STEPS} steps in {dt:.1f}s = {dt / STEPS * 1000:.0f} ms/step "
          f"(batch={BATCH_SIZE}, CPU) ===")
    ckpt = OUTPUT_DIR / "checkpoints" / "last" / "pretrained_model"
    print("checkpoint:", ckpt, "exists:", ckpt.exists())


if __name__ == "__main__":
    main()
