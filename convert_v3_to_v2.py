"""Convert a LeRobot v3 dataset to v2.1 layout.

v3 packs all episodes into a single parquet + per-camera concatenated MP4s.
v2 expects one parquet per episode and one MP4 per (episode, camera).

Usage:
    python convert_v3_to_v2.py --src data/lerobot/kinova_gen3_lite_smoke \
                               --dst data/lerobot/kinova_gen3_lite_smoke_v2
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pandas as pd

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()

    src, dst = args.src, args.dst
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "videos" / "chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)

    info = json.loads((src / "meta" / "info.json").read_text())
    fps = info["fps"]
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    # --- Per-episode parquets ---
    df = pd.read_parquet(src / "data" / "chunk-000" / "file-000.parquet")
    ep_meta_df = pd.read_parquet(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    n_eps = len(ep_meta_df)

    for ep_idx in range(n_eps):
        ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
        out = dst / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        ep_df.to_parquet(out, index=False)
        print(f"  wrote {out.name}  ({len(ep_df)} frames)")

    # --- Per-episode MP4s ---
    for vk in video_keys:
        (dst / "videos" / "chunk-000" / vk).mkdir(parents=True, exist_ok=True)
        src_mp4 = src / "videos" / vk / "chunk-000" / "file-000.mp4"

        for ep_idx in range(n_eps):
            row = ep_meta_df[ep_meta_df["episode_index"] == ep_idx].iloc[0]
            t0 = row[f"videos/{vk}/from_timestamp"]
            t1 = row[f"videos/{vk}/to_timestamp"]
            out_mp4 = dst / "videos" / "chunk-000" / vk / f"episode_{ep_idx:06d}.mp4"
            cmd = [
                FFMPEG, "-y", "-loglevel", "error",
                "-i", str(src_mp4),
                "-ss", f"{t0}", "-to", f"{t1}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-r", str(fps), "-an", str(out_mp4),
            ]
            subprocess.run(cmd, check=True)
            print(f"  wrote {out_mp4.relative_to(dst)}")

    # --- meta/info.json (v2.1 schema) ---
    v2_info = {
        "codebase_version": "v2.1",
        "robot_type": info.get("robot_type"),
        "total_episodes": n_eps,
        "total_frames": int(df.shape[0]),
        "total_tasks": info["total_tasks"],
        "total_videos": n_eps * len(video_keys),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": info["features"],
    }
    (dst / "meta" / "info.json").write_text(json.dumps(v2_info, indent=4))

    # --- meta/episodes.jsonl ---
    with open(dst / "meta" / "episodes.jsonl", "w") as f:
        for ep_idx in range(n_eps):
            row = ep_meta_df[ep_meta_df["episode_index"] == ep_idx].iloc[0]
            tasks = list(row["tasks"]) if hasattr(row["tasks"], "__iter__") else [str(row["tasks"])]
            f.write(json.dumps({
                "episode_index": int(ep_idx),
                "tasks": tasks,
                "length": int(row["length"]),
            }) + "\n")

    # --- meta/tasks.jsonl ---
    tasks_df = pd.read_parquet(src / "meta" / "tasks.parquet")
    with open(dst / "meta" / "tasks.jsonl", "w") as f:
        if "task" in tasks_df.columns:
            for _, row in tasks_df.iterrows():
                f.write(json.dumps({"task_index": int(row["task_index"]), "task": row["task"]}) + "\n")
        else:
            # tasks.parquet doesn't have a "task" column; pull from episodes.tasks
            seen = {}
            for _, row in ep_meta_df.iterrows():
                for t in row["tasks"]:
                    if t not in seen:
                        seen[t] = len(seen)
            for t, ti in seen.items():
                f.write(json.dumps({"task_index": ti, "task": t}) + "\n")

    # --- meta/stats.json — copy if present ---
    src_stats = src / "meta" / "stats.json"
    if src_stats.exists():
        shutil.copy(src_stats, dst / "meta" / "stats.json")

    print(f"\nDone. v2 dataset at: {dst}")


if __name__ == "__main__":
    main()
