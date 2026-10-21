"""Side-by-side video: scripted demonstration vs a trained policy, from the same start poses.

Needs the sweep sim up with its GUI on a virtual display (teaching repo):
    GUI=1 native/sim_up.sh
For each seed, records the Gazebo window from the moment the episode starts for --clip
seconds: once for docker/sweep_demo.py --phase collect (the demonstrator), once for
eval_gen3_sweep_sim.py (the policy). Both draw the start pose from random.Random(seed),
so each pair starts identically. Pairs are stacked side by side, labelled with the number
of cubes swept, and concatenated in seed order.

    python video_gen3_sweep_sim.py outputs/gen3_sweep_sim_demo_vs_dp.mp4 --seeds 1000 1001 1002
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from gen3_sweep_sim import NUM_CUBES, sweep_demo

HERE = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SCREEN = "1400x900"
TITLE_BAR = 48   # Gazebo window title bar, cropped off


def record(proc: subprocess.Popen, marker: str, out: Path, clip: float, display: str) -> list[str]:
    """Echo proc's output; at the marker line start a clip-long x11grab of the display."""
    ff, lines = None, []
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        print("   ", line, flush=True)
        if ff is None and line.startswith(marker):
            ff = subprocess.Popen([
                "ffmpeg", "-loglevel", "error", "-y", "-f", "x11grab", "-draw_mouse", "0",
                "-video_size", SCREEN, "-framerate", "15", "-i", f"{display}+0,0", "-t", str(clip),
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out),
            ])
    proc.wait()
    if ff is None:
        raise RuntimeError(f"never saw {marker}")
    ff.wait()   # fixed-length clip; the arm just holds still at the end
    return lines


def demo_result(lines: list[str]) -> str:
    # "[hh:mm:ss] /sweep_task_monitor/status -> success=True 0/15 cubes still on the table. ..."
    for line in lines:
        if "sweep_task_monitor/status ->" in line:
            try:
                remaining = int(line.split("success=")[1].split(" ", 1)[1].split("/")[0])
                return f"{NUM_CUBES - remaining}/{NUM_CUBES} cubes swept"
            except (IndexError, ValueError):
                return line.split("-> ", 1)[1]
    return "?"


def policy_result(lines: list[str]) -> str:
    for line in lines:
        if "episode 0: " in line:
            return line.split("episode 0: ", 1)[1]
    return "?"


def label(text: str, bottom: bool = False) -> str:
    text = text.replace(":", r"\:").replace("'", "")
    y = "h-70" if bottom else "20"
    return (f"drawtext=fontfile={FONT}:text='{text}':x=20:y={y}:fontsize=34:fontcolor=white:"
            "box=1:boxcolor=black@0.6:boxborderw=10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: eval's default")
    ap.add_argument("--clip", type=float, default=36.0, help="seconds per clip")
    ap.add_argument("--display", default=os.environ.get("DISP", ":99"))
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="sweep_video_"))
    rows = []
    for seed in args.seeds:
        demo_clip, policy_clip = work / f"demo_{seed}.mp4", work / f"policy_{seed}.mp4"
        print(f"seed {seed}: scripted demonstration", flush=True)
        proc = sweep_demo("--phase", "collect", "--episodes", 1, "--seed", seed,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        demo = demo_result(record(proc, "EPISODE_START", demo_clip, args.clip, args.display))

        print(f"seed {seed}: policy", flush=True)
        cmd = [sys.executable, "-u", str(HERE / "eval_gen3_sweep_sim.py"), "--episodes", "1",
               "--seed", str(seed), "--duration", str(args.clip - 1)]
        if args.checkpoint:
            cmd += ["--checkpoint", str(args.checkpoint)]
        proc = subprocess.Popen(cmd, cwd=HERE, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=1)
        policy = policy_result(record(proc, "ROLLOUT_START", policy_clip, args.clip, args.display))
        print(f"seed {seed}: demo [{demo}]  policy [{policy}]", flush=True)

        row = work / f"pair_{seed}.mp4"
        crop = f"crop=iw:ih-{TITLE_BAR}:0:{TITLE_BAR},scale=960:-2"
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(demo_clip), "-i", str(policy_clip),
            "-filter_complex",
            f"[0:v]{crop},{label(f'Scripted demo  (start seed {seed})')},{label(demo, True)}[a];"
            f"[1:v]{crop},{label(f'Diffusion policy  (start seed {seed})')},{label(policy, True)}[b];"
            "[a][b]hstack[v]",
            "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
            str(row),
        ], check=True)
        rows.append(row)

    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{r}'\n" for r in rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", str(args.out)], check=True)
    print(f"wrote {args.out} (clips in {work})")


if __name__ == "__main__":
    main()
