"""Roll out a trained HIL-SERL checkpoint in Genesis, hands off, and save a video per episode.

python docs/genesis_results/eval_policy.py --config_path=config/hilserl/genesis_cup_train.json \
    --eval.checkpoint=runs/genesis_cup_s0/learner/checkpoints/040000 --eval.out=runs/genesis_cup_s0/review_040000 --eval.episodes=12
Each video: a high-res review camera (not seen by the policy) beside the policy's own 128x128 view.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---- our own args, removed before lerobot parses the rest
ev = {"checkpoint": None, "out": None, "episodes": 10, "seed": 100}
rest = []
for a in sys.argv:
    if a.startswith("--eval."):
        k, _, v = a[len("--eval."):].partition("=")
        ev[k] = type(ev[k])(v) if ev[k] is not None else v
    else:
        rest.append(a)
sys.argv = rest
OUT = Path(ev["out"]); OUT.mkdir(parents=True, exist_ok=True)

# ---- review camera: added to the Genesis scene beside the policy's camera, rendered in the sim thread
import genesis as gs  # noqa: E402

REVIEW = {"frame": None, "cam": None, "n": 0}
_orig_build = gs.Scene.build


def _build(self, *a, **k):
    REVIEW["cam"] = self.add_camera(res=(480, 360), pos=(0.75, -0.45, 0.45), lookat=(0.30, 0.16, 0.10), fov=45, GUI=False)
    out = _orig_build(self, *a, **k)
    return out


gs.Scene.build = _build

from lerobot_robot_ros import genesis_backend as gb  # noqa: E402

_orig_frames_render = None
_orig_fc = gb.fingertip_center


def _fc_and_review(*a, **k):  # called once per sim step in the sim thread; render the review cam at 10 Hz
    REVIEW["n"] += 1
    if REVIEW["cam"] is not None and REVIEW["n"] % 10 == 0:
        REVIEW["frame"] = np.asarray(REVIEW["cam"].render(rgb=True)[0], dtype=np.uint8).copy()
    return _orig_fc(*a, **k)


gb.fingertip_center = _fc_and_review

import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from hilserl_gen3_lite import compat  # noqa: E402

compat.apply()
import lerobot_robot_ros  # noqa: E402,F401
import lerobot_teleoperator_devices  # noqa: E402,F401
from lerobot.configs import parser  # noqa: E402
from lerobot.rl.train_rl import TrainRLServerPipelineConfig  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.rl.gym_manipulator import reset_and_build_transition, step_env_and_process_transition  # noqa: E402
from lerobot.processor import TransitionKey  # noqa: E402
from lerobot.teleoperators.utils import TeleopEvents  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hilserl_gen3_lite.env import make_processors, make_robot_env  # noqa: E402
from lerobot_teleoperator_devices.scripted_pick_place import PICK_PLACE_SOURCE  # noqa: E402


def label(img, lines, size):
    im = Image.fromarray(img).resize(size, Image.NEAREST if img.shape[0] < 200 else Image.BILINEAR)
    d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        d.rectangle([4, 4 + 16 * i, 8 + 7 * len(t), 18 + 16 * i], fill=(0, 0, 0))
        d.text((6, 5 + 16 * i), t, fill=(255, 255, 255))
    return np.asarray(im)


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    cfg.validate()
    cfg.env.teleop.takeovers = []  # hands off: the scripted operator only scores success
    cfg.env.teleop.always_intervene = False
    cfg.env.robot.genesis_seed = ev["seed"]
    env, teleop = make_robot_env(cfg.env)
    env_proc, act_proc = make_processors(env, teleop, cfg.env, cfg.policy.device)
    torch.manual_seed(ev["seed"])
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    sd = load_file(str(Path(ev["checkpoint"]) / "pretrained_model" / "model.safetensors"))
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} tensors; missing {len(missing)} {missing[:5]}; unexpected {len(unexpected)} {unexpected[:5]}", flush=True)
    print('encoder shared with critic:', policy.encoder_critic is policy.actor.encoder, flush=True)
    policy = policy.to(cfg.policy.device).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg.policy, dataset_stats=cfg.policy.dataset_stats)
    robot = env.robot
    results = []
    try:
        for ep in range(ev["episodes"]):
            tr = reset_and_build_transition(env, env_proc, act_proc)
            start = robot.backend.task_state()["object_position"].round(3).tolist()
            frames, success, steps, t0 = [], False, 0, time.monotonic()
            while True:
                t_step = time.perf_counter()
                obs = {k: v for k, v in tr[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features}
                with torch.no_grad():
                    a = policy.select_action(batch=pre.process_observation(obs))
                    if cfg.policy.num_discrete_actions is not None:
                        c = post.process_action(a[..., :-1]); a = torch.cat([c, a[..., -1:].to(device=c.device, dtype=c.dtype)], dim=-1)
                    else:
                        a = post.process_action(a)
                tr = step_env_and_process_transition(env=env, transition=tr, action=a, env_processor=env_proc, action_processor=act_proc)
                steps += 1
                assert not tr[TransitionKey.INFO].get(TeleopEvents.IS_INTERVENTION, False)
                success = success or float(tr[TransitionKey.REWARD]) > 0
                st = robot.backend.task_state()
                view = robot.backend._frames["front"].copy()
                g = gb.theta_to_gripper(robot.backend._theta, robot.backend._g_open, robot.backend._g_close)
                act = a.detach().cpu().numpy().reshape(-1)
                rv = REVIEW["frame"] if REVIEW["frame"] is not None else np.zeros((360, 480, 3), np.uint8)
                left = label(rv, [f"ep {ep}  step {steps:3d}  t {steps/10:4.1f}s", f"grip {g:.2f}  cmd {['close','stay','open'][int(round(act[-1]))] if act.size == 4 else ''}",
                                   f"obj z {st['object_position'][2]:.3f}  on_shelf {st['on_shelf']}", "SUCCESS" if success else ""], (480, 360))
                right = label(view, ["policy view 128x128"], (360, 360))
                frames.append(np.concatenate([left, right], axis=1))
                if tr.get(TransitionKey.DONE, False) or tr.get(TransitionKey.TRUNCATED, False):
                    break
                time.sleep(max(1.0 / cfg.env.fps - (time.perf_counter() - t_step), 0.0))  # as upstream's actor
            for _ in range(10):
                frames.append(frames[-1])
            name = f"ep{ep:02d}_{'success' if success else 'fail'}.mp4"
            imageio.mimwrite(OUT / name, frames, fps=10, macro_block_size=8)
            r = {"episode": ep, "success": success, "steps": steps, "object_start": start,
                 "object_final": st["object_position"].round(3).tolist(), "object_upright": round(float(st["object_upright"]), 3),
                 "on_shelf": bool(st["on_shelf"]), "video": name, "wall_s": round(time.monotonic() - t0, 1),
                 "sim_late_steps": robot.backend.late_steps, "sim_lag_s": round(robot.backend.realtime_lag_s, 3)}
            results.append(r)
            print(json.dumps(r), flush=True)
    finally:
        print("counters", robot.counters(), flush=True)
        (OUT / "results.json").write_text(json.dumps({"checkpoint": ev["checkpoint"], "episodes": results,
                                                       "successes": sum(r["success"] for r in results),
                                                       "counters": robot.counters()}, indent=2))
        env.close()
        teleop.disconnect()


if __name__ == "__main__":
    main()
