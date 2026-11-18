"""Checks of the Gen3 Lite HIL adapter that need a simulated arm with real controllers.

Run inside the Gazebo container (docker/gazebo/run.sh up), never against a real arm:

    docker/gazebo/run.sh exec 'python docker/gazebo/sim_checks.py'

  sustained   20 unit steps in +x from the reset pose: tracking lag, tool speed, and the
              stop at the tool-box face 13 cm away, with no overshoot past it
  descend     30 unit steps in -z: the tool-box floor or the workspace guard stops it, and
              the measured pose never violates the guard
  gripper     close and open through the gripper action (use_gripper=True)
  stale       pause the simulation: observations and commands raise within the limit;
              resume: control recovers
  no_go       a no-go box 11 cm from the tool in +y: driving at it, the arm stops with
              every link outside the margin, and backs away freely
  orientation roll, pitch and yaw stay frozen through a long diagonal move; a 20 degree
              yaw set-up turn happens on the spot and the new orientation then holds
  runaway     (``--runaway``) send +x forever; for the kill -9 test in README.md

Exits non-zero if a check fails.
"""

import argparse
import subprocess
import sys
import time

import numpy as np

from lerobot_robot_ros import Gen3LiteHil, Gen3LiteHilConfig, StaleObservationError

RESET = [-0.255, -0.607, 2.202, 0.249, -1.469, -1.227]  # tool at (0.30, 0.16, 0.22)
PERIOD = 0.1
ZERO = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0}


def tool(obs) -> np.ndarray:
    return np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]])


class Session:
    def __init__(self, robot: Gen3LiteHil):
        self.robot = robot
        self.failures: list[str] = []

    def check(self, ok: bool, message: str) -> None:
        print(("  ok    " if ok else "  FAIL  ") + message, flush=True)
        if not ok:
            self.failures.append(message)

    def step(self, **action) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        applied = self.robot.send_action({**ZERO, **action})
        obs = self.robot.get_observation()
        time.sleep(max(PERIOD - (time.perf_counter() - t0), 0.0))
        return applied, obs

    def settle(self, steps: int = 8) -> dict:
        obs = None
        for _ in range(steps):
            _, obs = self.step()
        return obs

    def to_reset(self) -> np.ndarray:
        self.robot.go_home(RESET)
        return tool(self.settle(3))


def sustained(s: Session) -> None:
    print("\nsustained: 20 unit steps in +x (tool box face at x = 0.43)")
    cfg = s.robot.config
    start = s.to_reset()
    face = cfg.ee_bounds_max[0]
    clipped0, capped0 = s.robot.clipped_commands, s.robot.joint_capped_commands
    target, prev, lags, speeds, xs = start.copy(), start.copy(), [], [], []
    for _ in range(20):
        applied, obs = s.step(delta_x=1.0)
        target = target + np.array([applied[k] for k in ZERO]) * cfg.ee_step_size_m
        now = tool(obs)
        lags.append(np.linalg.norm(target - now))
        speeds.append(np.linalg.norm(now - prev) / PERIOD)
        xs.append(now[0])
        prev = now
    for _ in range(8):
        _, obs = s.step()
        xs.append(tool(obs)[0])
    final = tool(obs)
    print(
        f"  lag mm: max {max(lags) * 1e3:.1f}, mean over steps 3-10 {np.mean(lags[2:10]) * 1e3:.1f}; "
        f"tool speed m/s: max {max(speeds):.3f}; limit-clipped steps {s.robot.clipped_commands - clipped0}, "
        f"joint-capped steps {s.robot.joint_capped_commands - capped0}"
    )
    s.check(max(lags) <= cfg.max_ee_lead_m + 1e-3, f"lag stays within the {cfg.max_ee_lead_m * 100:.0f} cm lead limit")
    s.check(max(speeds) <= cfg.max_ee_speed_m_s * 1.25, f"tool speed stays near {cfg.max_ee_speed_m_s} m/s")
    s.check(abs(final[0] - face) < 0.003, f"tool stops at the box face (x = {final[0]:.4f})")
    s.check(max(xs) <= face + 0.003, f"no overshoot past the face (max x = {max(xs):.4f})")
    s.check(abs(final[1] - start[1]) < 0.004 and abs(final[2] - start[2]) < 0.004, "y and z held along the way")


def descend(s: Session) -> None:
    print("\ndescend: 30 unit steps in -z (tool box floor at z = 0.04; table plane guarded)")
    cfg = s.robot.config
    s.to_reset()
    blocked0 = s.robot.blocked_commands
    zs, worst = [], 0.0
    for _ in range(30):
        _, obs = s.step(delta_z=-1.0)
        zs.append(tool(obs)[2])
        arm, grip = s.robot._measured()
        worst = max(worst, s.robot._guard.assess(arm, grip)[0])
    obs = s.settle()
    final = tool(obs)
    blocks = s.robot.blocked_commands - blocked0
    print(f"  final z {final[2]:.4f} m, lowest z {min(zs):.4f} m, guard blocks {blocks}")
    s.check(min(zs) >= cfg.ee_bounds_min[2] - 0.003, "tool never below the box floor")
    s.check(worst == 0.0, f"measured pose never violated the workspace guard (worst {worst * 1e3:.2f} mm)")
    s.check(final[2] < 0.12, "the arm did descend (stopped by the floor or the guard, not earlier)")
    _, obs = s.step(delta_z=1.0)
    obs = s.settle(4)
    s.check(tool(obs)[2] > final[2] + 0.005, "moving back up works straight away")


def gripper(s: Session) -> None:
    print("\ngripper: close, then open, through the gripper action")
    iface = s.robot.config.ros2_interface
    s.to_reset()
    for _ in range(25):
        _, obs = s.step(gripper=0)
    closed = obs["gripper.pos"]
    for _ in range(25):
        _, obs = s.step(gripper=2)
    opened = obs["gripper.pos"]
    print(f"  finger joint after close {closed:.3f}, after open {opened:.3f} (range {iface.gripper_open_position}"
          f" .. {iface.gripper_close_position})")
    s.check(closed > 0.6 * iface.gripper_close_position, "gripper closed")
    s.check(opened < 0.1 * iface.gripper_close_position, "gripper opened")


def pause_sim(pause: bool) -> None:
    subprocess.run(
        ["gz", "service", "-s", "/world/empty/control", "--reqtype", "gz.msgs.WorldControl", "--reptype",
         "gz.msgs.Boolean", "--timeout", "3000", "--req", f"pause: {'true' if pause else 'false'}"],
        check=True, capture_output=True,
    )


def stale(s: Session) -> None:
    print("\nstale: pause the simulation while commanding")
    limit = s.robot.config.max_obs_age_s
    s.to_reset()
    for _ in range(3):
        s.step(delta_y=1.0)
    stops0 = s.robot.stale_stops
    pause_sim(True)
    paused_at = time.monotonic()
    try:
        raised_after = None
        while time.monotonic() - paused_at < 3.0:
            try:
                s.robot.send_action({**ZERO, "delta_y": 1.0})
                time.sleep(0.02)
            except StaleObservationError:
                raised_after = time.monotonic() - paused_at
                break
        s.check(raised_after is not None, "a command was refused after the simulation stopped")
        if raised_after is not None:
            print(f"  refused {raised_after * 1e3:.0f} ms after the pause (limit {limit * 1e3:.0f} ms)")
            s.check(raised_after < limit + 0.25, "refusal came promptly")
        try:
            s.robot.get_observation()
            s.check(False, "get_observation raised on stale state")
        except StaleObservationError:
            s.check(True, "get_observation raised on stale state")
    finally:
        pause_sim(False)
    time.sleep(0.5)
    before = tool(s.robot.get_observation())
    for _ in range(4):
        s.step(delta_y=-1.0)
    after = tool(s.settle(4))
    s.check(after[1] < before[1] - 0.02, "control recovered after the simulation resumed")
    s.check(s.robot.stale_stops > stops0, f"stale stops counted ({s.robot.stale_stops - stops0})")


def no_go(s: Session) -> None:
    print("\nno_go: a box 11 cm ahead of the tool in +y")
    from lerobot_robot_ros.config_gen3_lite_ros import NoGoBox

    robot = s.robot
    start = s.to_reset()
    box_y = start[1] + 0.11
    robot.set_no_go_boxes([NoGoBox(name="camera", min=[0.20, box_y, 0.12], max=[0.40, box_y + 0.06, 0.32])])
    try:
        blocked0 = robot.blocked_commands
        worst, ys = 0.0, []
        for _ in range(20):
            _, obs = s.step(delta_y=1.0)
            ys.append(tool(obs)[1])
            arm, grip = robot._measured()
            worst = max(worst, robot._guard.assess(arm, grip)[0])
        final = tool(s.settle())
        print(f"  tool stopped at y = {final[1]:.4f} (start {start[1]:.4f}, box face {box_y:.4f}), "
              f"guard blocks {robot.blocked_commands - blocked0}")
        s.check(robot.blocked_commands - blocked0 > 5, "the guard held the arm")
        s.check(worst == 0.0, f"no link came inside the {robot.config.no_go_margin * 100:.0f} cm margin (worst {worst * 1e3:.2f} mm)")
        s.check(start[1] + 0.01 < final[1] < box_y - robot.config.no_go_margin, "it moved towards the box and stopped short")
        before = robot.blocked_commands
        for _ in range(4):
            s.step(delta_y=-1.0)
        s.check(robot.blocked_commands == before and tool(s.settle(4))[1] < final[1] - 0.02, "backing away is not blocked")
    finally:
        robot.set_no_go_boxes([])


def orientation(s: Session) -> None:
    print("\norientation: frozen through a diagonal move, then a 20 degree yaw on the spot")
    robot = s.robot

    def measured_rot():
        return robot._kin.fk(robot._measured()[0])[1]

    def angle(a, b):
        return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))

    robot.freeze_orientation_at(RESET)
    start = s.to_reset()
    frozen = robot._kin.fk(RESET)[1]
    worst = 0.0
    for i in range(24):
        direction = 1.0 if i < 12 else -1.0
        s.step(delta_x=direction, delta_y=-0.5 * direction, delta_z=0.7 * direction)
        worst = max(worst, angle(frozen, measured_rot()))
    s.settle()
    print(f"  worst deviation while moving {worst:.2f} deg, after settling {angle(frozen, measured_rot()):.3f} deg")
    s.check(worst < 2.0, "orientation stays within 2 degrees while moving")
    s.check(angle(frozen, measured_rot()) < 0.3, "and returns to the frozen orientation at rest")

    start = tool(s.settle())  # where the out-and-back move actually ended
    step = np.radians(1.5)
    c, sn = np.cos(-step), np.sin(-step)
    yaw = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    rejections = robot.ik_rejections
    for _ in range(int(round(20 / 1.5))):
        robot.set_held_orientation(yaw @ robot.held_orientation())
        s.step()
    here = tool(s.settle())
    turned = angle(frozen, measured_rot())
    print(f"  turned {turned:.1f} deg, tool moved {np.linalg.norm(here - start) * 1e3:.1f} mm, "
          f"IK rejections {robot.ik_rejections - rejections}")
    s.check(abs(turned - 19.5) < 1.0, "the yaw turn was carried out")
    s.check(np.linalg.norm(here - start) < 0.002, "on the spot (tool within 2 mm)")
    new = robot.held_orientation()
    for _ in range(8):
        s.step(delta_z=1.0)
    s.settle()
    s.check(angle(new, measured_rot()) < 0.3, "the new orientation holds while the tool moves")
    robot.freeze_orientation_at(RESET)


def runaway(robot: Gen3LiteHil) -> None:
    robot.go_home(RESET)
    print("runaway: sending +y forever; kill this process", flush=True)
    while True:
        t0 = time.perf_counter()
        robot.send_action({**ZERO, "delta_y": 0.3})
        time.sleep(max(PERIOD - (time.perf_counter() - t0), 0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runaway", action="store_true")
    parser.add_argument("--only", choices=["sustained", "descend", "gripper", "stale", "no_go", "orientation"])
    args = parser.parse_args()

    clock = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True, timeout=30).stdout.split()
    if "/clock" not in clock:
        print("No /clock topic: this is not a simulation. Refusing to run.")
        return 2

    cfg = Gen3LiteHilConfig(use_sim_time=True, use_gripper=True)
    robot = Gen3LiteHil(cfg)
    robot.connect()
    if args.runaway:
        runaway(robot)
    session = Session(robot)
    try:
        for check in (sustained, descend, gripper, stale, no_go, orientation):
            if args.only in (None, check.__name__):
                check(session)
        robot.go_home(RESET)
    finally:
        robot.disconnect()
    print(f"\nlimit counters: {robot.counters()}")
    print("\nSIM CHECKS " + ("PASSED" if not session.failures else f"FAILED: {session.failures}"))
    return 1 if session.failures else 0


if __name__ == "__main__":
    sys.exit(main())
