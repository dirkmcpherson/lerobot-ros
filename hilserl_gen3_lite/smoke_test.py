"""Supervised hardware smoke test for the Gen3 Lite HIL adapter. THIS MOVES THE ARM.

The operator runs it, with the driver up, the arm homed and a hand on the emergency stop:

    . gen3_lite_env.sh
    python -m hilserl_gen3_lite.smoke_test --robot.type=gen3_lite_hil --run.confirm_motion=true

What it does, at 10 Hz, never further than ``leg_cm`` from the start pose:
  1. connects (homes if ``robot.home_on_connect``), moves to the reset pose with the
     guarded joint move, prints the limits;
  2. runs observe -> act(zero) -> observe for ``idle_steps`` steps and checks the loop
     period and that no observation went stale and the arm did not move;
  3. drives +x, -x, +y, -y, +z, -z by ``leg_cm`` and back, printing commanded against
     measured tool position after every step;
  4. stops sending actions and checks that the watchdog commands a hold;
  5. disconnects.

It exits non-zero if tracking error, loop timing or any check is out of tolerance.
Pass ``--robot.type=gen3_lite_hil_fake --robot.urdf_path=tests/fixtures/gen3_lite.urdf``
to rehearse it without an arm.
"""

import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.robot_utils import precise_sleep

from .settings import RunSettings, pop_run_settings, require_motion_confirmation

_SETTINGS = RunSettings()

DEFAULT_RESET = [-0.255, -0.607, 2.202, 0.249, -1.469, -1.227]  # tool at (0.30, 0.16, 0.22)
AXES = (("delta_x", 0), ("delta_y", 1), ("delta_z", 2))


@dataclass
class SmokeTestConfig:
    robot: RobotConfig
    # Length of each out-and-back leg (cm). Each step is at most robot.max_ee_step_m.
    leg_cm: float = 3.0
    idle_steps: int = 30
    # Zero-action steps after each leg, to let the arm settle before measuring.
    settle_steps: int = 5
    # Pass/fail tolerances.
    max_tracking_error_mm: float = 8.0  # |target - measured| after settling
    max_period_error_ms: float = 30.0  # worst loop period deviation from the control period
    max_idle_drift_mm: float = 1.5


def tool(obs) -> np.ndarray:
    return np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]])


def run(cfg: SmokeTestConfig) -> bool:
    robot_cfg = cfg.robot
    if not hasattr(robot_cfg, "ee_limits"):
        raise SystemExit("smoke_test needs --robot.type=gen3_lite_hil (or gen3_lite_hil_fake)")
    require_motion_confirmation(robot_cfg, _SETTINGS)
    period = robot_cfg.control_period_s
    step_m = min(robot_cfg.ee_step_size_m, robot_cfg.max_ee_step_m)
    leg_steps = max(1, int(round(cfg.leg_cm / 100 / step_m)))
    print(f"Plan: 6 legs of {leg_steps} x {step_m * 100:.1f} cm at {1 / period:.0f} Hz, then a watchdog check.")
    print("Starting in 5 s. Ctrl+C or the emergency stop aborts.", flush=True)
    time.sleep(5)

    robot = make_robot_from_config(robot_cfg)
    ok = True
    try:
        robot.connect()
        reset = robot_cfg.reset_joint_positions or DEFAULT_RESET
        print(f"Moving to the reset pose {reset} (guarded joint move)...", flush=True)
        robot.freeze_orientation_at(list(reset))
        robot.go_home(list(reset))
        start = tool(robot.get_observation())
        lo, hi = np.array(robot_cfg.ee_bounds_min), np.array(robot_cfg.ee_bounds_max)
        print(f"Start tool position {np.round(start, 4).tolist()} m")
        if np.any(start - lo < cfg.leg_cm / 100) or np.any(hi - start < cfg.leg_cm / 100):
            print("WARNING: a leg will reach the tool box; expect clipped steps on that leg.")

        def step(action: dict) -> tuple[dict, dict, float]:
            t0 = time.perf_counter()
            applied = robot.send_action(action)
            obs = robot.get_observation()
            precise_sleep(max(period - (time.perf_counter() - t0), 0.0))
            return applied, obs, time.perf_counter() - t0

        # ---- 2. idle loop: timing, staleness, no drift
        periods, obs = [], None
        for _ in range(cfg.idle_steps):
            _, obs, took = step({"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0})
            periods.append(took)
        periods = np.array(periods)
        drift_mm = float(np.linalg.norm(tool(obs) - start) * 1000)
        ages = {k: (None if v is None else round(v * 1000, 1)) for k, v in robot.observation_ages().items()}
        worst_ms = float(np.abs(periods - period).max() * 1000)
        print(
            f"Idle loop: {cfg.idle_steps} steps, mean period {periods.mean() * 1000:.1f} ms, worst deviation "
            f"{worst_ms:.1f} ms, drift {drift_mm:.2f} mm, observation ages (ms) {ages}"
        )
        if worst_ms > cfg.max_period_error_ms:
            ok = False
            print(f"FAIL: loop period deviates by more than {cfg.max_period_error_ms} ms")
        if drift_mm > cfg.max_idle_drift_mm:
            ok = False
            print(f"FAIL: arm drifted {drift_mm:.2f} mm on zero actions")

        # ---- 3. six legs, out and back
        print("\n leg   step   commanded target (m)         measured (m)                 error mm")
        for name, axis in AXES:
            for sign in (1.0, -1.0):
                origin = tool(robot.get_observation())
                target = origin.copy()
                for direction in (sign, -sign):  # out, then back
                    for i in range(leg_steps):
                        action = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, name: direction}
                        applied, obs, _ = step(action)
                        target = target + np.array([applied[k] for k, _ in AXES]) * robot_cfg.ee_step_size_m
                        err = np.linalg.norm(target - tool(obs)) * 1000
                        print(
                            f" {'+' if sign > 0 else '-'}{name[-1]}{'>' if direction == sign else '<'}  {i + 1:3d}   "
                            f"{np.round(target, 4)}   {np.round(tool(obs), 4)}   {err:6.2f}"
                        )
                    for _ in range(cfg.settle_steps):
                        _, obs, _ = step({"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0})
                    err = float(np.linalg.norm(target - tool(obs)) * 1000)
                    moved = (tool(obs) - origin)[axis] * 100
                    print(f"      settled: error {err:.2f} mm, displacement along {name[-1]} {moved:+.2f} cm")
                    if err > cfg.max_tracking_error_mm:
                        ok = False
                        print(f"FAIL: tracking error above {cfg.max_tracking_error_mm} mm")
        end_err = float(np.linalg.norm(tool(robot.get_observation()) - start) * 1000)
        print(f"\nReturned to within {end_err:.2f} mm of the start pose.")

        # ---- 4. watchdog
        trips = robot.watchdog_trips
        step({"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0})
        time.sleep(period * (robot_cfg.watchdog_periods + 3))
        if robot.watchdog_trips == trips + 1:
            print("Watchdog: hold commanded after the actions stopped. OK")
        else:
            ok = False
            print(f"FAIL: watchdog trips went from {trips} to {robot.watchdog_trips}, expected one more")

        print(f"\nLimit counters: {robot.counters()}")
        if robot.ik_rejections or robot.stale_stops:
            ok = False
            print("FAIL: IK rejections or stale observations occurred")
    except KeyboardInterrupt:
        ok = False
        print("\nAborted by the operator.")
    finally:
        robot.disconnect()  # commands a hold first
    print("\nSMOKE TEST " + ("PASSED" if ok else "FAILED"))
    return ok


@parser.wrap()
def smoke_test_cli(cfg: SmokeTestConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    sys.exit(0 if run(cfg) else 1)


def main() -> None:
    global _SETTINGS
    _SETTINGS, sys.argv = pop_run_settings()
    import lerobot_robot_ros  # noqa: F401  (registers the robot types before parsing)

    smoke_test_cli()


if __name__ == "__main__":
    main()
