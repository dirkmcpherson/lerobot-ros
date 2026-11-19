"""The Genesis backend's pure helpers and the scripted pick-and-place planner (no Genesis needed)."""

import numpy as np
import pytest

from lerobot_robot_ros.genesis_backend import (
    THETA_CLOSED,
    THETA_OPEN,
    finger_targets,
    fingertip_center,
    gripper_to_theta,
    interpolate,
    on_shelf,
    theta_to_gripper,
)
from lerobot_teleoperator_devices.config_scripted_pick_place import ScriptedPickPlaceTeleopConfig
from lerobot_teleoperator_devices.scripted_pick_place import CLOSE, OPEN, plan_step

OPEN_POS, CLOSE_POS = 0.0, 0.85  # Gen3 Lite gripper convention of the robot class


def test_gripper_mapping_round_trip_and_direction():
    # The robot class's "open" must open the Genesis fingers (measured: 0.96 rad = 10.5 cm apart).
    assert gripper_to_theta(OPEN_POS, OPEN_POS, CLOSE_POS) == pytest.approx(THETA_OPEN)
    assert gripper_to_theta(CLOSE_POS, OPEN_POS, CLOSE_POS) == pytest.approx(THETA_CLOSED)
    assert THETA_OPEN > THETA_CLOSED
    for g in (0.0, 0.2, 0.5, 0.85):
        assert theta_to_gripper(gripper_to_theta(g, OPEN_POS, CLOSE_POS), OPEN_POS, CLOSE_POS) == pytest.approx(g)
    assert gripper_to_theta(2.0, OPEN_POS, CLOSE_POS) == pytest.approx(THETA_CLOSED)  # clipped


def test_finger_targets_follow_the_urdf_mimic():
    right, left, tip_r, tip_l = finger_targets(0.5)
    assert left == -right and tip_r == tip_l == pytest.approx(-0.676 * 0.5 + 0.149)


def test_fingertip_center_uses_the_lowest_vertices():
    right = np.array([[0.0, -0.03, 0.10], [0.0, -0.03, 0.05], [0.0, -0.02, 0.045]])
    left = np.array([[0.0, 0.03, 0.10], [0.0, 0.03, 0.05], [0.0, 0.02, 0.045]])
    c = fingertip_center([right, left], band=0.01)
    assert c[1] == pytest.approx(0.0) and c[2] == pytest.approx(0.0475)


def test_interpolate_reaches_goal_after_duration():
    a, b = np.zeros(2), np.ones(2)
    assert np.allclose(interpolate(a, b, 0.0, 0.2, 0.1), 0.5)
    assert np.allclose(interpolate(a, b, 0.0, 0.2, 5.0), 1.0)
    assert np.allclose(interpolate(a, b, 0.0, 0.0, 0.0), 1.0)


SHELF, SIZE, H = [0.32, 0.24, 0.04], [0.12, 0.10, 0.08], 0.10


def test_on_shelf():
    assert on_shelf([0.32, 0.24, 0.13], 1.0, SHELF, SIZE, H)
    assert not on_shelf([0.32, 0.24, 0.20], 1.0, SHELF, SIZE, H)  # still carried above it
    assert not on_shelf([0.32, 0.10, 0.05], 1.0, SHELF, SIZE, H)  # on the table
    assert not on_shelf([0.32, 0.24, 0.13], 0.5, SHELF, SIZE, H)  # tipped


def _task(obj, on=False):
    return {"object_position": np.asarray(obj, float), "object_height": H, "shelf_position": np.asarray(SHELF),
            "shelf_size": np.asarray(SIZE), "on_shelf": on}


def test_planner_phases_through_a_pick_and_place():
    cfg = ScriptedPickPlaceTeleopConfig(grasp_dz=-0.015)
    obj = [0.32, 0.08, 0.05]
    # far above and to the side: move over it without descending, gripper open
    tgt, g, phase, _ = plan_step(cfg, [0.30, 0.15, 0.18], 0.0, _task(obj), 0)
    assert phase == "approach" and g == OPEN and np.allclose(tgt, [0.32, 0.08, 0.18])
    # close above it: approach at the pre-grasp height
    tgt, _, phase, _ = plan_step(cfg, [0.335, 0.08, 0.18], 0.0, _task(obj), 0)
    assert phase == "approach" and tgt[2] == pytest.approx(0.035 + cfg.approach_height)
    # aligned above: descend to the grasp point
    tgt, g, phase, _ = plan_step(cfg, [0.32, 0.08, 0.12], 0.0, _task(obj), 0)
    assert phase == "descend" and np.allclose(tgt, [0.32, 0.08, 0.035])
    # at the grasp point: close, then keep closing while counting
    _, g, phase, n = plan_step(cfg, [0.32, 0.08, 0.036], 0.0, _task(obj), 0)
    assert (phase, g, n) == ("close", CLOSE, 1)
    _, g, phase, n = plan_step(cfg, [0.32, 0.08, 0.036], 0.05, _task(obj), n)
    assert (phase, g, n) == ("close", CLOSE, 2)
    # closed on it: lift first, then carry towards the shelf
    tgt, g, phase, _ = plan_step(cfg, [0.32, 0.08, 0.036], 0.4, _task(obj), 0)
    assert phase == "lift" and g == CLOSE and tgt[2] > 0.15
    carried = [0.32, 0.08, 0.19 + 0.015]
    tgt, _, phase, _ = plan_step(cfg, [0.32, 0.08, 0.19], 0.4, _task(carried), 0)
    assert phase == "carry" and tgt[1] == pytest.approx(0.24)
    # over the shelf: lower, then open
    above = [0.32, 0.24, 0.20]
    _, _, phase, _ = plan_step(cfg, [0.32, 0.24, 0.185], 0.4, _task(above), 0)
    assert phase == "lower"
    resting = [0.32, 0.24, 0.13 + cfg.place_drop]
    _, g, phase, _ = plan_step(cfg, [0.32, 0.24, 0.12], 0.4, _task(resting), 0)
    assert (phase, g) == ("open", OPEN)
    _, g, phase, _ = plan_step(cfg, [0.32, 0.24, 0.12], 0.0, _task(resting, on=True), 0)
    assert (phase, g) == ("release", OPEN)


def test_planner_retries_a_missed_close():
    cfg = ScriptedPickPlaceTeleopConfig(grasp_dz=-0.015, close_wait_steps=5)
    _, g, phase, n = plan_step(cfg, [0.32, 0.08, 0.036], 0.85, _task([0.40, 0.08, 0.05]), 5)
    assert (phase, g, n) == ("retry", OPEN, 0)


def test_scripted_pick_place_noise_is_per_step_and_bounded():
    from lerobot_teleoperator_devices.scripted_ee import TOOL_POSITION_SOURCE
    from lerobot_teleoperator_devices.scripted_pick_place import PICK_PLACE_SOURCE, ScriptedPickPlaceTeleop

    task = {**_task([0.32, 0.08, 0.05]), "finger_center": np.array([0.32, 0.08, 0.20])}
    PICK_PLACE_SOURCE["task"], TOOL_POSITION_SOURCE["gripper"] = (lambda: task), (lambda: 0.0)
    try:
        t = ScriptedPickPlaceTeleop(ScriptedPickPlaceTeleopConfig(grasp_dz=-0.015, noise_std=0.3, seed=0))
        acts = np.array([[t.get_action()[k] for k in ("delta_x", "delta_y", "delta_z")] for _ in range(200)])
        quiet = ScriptedPickPlaceTeleop(ScriptedPickPlaceTeleopConfig(grasp_dz=-0.015, noise_std=0.0))
        clean = np.array([quiet.get_action()[k] for k in ("delta_x", "delta_y", "delta_z")])
    finally:
        PICK_PLACE_SOURCE["task"], TOOL_POSITION_SOURCE["gripper"] = None, None
    assert np.all(np.abs(acts) <= 1.0)
    assert clean[2] == pytest.approx(-1.0) and np.allclose(clean[:2], 0.0)  # straight down, at full speed
    assert acts[:, 0].std() > 0.2 and abs(acts[:, 0].mean()) < 0.1  # x noise, zero mean
