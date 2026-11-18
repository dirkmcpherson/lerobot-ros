"""limit_step and ArmKinematics: no robot, no ROS."""

import numpy as np
import pytest

from lerobot_robot_ros.ee_delta import EeLimits, limit_step

from .conftest import START_JOINTS

LIM = EeLimits()  # 1 cm per unit, 1 cm max step, 0.1 m/s at 10 Hz
INSIDE = np.array([0.30, 0.16, 0.22])


def test_unit_action_moves_one_step():
    target, rep = limit_step([1, 0, 0], INSIDE, INSIDE, LIM)
    np.testing.assert_allclose(target - INSIDE, [0.01, 0, 0], atol=1e-12)
    assert not rep.any_clipped


def test_action_is_clipped_to_unit_range():
    target, rep = limit_step([5, 0, 0], INSIDE, INSIDE, LIM)
    np.testing.assert_allclose(target - INSIDE, [0.01, 0, 0], atol=1e-12)
    assert rep.action_clipped


def test_diagonal_step_is_capped_to_max_length():
    target, rep = limit_step([1, 1, 1], INSIDE, INSIDE, LIM)
    assert np.linalg.norm(target - INSIDE) == pytest.approx(0.01)
    assert rep.step_clipped


def test_speed_limit_shrinks_steps_sent_faster_than_the_control_rate():
    # 20 ms after the previous command, 0.1 m/s allows 2 mm.
    target, rep = limit_step([1, 0, 0], INSIDE, INSIDE, LIM, dt_since_last_command=0.02)
    assert np.linalg.norm(target - INSIDE) == pytest.approx(0.002)
    assert rep.step_clipped
    # A long pause does not allow a longer step than one control period's worth.
    target, _ = limit_step([1, 0, 0], INSIDE, INSIDE, LIM, dt_since_last_command=5.0)
    assert np.linalg.norm(target - INSIDE) == pytest.approx(0.01)


def test_target_cannot_lead_the_measured_position():
    reference = INSIDE + [0.05, 0, 0]  # target ran 5 cm ahead of the arm
    target, rep = limit_step([1, 0, 0], reference, INSIDE, LIM)
    assert np.linalg.norm(target - INSIDE) == pytest.approx(LIM.max_lead_m)
    assert rep.lead_clipped


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_box_stops_motion_at_each_face(axis, sign):
    face = (LIM.box_max if sign > 0 else LIM.box_min)[axis]
    start = INSIDE.copy()
    start[axis] = face - sign * 0.003  # 3 mm inside the face
    action = np.zeros(3)
    action[axis] = sign
    target, rep = limit_step(action, start, start, LIM)
    assert target[axis] == pytest.approx(face)
    assert rep.box_clipped


def test_outside_the_box_zero_action_does_not_move_and_only_inward_motion_is_allowed():
    outside = np.array([LIM.box_max[0] + 0.02, 0.16, 0.22])
    target, _ = limit_step([0, 0, 0], outside, outside, LIM)
    np.testing.assert_allclose(target, outside)
    target, rep = limit_step([1, 0, 0], outside, outside, LIM)  # further out
    np.testing.assert_allclose(target, outside)
    assert rep.box_clipped
    target, _ = limit_step([-1, 0, 0], outside, outside, LIM)  # back in
    assert target[0] == pytest.approx(outside[0] - 0.01)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_action_is_refused(bad):
    with pytest.raises(ValueError):
        limit_step([bad, 0, 0], INSIDE, INSIDE, LIM)


def test_random_walk_never_leaves_the_box_or_exceeds_the_step():
    rng = np.random.default_rng(0)
    pos = INSIDE.copy()
    for _ in range(5000):
        new, _ = limit_step(rng.uniform(-3, 3, 3), pos, pos, LIM)
        assert np.linalg.norm(new - pos) <= LIM.max_step_m + 1e-12
        assert np.all(new >= np.array(LIM.box_min) - 1e-12) and np.all(new <= np.array(LIM.box_max) + 1e-12)
        pos = new


def test_limits_validation():
    with pytest.raises(ValueError):
        EeLimits(box_min=[0.3, 0, 0], box_max=[0.2, 1, 1]).validate()
    with pytest.raises(ValueError):
        EeLimits(max_step_m=0).validate()


def test_fk_matches_the_home_pose_measured_on_the_real_arm(kinematics):
    # docs/GEN3_LITE_DAGGER.md: Kinova home has the tool at (0.439, 0.193, 0.443) on the real arm.
    pos, _ = kinematics.fk([0.0, -0.282, 1.315, -0.003, -1.047, 0.0])
    np.testing.assert_allclose(pos, [0.439, 0.193, 0.443], atol=0.003)


def test_ik_reaches_a_one_centimetre_step_and_holds_orientation(kinematics):
    pos, rot = kinematics.fk(START_JOINTS)
    rng = np.random.default_rng(1)
    for _ in range(50):
        step = rng.normal(size=3)
        step *= 0.01 / np.linalg.norm(step)
        q, pos_err, rot_err = kinematics.ik(START_JOINTS, pos + step, rot)
        assert pos_err < 1e-4 and rot_err < 1e-3
        assert np.abs(q - START_JOINTS).max() < 0.1  # a small Cartesian step is a small joint step
        new_pos, new_rot = kinematics.fk(q)
        np.testing.assert_allclose(new_pos, pos + step, atol=1e-4)
        np.testing.assert_allclose(new_rot, rot, atol=2e-3)


def test_ik_reports_an_unreachable_target(kinematics):
    pos, rot = kinematics.fk(START_JOINTS)
    _, pos_err, _ = kinematics.ik(START_JOINTS, pos + [2.0, 0, 0], rot)
    assert pos_err > 0.5
