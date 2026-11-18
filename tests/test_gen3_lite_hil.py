"""Gen3LiteHil on a fake backend: limits, staleness, watchdog, lifecycle. No hardware."""

import time

import numpy as np
import pytest

from lerobot_robot_ros import Gen3LiteHilFakeConfig, StaleObservationError
from lerobot_robot_ros.gen3_lite_hil import DELTA_KEYS

from .conftest import KORTEX_SRC, START_JOINTS

PERIOD = 0.02  # fast control period so the tests run quickly


def act(robot, x=0.0, y=0.0, z=0.0, **extra):
    return robot.send_action({"delta_x": x, "delta_y": y, "delta_z": z, **extra})


def tool(robot):
    obs = robot.get_observation()
    return np.array([obs["ee.x"], obs["ee.y"], obs["ee.z"]])


# ----------------------------------------------------------------- features / observation


def test_features(make_robot):
    robot = make_robot()
    assert list(robot.action_features) == list(DELTA_KEYS)
    obs = robot.get_observation()
    assert set(obs) == set(robot.observation_features)
    assert robot.state_keys == [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos", "ee.x", "ee.y", "ee.z"]
    np.testing.assert_allclose(tool(robot), [0.30, 0.16, 0.22], atol=1e-3)


def test_gripper_and_velocity_features_are_optional(make_robot):
    robot = make_robot(use_gripper=True, observe_joint_velocities=True)
    assert list(robot.action_features) == [*DELTA_KEYS, "gripper"]
    assert "joint_3.vel" in robot.get_observation()


def test_observation_ages_are_reported(make_robot):
    robot = make_robot()
    robot.get_observation()
    assert robot.observation_ages()["joint_state"] < 0.05


# ------------------------------------------------------------------------------- limits


def test_unit_action_moves_the_tool_one_centimetre_and_holds_orientation(make_robot, kinematics):
    robot = make_robot()
    _, rot0 = kinematics.fk(robot.backend.joints)
    start = tool(robot)
    applied = act(robot, x=1.0)
    np.testing.assert_allclose(tool(robot) - start, [0.01, 0, 0], atol=2e-4)
    np.testing.assert_allclose([applied[k] for k in DELTA_KEYS], [1, 0, 0], atol=0.02)
    _, rot1 = kinematics.fk(robot.backend.joints)
    np.testing.assert_allclose(rot1, rot0, atol=2e-3)


def test_zero_action_does_not_move(make_robot):
    robot = make_robot()
    joints = robot.backend.joints.copy()
    act(robot)
    np.testing.assert_allclose(robot.backend.joints, joints, atol=1e-6)


def test_oversized_action_is_clipped(make_robot):
    robot = make_robot()
    start = tool(robot)
    act(robot, x=50.0, y=50.0, z=50.0)
    assert np.linalg.norm(tool(robot) - start) <= 0.01 + 2e-4
    assert robot.clipped_commands == 1
    assert robot.last_report.action_clipped and robot.last_report.step_clipped


def test_commands_sent_faster_than_the_control_rate_are_slowed(make_robot):
    robot = make_robot()  # 0.1 s control period, 0.1 m/s
    start = tool(robot)
    t0 = time.monotonic()
    for _ in range(20):
        act(robot, x=1.0)
    elapsed = time.monotonic() - t0
    travelled = np.linalg.norm(tool(robot) - start)
    # first step may be a full 1 cm (no previous command), the rest obey 0.1 m/s
    assert travelled <= 0.01 + 0.1 * elapsed + 1e-3


def test_random_policy_stays_inside_the_tool_box(make_robot):
    robot = make_robot(control_period_s=PERIOD, max_ee_speed_m_s=0.01 / PERIOD)
    cfg = robot.config
    rng = np.random.default_rng(0)
    bias = np.array([1.0, 1.0, -1.0])  # push towards a corner of the box
    prev = tool(robot)
    for i in range(400):
        if i == 200:
            bias = -bias
        act(robot, *(rng.uniform(-1, 1, 3) + bias))
        time.sleep(PERIOD)
        now = np.array(robot._kin.fk(robot.backend.joints)[0])
        assert np.linalg.norm(now - prev) <= cfg.max_ee_step_m + 5e-4
        assert np.all(now >= np.array(cfg.ee_bounds_min) - 5e-4), now
        assert np.all(now <= np.array(cfg.ee_bounds_max) + 5e-4), now
        prev = now
    assert robot.clipped_commands > 0


def test_ee_bounds_must_lie_inside_the_guard_box(urdf_path):
    with pytest.raises(ValueError, match="inside the workspace guard box"):
        Gen3LiteHilFakeConfig(urdf_path=urdf_path, ee_bounds_max=[0.60, 0.29, 0.40])


def test_unreachable_step_is_rejected_and_the_arm_holds(make_robot):
    robot = make_robot(ik_position_tolerance_m=1e-9)  # nothing passes
    joints = robot.backend.joints.copy()
    act(robot, x=1.0)
    assert robot.ik_rejections == 1
    np.testing.assert_allclose(robot.backend.joints, joints, atol=1e-9)


def test_non_finite_action_raises_and_sends_nothing(make_robot):
    robot = make_robot()
    sent = len(robot.backend.sent)
    with pytest.raises(ValueError):
        act(robot, x=float("nan"))
    assert len(robot.backend.sent) == sent


# ------------------------------------------------------------------------------ gripper


def test_gripper_discrete_commands(make_robot):
    robot = make_robot(use_gripper=True)
    iface = robot.config.ros2_interface
    act(robot, gripper=1)  # stay
    assert robot.backend.gripper == pytest.approx(iface.gripper_open_position)
    act(robot, gripper=0)  # close
    assert robot.backend.gripper == pytest.approx(iface.gripper_close_position)
    act(robot, gripper=1)  # stay closed
    assert robot.backend.gripper == pytest.approx(iface.gripper_close_position)
    act(robot, gripper=2)  # open
    assert robot.backend.gripper == pytest.approx(iface.gripper_open_position)


def test_gripper_is_left_alone_when_unused(make_robot):
    robot = make_robot()
    act(robot, x=1.0, gripper=0)
    assert all("gripper.pos" not in s for s in robot.backend.sent)


# ---------------------------------------------------------------------------- staleness


def test_stale_joint_state_blocks_commands(make_robot):
    robot = make_robot(max_obs_age_s=0.05)
    act(robot, x=1.0)
    robot.backend.frozen = True
    time.sleep(0.08)
    sent = len(robot.backend.sent)
    with pytest.raises(StaleObservationError):
        act(robot, x=1.0)
    with pytest.raises(StaleObservationError):
        robot.get_observation()
    assert len(robot.backend.sent) == sent  # nothing was commanded on stale data
    assert robot.stale_stops >= 2
    robot.backend.frozen = False
    act(robot, x=1.0)  # recovers once state is fresh again


def test_stale_camera_frame_raises(make_robot):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cams = {"front": OpenCVCameraConfig(index_or_path=0, fps=10, width=32, height=24)}
    robot = make_robot(max_obs_age_s=0.05, cameras=cams)
    assert robot.get_observation()["front"].shape == (24, 32, 3)
    robot.backend.camera_frozen = True
    time.sleep(0.08)
    with pytest.raises(StaleObservationError, match="front"):
        robot.get_observation()


# ----------------------------------------------------------------------------- watchdog


def test_watchdog_holds_when_actions_stop(make_robot):
    robot = make_robot(control_period_s=PERIOD, max_ee_speed_m_s=0.01 / PERIOD, watchdog_periods=3)
    act(robot, x=1.0)
    assert robot.watchdog_trips == 0
    time.sleep(PERIOD * 8)
    assert robot.watchdog_trips == 1
    hold = robot.backend.sent[-1]
    np.testing.assert_allclose([hold[f"joint_{i}.pos"] for i in range(1, 7)], robot.backend.joints, atol=1e-9)
    time.sleep(PERIOD * 8)
    assert robot.watchdog_trips == 1  # one hold per silence, not a stream of them


def test_watchdog_stays_quiet_while_actions_arrive_and_rearms(make_robot):
    robot = make_robot(control_period_s=PERIOD, max_ee_speed_m_s=0.01 / PERIOD, watchdog_periods=3)
    for _ in range(15):
        act(robot, x=0.2)
        time.sleep(PERIOD)
    assert robot.watchdog_trips == 0
    time.sleep(PERIOD * 8)
    assert robot.watchdog_trips == 1
    act(robot, x=0.2)
    time.sleep(PERIOD * 8)
    assert robot.watchdog_trips == 2


def test_watchdog_does_not_fire_before_the_first_action(make_robot):
    robot = make_robot(control_period_s=PERIOD, max_ee_speed_m_s=0.01 / PERIOD)
    time.sleep(PERIOD * 8)
    assert robot.watchdog_trips == 0 and robot.backend.sent == []


def test_first_action_after_a_pause_restarts_from_the_measured_pose(make_robot):
    robot = make_robot(control_period_s=PERIOD, max_ee_speed_m_s=0.01 / PERIOD, resync_after_s=0.1)
    act(robot, x=1.0)
    time.sleep(0.2)
    robot.backend.joints = np.array(START_JOINTS)  # the arm is somewhere else now (e.g. homed)
    start = tool(robot)
    act(robot, y=1.0)
    np.testing.assert_allclose(tool(robot) - start, [0, 0.01, 0], atol=2e-4)


# ---------------------------------------------------------------------------- lifecycle


def test_connect_and_disconnect_are_safe_to_call_twice(make_robot):
    robot = make_robot(connect=False)
    assert not robot.is_connected
    robot.connect()
    robot.connect()
    assert robot.is_connected
    thread = robot._watchdog_thread
    robot.disconnect()
    assert not robot.is_connected and not thread.is_alive()
    robot.disconnect()
    robot.connect()  # and it can come back
    act(robot, x=1.0)


def test_disconnect_commands_a_hold(make_robot):
    robot = make_robot()
    act(robot, x=1.0)
    robot.backend.joints = robot.backend.joints + 0.001  # arm still settling
    robot.disconnect()
    hold = robot.backend.sent[-1]
    np.testing.assert_allclose([hold[f"joint_{i}.pos"] for i in range(1, 7)], robot.backend.joints, atol=1e-9)


def test_not_connected_raises(make_robot):
    from lerobot.utils.errors import DeviceNotConnectedError

    robot = make_robot(connect=False)
    with pytest.raises(DeviceNotConnectedError):
        act(robot, x=1.0)
    with pytest.raises(DeviceNotConnectedError):
        robot.get_observation()


def test_homing_goes_through_the_joint_path_and_restarts_the_reference(make_robot):
    robot = make_robot(home_speed=5.0, home_tolerance=0.02)
    act(robot, x=1.0)
    robot.go_home()
    np.testing.assert_allclose(robot.backend.joints, robot.config.home_position, atol=0.02)
    start = tool(robot)  # Kinova home is outside the default tool box
    act(robot)  # zero action: must not move
    np.testing.assert_allclose(tool(robot), start, atol=1e-6)
    act(robot, x=1.0)  # further out in x: blocked
    np.testing.assert_allclose(tool(robot), start, atol=1e-6)
    time.sleep(0.1)  # one control period, so the speed limit allows a full step
    act(robot, x=-1.0, z=-1.0)  # back towards the box: allowed
    assert np.all(tool(robot)[[0, 2]] < start[[0, 2]] - 0.005)


# -------------------------------------------------------------------- workspace guard


class _BlockEverythingBelow:
    """Guard stand-in: blocks any arm target whose tool z is under a limit."""

    def __init__(self, kinematics, z_min):
        self.kin, self.z_min = kinematics, z_min

    def assess(self, arm, gripper=0.0):
        z = self.kin.fk(arm)[0][2]
        return (self.z_min - z, "below test plane") if z < self.z_min else (0.0, "")


def test_guard_block_holds_the_arm_and_the_reference_follows(make_robot, kinematics):
    robot = make_robot()
    z0 = tool(robot)[2]
    robot._guard = _BlockEverythingBelow(kinematics, z0 - 0.015)
    for _ in range(5):
        act(robot, z=-1.0)
        time.sleep(0.1)
    assert robot.blocked_commands >= 3
    assert tool(robot)[2] >= z0 - 0.015 - 1e-6
    before = tool(robot)
    act(robot, z=1.0)  # moving away from the plane works straight away
    assert tool(robot)[2] == pytest.approx(before[2] + 0.01, abs=5e-4)


@pytest.mark.skipif(not (KORTEX_SRC / "kortex_description").is_dir(), reason="kortex_description meshes not found")
def test_real_guard_stops_the_tool_above_the_table(make_robot, urdf_path):
    """The mesh-based guard (table plane, self-collision) under the EE-delta path."""
    srdf = KORTEX_SRC / "kortex_moveit_config/kinova_gen3_lite_moveit_config/config/gen3_lite.srdf"
    from lerobot_backends.ros2.workspace_guard import WorkspaceGuard

    robot = make_robot(ee_bounds_min=[0.15, 0.03, 0.021], max_obs_age_s=1.0)
    cfg = robot.config
    robot._guard = WorkspaceGuard(
        urdf_path=urdf_path,
        arm_joint_names=cfg.ros2_interface.arm_joint_names,
        gripper_joint_name=cfg.ros2_interface.gripper_joint_name,
        plane_z=0.10,  # a raised "table" so the fingers reach it before the tool box floor
        plane_margin=cfg.plane_margin,
        box_min=cfg.workspace_min,
        box_max=cfg.workspace_max,
        srdf_path=str(srdf) if srdf.is_file() else None,
        package_dirs=[str(KORTEX_SRC)],
    )
    assert robot._guard.assess(list(robot.backend.joints), 0.0)[0] == 0.0
    for _ in range(25):
        act(robot, z=-1.0)
        time.sleep(0.1)
    assert robot.blocked_commands > 0
    violation, reason = robot._guard.assess(list(robot.backend.joints), 0.0)
    assert violation == 0.0, reason
    assert tool(robot)[2] > 0.10
