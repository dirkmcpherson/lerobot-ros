"""The shipped HIL-SERL configs parse with lerobot's own config classes and agree with the robot."""

from pathlib import Path

import draccus
import pytest

CONFIG_DIR = Path(__file__).parent.parent / "config" / "hilserl"


@pytest.fixture(scope="module", autouse=True)
def _plugins():
    import lerobot_robot_ros  # noqa: F401
    import lerobot_teleoperator_devices  # noqa: F401
    from lerobot.cameras import opencv  # noqa: F401
    from lerobot.teleoperators import gamepad  # noqa: F401

    from hilserl_gen3_lite import compat

    compat.apply()


def _load(cls, name):
    with draccus.config_type("json"):
        return draccus.parse(cls, CONFIG_DIR / name, args=[])


@pytest.mark.parametrize("name", ["train_gen3_lite_reach.json", "dryrun_train.json", "gazebo_train.json"])
def test_train_config(name):
    from lerobot.rl.train_rl import TrainRLServerPipelineConfig

    cfg = _load(TrainRLServerPipelineConfig, name)
    cfg.validate()
    cfg.to_dict()  # the learner logs this on start-up; needs compat.apply()
    robot = cfg.env.robot
    assert robot.type in ("gen3_lite_hil", "gen3_lite_hil_fake")

    # state and action sizes in the policy match what the robot produces
    n_state = 6 + 1 + 3 + (6 if robot.observe_joint_velocities else 0)
    n_action = 3 + (1 if robot.use_gripper else 0)
    assert cfg.policy.input_features["observation.state"].shape == (n_state,)
    assert cfg.policy.output_features["action"].shape == (n_action,)
    stats = cfg.policy.dataset_stats
    assert len(stats["observation.state"]["min"]) == len(stats["observation.state"]["max"]) == n_state
    assert len(stats["action"]["min"]) == n_action
    for cam in robot.cameras:
        assert f"observation.images.{cam}" in cfg.policy.input_features
    assert robot.use_gripper == cfg.env.processor.gripper.use_gripper == cfg.env.teleop.use_gripper
    assert cfg.env.fps == round(1 / robot.control_period_s)

    # stock HIL-SERL: half of each batch from each buffer, demos loaded as the offline buffer
    assert cfg.online_ratio == 0.5 and cfg.mixer == "online_offline"
    assert cfg.dataset is not None and cfg.algorithm.type == "sac"

    # conservative limits (brief: at most 1 cm per step at 10 Hz, 0.1 m/s)
    assert robot.max_ee_step_m <= 0.01 and robot.max_ee_speed_m_s <= 0.1
    assert robot.reset_joint_positions is not None


@pytest.mark.parametrize("name", ["env_gen3_lite_reach.json", "dryrun_env.json", "gazebo_env.json"])
def test_record_config(name):
    from lerobot.rl.gym_manipulator import GymManipulatorConfig

    cfg = _load(GymManipulatorConfig, name)
    assert cfg.mode == "record" and cfg.dataset.num_episodes_to_record > 0


def test_reset_pose_is_inside_the_tool_box(kinematics):
    from lerobot.rl.train_rl import TrainRLServerPipelineConfig

    robot = _load(TrainRLServerPipelineConfig, "train_gen3_lite_reach.json").env.robot
    pos, _ = kinematics.fk(robot.reset_joint_positions)
    assert all(lo + 0.03 < p < hi - 0.03 for p, lo, hi in zip(pos, robot.ee_bounds_min, robot.ee_bounds_max, strict=True))
