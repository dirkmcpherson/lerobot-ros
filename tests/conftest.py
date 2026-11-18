import os
from pathlib import Path

import pytest

# lerobot's ReplayBuffer wraps its image augmentation in torch.compile, which needs a C++
# compiler at first use. Tests must run on machines without one.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

FIXTURES = Path(__file__).parent / "fixtures"
URDF = str(FIXTURES / "gen3_lite.urdf")

# A pose inside the default tool box: tool at (0.30, 0.16, 0.22), same orientation as Kinova home.
START_JOINTS = [-0.255, -0.607, 2.202, 0.249, -1.469, -1.227]


@pytest.fixture(scope="session")
def urdf_path() -> str:
    return URDF


@pytest.fixture(scope="session")
def kinematics(urdf_path):
    from lerobot_robot_ros.ee_delta import ArmKinematics

    return ArmKinematics(urdf_path, [f"joint_{i}" for i in range(1, 7)], "tool_frame")


@pytest.fixture
def make_robot(urdf_path, kinematics, tmp_path):
    """Build a connected Gen3LiteHil on a FakeArmBackend; disconnects at teardown."""
    from lerobot_robot_ros import FakeArmBackend, Gen3LiteHil, Gen3LiteHilFakeConfig

    robots = []

    def _make(start_joints=None, connect=True, **overrides):
        cfg = Gen3LiteHilFakeConfig(urdf_path=urdf_path, calibration_dir=tmp_path, **overrides)
        backend = FakeArmBackend(cfg, start_joints=START_JOINTS if start_joints is None else start_joints)
        robot = Gen3LiteHil(cfg, backend=backend, kinematics=kinematics)
        if connect:
            robot.connect()
        robots.append(robot)
        return robot

    yield _make
    for r in robots:
        r.disconnect()


KORTEX_SRC = Path(os.environ.get("KORTEX_DESCRIPTION_PARENT", Path.home() / "gen3_docker_ws/src/ros2_kortex"))
