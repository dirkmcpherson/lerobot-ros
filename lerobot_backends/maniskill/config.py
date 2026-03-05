"""ManiSkill backend configuration."""

from dataclasses import dataclass, field

from ..config import BackendRobotConfig


@dataclass
class ManiSkillBackendConfig(BackendRobotConfig):
    """Config that produces a :class:`ManiSkillBackend` via the factory."""

    env_id: str = "PickCube-v1"
    obs_mode: str = "rgbd"               # "state", "rgbd", "pointcloud"
    control_mode: str = "pd_joint_pos"
    render_mode: str | None = "rgb_array"
    seed: int | None = None
    env_kwargs: dict = field(default_factory=dict)

    # Mapping: which joints in the env correspond to our standard names
    joint_names: list[str] = field(default_factory=lambda: [
        "joint_1", "joint_2", "joint_3", "joint_4",
        "joint_5", "joint_6", "joint_7",
    ])
    gripper_joint_name: str | None = "gripper"

    # Index into qpos where the gripper observation lives.
    # For Panda: qpos has [7 arm, 2 finger] — index 7 is the first finger.
    # Set to None to skip gripper observation extraction.
    gripper_qpos_index: int | None = 7

    # Extra state keys to extract from obs["extra"] as environment_state.
    # Values are flattened and concatenated in order.
    # E.g. ["tcp_pose", "obj_pose", "goal_pos"] for PickCube.
    env_state_keys: list[str] = field(default_factory=list)

    # Camera mapping: our_name -> (height, width)
    camera_specs: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "camera_rgb": (128, 128),
    })
    # our_name -> maniskill_camera_name
    camera_mapping: dict[str, str] = field(default_factory=lambda: {
        "camera_rgb": "base_camera",
    })


@dataclass
class ManiSkillPickCubeConfig(ManiSkillBackendConfig):
    """Ready-to-use config for ManiSkill PickCube-v1."""

    env_id: str = "PickCube-v1"
    control_mode: str = "pd_joint_pos"
    obs_mode: str = "rgbd"

    # Extract extra state from ManiSkill "extra" obs dict.
    # tcp_pose=7, goal_pos=3 → 10-dim environment state
    env_state_keys: list[str] = field(default_factory=lambda: [
        "tcp_pose", "goal_pos",
    ])
