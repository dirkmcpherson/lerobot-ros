"""Gym environment and processor pipelines for HIL-SERL on the Gen3 Lite.

Replaces two functions of ``lerobot.rl.gym_manipulator`` (0.6.1):

``make_robot_env``  upstream's ``RobotEnv`` reads ``robot.bus.motors`` and resets through
    ``robot.bus.sync_read/sync_write`` (a Feetech/Dynamixel motor bus). The Gen3 Lite has
    no such bus, so ``Gen3LiteHilEnv`` talks to the robot through the lerobot ``Robot``
    interface instead, and resets with the robot's guarded homing.

``make_processors`` same steps and same order as upstream's real-robot branch, minus
    the inverse-kinematics steps: ``Gen3LiteHil`` takes end-effector deltas directly and
    does its own IK and bounds, so limits are enforced in the adapter rather than in a
    config-built pipeline. The takeover steps (``AddTeleopActionAsComplimentaryDataStep``,
    ``AddTeleopEventsAsInfoStep``, ``InterventionActionProcessorStep``) are upstream's.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from lerobot.envs import HILSerlRobotEnvConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    ProcessorStep,
    RewardClassifierProcessorStep,
    TimeLimitProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    identity_transition,
)
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators.utils import TeleopEvents, make_teleoperator_from_config
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

GRIPPER_TOGGLED = "gripper_toggled"
DISCRETE_PENALTY = "discrete_penalty"


class GripperTogglePenaltyStep(ProcessorStep):
    """Adds ``discrete_penalty`` to the complementary data: ``penalty`` (a negative number)
    on a step whose gripper command changed what the gripper is asked to do, else 0.

    Stands in for lerobot's ``GripperPenaltyProcessorStep``, which reads the last action
    dimension as a gripper joint target (its SO-100 pipeline converts to joint targets
    first). Here that dimension is the discrete command itself (0 close, 1 stay, 2 open):
    fed to the upstream step it would read "stay" as a close command and penalise every
    step taken with the gripper open. The environment reports the toggle; a button held
    down for a second while the fingers travel is one toggle, not ten.
    """

    def __init__(self, penalty: float = -0.02):
        self.penalty = float(penalty)

    def __call__(self, transition):
        out = transition.copy()
        info = out.get(TransitionKey.INFO) or {}
        data = dict(out.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        data[DISCRETE_PENALTY] = self.penalty if info.get(GRIPPER_TOGGLED, False) else 0.0
        out[TransitionKey.COMPLEMENTARY_DATA] = data
        return out

    def reset(self) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {"penalty": self.penalty}

    def transform_features(self, features):
        return features


class Gen3LiteHilEnv(gym.Env):
    """One Gen3LiteHil robot as a gym environment with end-effector delta actions."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        reset_joint_positions: list[float] | None = None,
        reset_time_s: float = 5.0,
        display_cameras: bool = False,
        on_reset: Callable[[], None] | None = None,
    ):
        super().__init__()
        self.robot = robot
        self.use_gripper = use_gripper
        self.reset_joint_positions = reset_joint_positions
        self.reset_time_s = reset_time_s
        self.display_cameras = display_cameras
        self.on_reset = on_reset
        self.current_step = 0
        self._raw_joint_positions: dict[str, float] | None = None
        self._gripper_asked_closed = False

        if not self.robot.is_connected:
            self.robot.connect()
        if reset_joint_positions is not None:  # roll, pitch, yaw frozen at the start pose's
            self.robot.freeze_orientation_at(list(reset_joint_positions))
        self._state_keys = list(self.robot.state_keys)
        self._image_keys = list(self.robot.config.cameras.keys())
        self._setup_spaces()

    def _get_observation(self) -> dict[str, Any]:
        obs = self.robot.get_observation()
        scalars = {k: float(obs[k]) for k in self._state_keys}
        self._raw_joint_positions = scalars
        return {
            "agent_pos": np.array([scalars[k] for k in self._state_keys], dtype=np.float32),
            "pixels": {k: obs[k] for k in self._image_keys},
            **scalars,
        }

    def _setup_spaces(self) -> None:
        obs = self._get_observation()
        spaces: dict[str, gym.Space] = {
            f"{OBS_IMAGES}.{k}": gym.spaces.Box(low=0, high=255, shape=v.shape, dtype=np.uint8)
            for k, v in obs["pixels"].items()
        }
        spaces[OBS_STATE] = gym.spaces.Box(low=-10, high=10, shape=obs["agent_pos"].shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        low, high = -np.ones(3), np.ones(3)
        if self.use_gripper:
            low, high = np.concatenate([low, [0]]), np.concatenate([high, [2]])
        self.action_space = gym.spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        start = time.perf_counter()
        if self.on_reset is not None:
            self.on_reset()
        if self.reset_joint_positions is not None:
            # Guarded, slow joint-space move; raises if the workspace guard blocks the path.
            self.robot.go_home(list(self.reset_joint_positions))
        precise_sleep(max(self.reset_time_s - (time.perf_counter() - start), 0.0))
        self.current_step = 0
        obs = self._get_observation()
        if self.use_gripper:  # homing opens the gripper; without homing, go by what is measured
            iface = self.robot.config.ros2_interface
            half = 0.5 * (iface.gripper_open_position + iface.gripper_close_position)
            self._gripper_asked_closed = obs.get("gripper.pos", iface.gripper_open_position) > half
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        a = np.asarray(action, dtype=float).reshape(-1)
        command = {"delta_x": a[0], "delta_y": a[1], "delta_z": a[2]}
        toggled = False
        if self.use_gripper and a.size > 3:
            command["gripper"] = a[3]
            g = int(round(float(a[3]))) if np.isfinite(a[3]) else 1
            if (g <= 0 and not self._gripper_asked_closed) or (g >= 2 and self._gripper_asked_closed):
                toggled = True
                self._gripper_asked_closed = g <= 0
        try:
            self.robot.send_action(command)
            obs = self._get_observation()
        except Exception:
            # Stale data, a dead driver, a bad action: stop the arm before the error travels up.
            try:
                self.robot.hold("error in step")
            except Exception as hold_error:
                logger.error(f"Hold after a failed step also failed: {hold_error}")
            raise
        if self.display_cameras:
            self.render()
        self.current_step += 1
        return obs, 0.0, False, False, {TeleopEvents.IS_INTERVENTION: False, GRIPPER_TOGGLED: toggled}

    def render(self) -> None:
        import cv2

        if self._image_keys:
            obs = self.robot.get_observation()
            for key in self._image_keys:
                cv2.imshow(key, cv2.cvtColor(obs[key], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self) -> None:
        self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float] | None:
        """Latest scalar observations (the name is upstream's; steps read ``gripper.pos`` from it)."""
        return self._raw_joint_positions


def make_robot_env(cfg: HILSerlRobotEnvConfig, on_reset: Callable[[], None] | None = None):
    """Drop-in for ``lerobot.rl.gym_manipulator.make_robot_env``: (env, teleop device)."""
    if cfg.robot is None or cfg.teleop is None:
        raise ValueError("env.robot and env.teleop must both be set")
    robot = make_robot_from_config(cfg.robot)
    if not hasattr(robot, "state_keys") or not hasattr(robot, "go_home"):
        raise TypeError(
            f"robot type '{cfg.robot.type}' is not an end-effector delta robot; use gen3_lite_hil "
            "(or gen3_lite_hil_fake for a dry run)"
        )
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    teleop_device.connect()

    proc = cfg.processor
    use_gripper = proc.gripper.use_gripper if proc.gripper is not None else False
    if use_gripper != bool(getattr(cfg.robot, "use_gripper", False)):
        raise ValueError("env.processor.gripper.use_gripper and env.robot.use_gripper must match")
    # The reset pose can be given on the robot (env.robot.reset_joint_positions) or, as
    # upstream does, in env.processor.reset.fixed_reset_joint_positions.
    reset_pose = getattr(cfg.robot, "reset_joint_positions", None)
    if reset_pose is None and proc.reset is not None:
        reset_pose = proc.reset.fixed_reset_joint_positions
    env = Gen3LiteHilEnv(
        robot=robot,
        use_gripper=use_gripper,
        reset_joint_positions=reset_pose,
        reset_time_s=proc.reset.reset_time_s if proc.reset is not None else 5.0,
        display_cameras=proc.observation.display_cameras if proc.observation is not None else False,
        on_reset=on_reset,
    )
    return env, teleop_device


def make_processors(env: gym.Env, teleop_device, cfg: HILSerlRobotEnvConfig, device: str = "cpu"):
    """Drop-in for ``lerobot.rl.gym_manipulator.make_processors``: (env pipeline, action pipeline)."""
    proc = cfg.processor
    terminate_on_success = proc.reset.terminate_on_success if proc.reset is not None else True
    use_gripper = proc.gripper.use_gripper if proc.gripper is not None else False
    if proc.inverse_kinematics is not None:
        logger.warning("env.processor.inverse_kinematics is ignored: Gen3LiteHil does its own IK and bounds.")

    env_steps: list = [VanillaObservationProcessorStep()]
    if proc.image_preprocessing is not None:
        env_steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=proc.image_preprocessing.crop_params_dict,
                resize_size=proc.image_preprocessing.resize_size,
            )
        )
    if proc.reset is not None:
        env_steps.append(TimeLimitProcessorStep(max_episode_steps=int(proc.reset.control_time_s * cfg.fps)))
    if use_gripper:
        penalty = proc.gripper.gripper_penalty
        if penalty > 0:
            raise ValueError(f"env.processor.gripper.gripper_penalty must be zero or negative, got {penalty}")
        env_steps.append(GripperTogglePenaltyStep(penalty=penalty))
    if proc.reward_classifier is not None and proc.reward_classifier.pretrained_path is not None:
        env_steps.append(
            RewardClassifierProcessorStep(
                pretrained_path=proc.reward_classifier.pretrained_path,
                device=device,
                success_threshold=proc.reward_classifier.success_threshold,
                success_reward=proc.reward_classifier.success_reward,
                terminate_on_success=terminate_on_success,
            )
        )
    env_steps += [AddBatchDimensionProcessorStep(), DeviceProcessorStep(device=device)]

    action_steps = [
        AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
        AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
        InterventionActionProcessorStep(use_gripper=use_gripper, terminate_on_success=terminate_on_success),
    ]
    return (
        DataProcessorPipeline(steps=env_steps, to_transition=identity_transition, to_output=identity_transition),
        DataProcessorPipeline(steps=action_steps, to_transition=identity_transition, to_output=identity_transition),
    )
