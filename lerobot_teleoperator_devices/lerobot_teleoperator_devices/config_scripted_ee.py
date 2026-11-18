from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("scripted_ee")
@dataclass
class ScriptedEeTeleopConfig(TeleoperatorConfig):
    """A stand-in for the operator and the gamepad, for tests and dry runs with
    ``gen3_lite_hil_fake`` and for unattended runs of ``gen3_lite_hil`` in Gazebo. Same
    actions and events as lerobot's ``gamepad`` teleop.

    Without a gripper it drives the tool towards ``target`` and reports success (the
    operator's success key) once the tool is within ``success_radius_m`` of it. With
    ``use_gripper`` it does a pick and place between two points (see ``place_target``).
    """

    use_gripper: bool = False
    # Step ranges [start, end) within each episode during which it takes over.
    takeovers: list[list[int]] = field(default_factory=lambda: [[10, 25]])
    # Stop taking over after this many episodes ("intervene often early, then less"), which
    # also leaves later evaluation blocks hands-off. None: take over in every episode.
    max_takeover_episodes: int | None = None
    # True: hold the takeover trigger for whole episodes (recording demos).
    always_intervene: bool = False
    target: list[float] = field(default_factory=lambda: [0.33, 0.12, 0.25])
    # With use_gripper, a pick and place without an object: go to ``target``, close the
    # gripper, carry to ``place_target``, open. Success needs the whole sequence: the
    # gripper closed while at the pick point, later open at the place point.
    place_target: list[float] = field(default_factory=lambda: [0.27, 0.20, 0.25])
    # Gripper joint position above which it counts as closed, below ``open`` as open.
    gripper_closed_above: float = 0.4
    gripper_open_below: float = 0.08
    success_radius_m: float = 0.015
    # Steps after which the step counter restarts (the env's time limit, in steps).
    episode_steps: int = 100
    # Metres per unit of action, to scale the proportional move (match robot.ee_step_size_m).
    ee_step_size_m: float = 0.01
