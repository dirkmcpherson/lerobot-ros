from abc import ABC, abstractmethod


class UserInput(ABC):
    """Abstract teleoperation interface for data recording.

    Each implementation converts raw device input into absolute joint target
    positions at each control step. The recording loop calls get_action() every
    tick and polls the flags to manage episode flow.
    """

    @abstractmethod
    def connect(self) -> None:
        """Initialize the device."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the device."""

    @abstractmethod
    def get_action(self, current_positions: list[float]) -> list[float]:
        """Return next absolute target joint positions (radians).

        Called at each control step. Must be non-blocking and always return a
        complete list the same length as current_positions.
        """

    @property
    @abstractmethod
    def episode_end_requested(self) -> bool:
        """True when user signals to save and end the current episode."""

    @property
    @abstractmethod
    def discard_requested(self) -> bool:
        """True when user signals to discard (not save) the current episode."""

    @property
    @abstractmethod
    def quit_requested(self) -> bool:
        """True when user signals to stop all recording."""

    @property
    def time_from_start_sec(self) -> float:
        """Trajectory controller horizon for each command.

        Velocity-integrated devices (spacemouse) should use a short horizon
        (~1/FPS) so the controller executes each incremental step immediately.
        Discrete devices (keyboard) can use a longer horizon (1.0 s).
        """
        return 1.0

    def reset(self, current_positions: list[float]) -> None:
        """Called at the start of each episode with the robot's actual position.

        Override to sync internal state so motion starts smoothly from wherever
        the robot actually is after the home reset.
        """
