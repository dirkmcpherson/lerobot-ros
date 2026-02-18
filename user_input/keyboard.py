import logging
import threading

from .base import UserInput

logger = logging.getLogger(__name__)


class KeyboardUserInput(UserInput):
    """Joint-space keyboard teleoperation using pynput (non-blocking).

    Controls
    --------
    1 – 7        Select joint to control
    ] / [        Increment / decrement selected joint by step_size radians
    Enter        Save episode and advance to next
    D            Discard episode (robot returns to home, frame data thrown away)
    Q / Escape   Quit recording session
    """

    def __init__(
        self,
        joint_names: list[str],
        min_joint_positions: list[float],
        max_joint_positions: list[float],
        step_size: float = 0.05,
    ):
        self._names = joint_names
        self._min = min_joint_positions
        self._max = max_joint_positions
        self._step = step_size
        self._n = len(joint_names)

        self._targets: list[float] = [0.0] * self._n
        self._selected: int = 0
        self._end: bool = False
        self._discard: bool = False
        self._quit: bool = False
        self._listener = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        from pynput import keyboard as kb

        self._listener = kb.Listener(on_press=self._on_press, suppress=False)
        self._listener.start()
        logger.info(
            "Keyboard input ready. "
            f"[1-{self._n}] select joint | ]/[ move | Enter=save | D=discard | Q=quit"
        )

    def disconnect(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def reset(self, current_positions: list[float]) -> None:
        with self._lock:
            self._targets = list(current_positions)
            self._end = False
            self._discard = False

    def get_action(self, current_positions: list[float]) -> list[float]:
        with self._lock:
            return list(self._targets)

    @property
    def episode_end_requested(self) -> bool:
        with self._lock:
            return self._end

    @property
    def discard_requested(self) -> bool:
        with self._lock:
            return self._discard

    @property
    def quit_requested(self) -> bool:
        with self._lock:
            return self._quit

    def _on_press(self, key) -> None:
        from pynput.keyboard import Key

        ch = getattr(key, "char", None)

        with self._lock:
            if ch and ch.isdigit():
                idx = int(ch) - 1
                if 0 <= idx < self._n:
                    self._selected = idx
                    logger.info(
                        f"Joint {idx+1} ({self._names[idx]}) selected — "
                        f"current target: {self._targets[idx]:.3f} rad"
                    )
            elif ch == "]":
                j = self._selected
                self._targets[j] = min(self._targets[j] + self._step, self._max[j])
                logger.info(f"Joint {j+1} → {self._targets[j]:.3f} rad")
            elif ch == "[":
                j = self._selected
                self._targets[j] = max(self._targets[j] - self._step, self._min[j])
                logger.info(f"Joint {j+1} → {self._targets[j]:.3f} rad")
            elif key == Key.enter:
                logger.info("Episode end requested (save).")
                self._end = True
            elif ch and ch.lower() == "d":
                logger.info("Episode end requested (discard).")
                self._discard = True
                self._end = True
            elif (ch and ch.lower() == "q") or key == Key.esc:
                logger.info("Quit requested.")
                self._quit = True
                self._end = True
