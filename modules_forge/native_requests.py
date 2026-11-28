"""Small dependency-free adapters for native generation UI contracts."""
import math
import threading
from contextlib import contextmanager


def script_control(runner, args, script_class, control_name, default=None):
    """Resolve a built-in control through the runner's actual input layout."""
    for script in runner.alwayson_scripts:
        if type(script).__name__ != script_class:
            continue
        control = getattr(script, control_name, None)
        if control is None:
            continue
        for index, component in enumerate(runner.inputs):
            if component is control and index < len(args):
                return args[index]
    return default


def positive_integer(value, label):
    try:
        number = int(value)
        if number < 1 or float(value) != number:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{label} must be a positive integer.") from None
    return number


def scaled_video_dimensions(width, height, scale):
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Resize scale must be positive and finite.")
    return tuple(max(64, math.floor(size * scale / 32 + 0.5) * 32) for size in (width, height))


class PageJobOwner:
    """Scope a dedicated page interrupt to its currently active session."""
    def __init__(self):
        self._lock = threading.Lock()
        self._session = None

    @contextmanager
    def active(self, session):
        with self._lock:
            self._session = session
        try:
            yield
        finally:
            with self._lock:
                self._session = None

    def interrupt(self, session, callback):
        with self._lock:
            if self._session is not None and self._session == session:
                callback()
                return True
            return False
