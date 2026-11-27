"""Low-overhead profiler annotations used by Forge hot paths."""

from contextlib import nullcontext
from threading import local

import torch


_state = local()


def profiler_enabled() -> bool:
    return bool(getattr(_state, "enabled", False))


def set_profiler_enabled(enabled: bool) -> None:
    _state.enabled = bool(enabled)


def record_region(name: str):
    """Create a profiler range only while Forge's profiler is collecting."""
    if not profiler_enabled():
        return nullcontext()
    return torch.profiler.record_function(name)
