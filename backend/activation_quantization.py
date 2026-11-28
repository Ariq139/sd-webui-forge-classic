"""Short-lived, opt-in reuse of deterministic unscaled INT8 quantization."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from backend.args import args
from backend.performance import record_region

_ACTIVE = ContextVar("forge_shared_activation_quantization", default=None)


@contextmanager
def shared_activation_scope(source):
    enabled = (getattr(args, "shared_activation_quantization", False)
               and type(source) is torch.Tensor and source.is_cuda
               and not torch.is_grad_enabled()
               and not any(getattr(args, key, None) is not None
                           for key in ("multi_gpu_anima", "parallel_cfg", "parallel_batch")))
    # Only retain a single activation within an explicitly read-only group.
    token = _ACTIVE.set((source, {}) if enabled else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def quantize_shared_activation(source, reshaped, layout, scale, quantize):
    active = _ACTIVE.get()
    if (active is None or active[0] is not source
            or layout != "TensorWiseINT8Layout" or scale is not None):
        return quantize(reshaped, layout, scale=scale)
    key = (layout, source.dtype, source.device, tuple(reshaped.shape), tuple(reshaped.stride()))
    if key not in active[1]:
        active[1][key] = quantize(reshaped, layout, scale=scale)
    else:
        with record_region("forge::shared_activation_quantization_hit"):
            return active[1][key]
    return active[1][key]
