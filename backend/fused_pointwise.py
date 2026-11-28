"""Optional compilation of pure tensor expressions, never weight loading."""
import logging
from functools import lru_cache

import torch
import torch.nn.functional as F

from backend.args import args

logger = logging.getLogger(__name__)
_FAILED = set()


@lru_cache(maxsize=8)
def _compile(fn):
    return torch.compile(fn, fullgraph=True, dynamic=True)


def _run(fn, x, *others):
    if (not getattr(args, "fused_denoiser_pointwise", False)
            or x.device.type != "cuda" or torch.is_grad_enabled()
            or fn in _FAILED or torch.compiler.is_compiling()
            or any(getattr(args, key, None) is not None
                   for key in ("multi_gpu_anima", "parallel_cfg", "parallel_batch"))):
        return fn(x, *others)
    from torch._dynamo.exc import TorchDynamoException
    try:
        return _compile(fn)(x, *others)
    except (TorchDynamoException, ImportError) as error:
        # A missing/unsupported compiler is not a reason to change the
        # numerical expression or intercept an actual device-memory error.
        if "out of memory" in str(error).lower():
            raise
        _FAILED.add(fn)
        logger.warning("Pointwise compilation unavailable for %s; using eager: %s",
                       fn.__name__, error)
        return fn(x, *others)


def _silu_gate(x, y):
    return F.silu(x) * y


def _silu_gate_fp16_fix(x, y):
    result = F.silu(x) * y
    if result.dtype == torch.float16:
        return torch.nan_to_num(result, nan=0.0, posinf=65504, neginf=-65504)
    return result


def _affine_modulation(x, scale, shift):
    return torch.addcmul(shift, x, 1 + scale)


def _scale_modulation(x, scale):
    return x * (1 + scale.unsqueeze(1))


def silu_gate(x, y, *, fix_fp16=False):
    return _run(_silu_gate_fp16_fix if fix_fp16 else _silu_gate, x, y)


def affine_modulation(x, scale, shift):
    return _run(_affine_modulation, x, scale, shift)


def scale_modulation(x, scale):
    return _run(_scale_modulation, x, scale)
