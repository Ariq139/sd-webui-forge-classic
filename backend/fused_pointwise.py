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


def fusion_enabled(x):
    return (getattr(args, "fused_denoiser_pointwise", False)
            and x.device.type == "cuda" and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            and not torch.is_autocast_enabled(x.device.type)
            and not any(getattr(args, key, None) is not None
                        for key in ("multi_gpu_anima", "parallel_cfg", "parallel_batch")))


def can_fuse_silu(module, x):
    return (fusion_enabled(x) and type(module) is torch.nn.SiLU and not module.inplace
            and not module._forward_hooks and not module._forward_pre_hooks
            and not torch.nn.modules.module._global_forward_hooks
            and not torch.nn.modules.module._global_forward_pre_hooks)


def _device_failure(error):
    # Compiler wrappers can hide asynchronous device failures or OOM in an
    # inner exception. Do not convert those into an eager retry.
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        message = str(current).lower()
        if (isinstance(current, torch.OutOfMemoryError)
                or any(marker in message for marker in ("out of memory", "cuda error",
                       "device-side assert", "illegal memory access", "cublas_status", "cudnn_status"))):
            return True
        pending.extend((current.__cause__, current.__context__, getattr(current, "inner_exception", None)))
    return False


def _run(fn, x, *others):
    if not fusion_enabled(x) or fn in _FAILED:
        return fn(x, *others)
    from torch._dynamo.exc import TorchDynamoException
    try:
        return _compile(fn)(x, *others)
    except (TorchDynamoException, ImportError) as error:
        # A missing/unsupported compiler is not a reason to change the
        # numerical expression or intercept an actual device-memory error.
        if _device_failure(error):
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


def _layer_norm_affine(x, scale, shift, shape, weight, bias, eps):
    normalized = F.layer_norm(x, shape, weight, bias, eps)
    return torch.addcmul(shift, normalized, 1 + scale)


def _rms_norm_scale(x, scale, shape, weight, eps):
    normalized = F.rms_norm(x, shape, weight, eps)
    return normalized * (1 + scale.unsqueeze(1))


def _resident_native_norm(norm, x, rms):
    if not fusion_enabled(x):
        return False
    from backend.operations import ForgeOperations
    supported = (torch.nn.RMSNorm, ForgeOperations.RMSNorm) if rms else (torch.nn.LayerNorm, ForgeOperations.LayerNorm)
    if (type(norm) not in supported or getattr(norm, "add", False)
            or getattr(norm, "weight_function", None) or getattr(norm, "bias_function", None)
            or norm._forward_hooks or norm._forward_pre_hooks
            or torch.nn.modules.module._global_forward_hooks
            or torch.nn.modules.module._global_forward_pre_hooks):
        return False
    return all(p is None or (type(p) in (torch.Tensor, torch.nn.Parameter)
                             and p.device == x.device and p.dtype == x.dtype)
               for p in (getattr(norm, "weight", None), getattr(norm, "bias", None)))


def norm_affine(norm, x, scale, shift):
    if _resident_native_norm(norm, x, rms=False):
        return _run(_layer_norm_affine, x, scale, shift, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
    return affine_modulation(norm(x), scale, shift)


def norm_scale(norm, x, scale):
    if _resident_native_norm(norm, x, rms=True):
        return _run(_rms_norm_scale, x, scale, norm.normalized_shape, norm.weight, norm.eps)
    return scale_modulation(norm(x), scale)
