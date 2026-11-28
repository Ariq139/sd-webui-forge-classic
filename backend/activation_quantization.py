"""Short-lived, opt-in reuse of deterministic unscaled INT8 quantization."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from importlib.metadata import version

import torch

from backend.args import args
from backend.performance import record_region

_ACTIVE = ContextVar("forge_shared_activation_quantization", default=None)
_NATIVE_CACHE = object()


@contextmanager
def shared_activation_scope(source, modules=()):
    enabled = (getattr(args, "shared_activation_quantization", False)
               and type(source) is torch.Tensor and source.is_cuda
               and not torch.is_grad_enabled()
               and not torch.is_autocast_enabled(source.device.type)
               and not torch.compiler.is_compiling()
               and not torch.nn.modules.module._global_forward_hooks
               and not torch.nn.modules.module._global_forward_pre_hooks
               and not any(m.training or m._forward_hooks or m._forward_pre_hooks
                           or getattr(m, "weight_function", None) or getattr(m, "bias_function", None)
                           for m in modules)
               and not any(getattr(args, key, None) is not None
                           for key in ("multi_gpu_anima", "parallel_cfg", "parallel_batch")))
    # Native row reuse is narrower than the original eager experiment: every
    # projection must be the audited mixed-precision Linear implementation.
    cache = {}
    if (enabled and getattr(args, "native_shared_int8_rows", False) and modules
            and all(type(m).__module__ == "backend.operations_mixed_precision"
                    and type(m).__qualname__ == "mixed_precision_ops.<locals>.MixedPrecisionOps.Linear"
                    and "forward" not in vars(m) and "_forward" not in vars(m)
                    for m in modules)):
        cache[_NATIVE_CACHE] = {}
    # Only retain a single activation within an explicitly read-only group.
    token = _ACTIVE.set((source, cache) if enabled else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _source_version(source):
    try:
        return source._version
    except RuntimeError:
        return None


def quantize_shared_activation(source, reshaped, layout, scale, quantize):
    active = _ACTIVE.get()
    if (active is None or active[0] is not source
            or layout != "TensorWiseINT8Layout" or scale is not None):
        return quantize(reshaped, layout, scale=scale)
    key = (layout, source.dtype, source.device, tuple(reshaped.shape), tuple(reshaped.stride()), _source_version(source))
    if key not in active[1]:
        active[1][key] = quantize(reshaped, layout, scale=scale)
    else:
        with record_region("forge::shared_activation_quantization_hit"):
            return active[1][key]
    return active[1][key]


@lru_cache(maxsize=1)
def _eager_int8_backend():
    # Match the exact arithmetic and dispatch of Forge's pinned dependency.
    # Do not silently substitute this composition on a different version.
    from importlib.metadata import PackageNotFoundError
    try:
        if version("comfy-kitchen") != "0.2.40":
            return None
        from comfy_kitchen.backends.eager import quantization
        return quantization
    except (ImportError, PackageNotFoundError):
        return None


def _eager_linear_from_shared_rows(source, weight, weight_scale, bias, cache, quantize, matmul, *, stream_key=None):
    """Reuse the row quantization, retaining the pinned eager INT8 epilogue.

    Matches comfy-kitchen 0.2.40 eager int8_linear (Apache-2.0).
    The native/triton implementations are deliberately not replaced.
    """
    flat = source.reshape(-1, source.shape[-1])
    key = ("eager_int8_rows", stream_key, _source_version(source), source.dtype,
           source.device, tuple(source.shape), tuple(source.stride()))
    rows = cache.get(key)
    if rows is None:
        rows = quantize(flat)
        cache[key] = rows
    else:
        with record_region("forge::shared_int8_rowwise_hit"):
            pass
    q, row_scale = rows
    acc = matmul(q, weight.T.contiguous())
    m, n = acc.shape
    chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))
    scales = weight_scale.to(device=source.device, dtype=torch.float32).reshape(1, -1)
    parts = []
    for i in range(0, m, chunk_size):
        end = min(i + chunk_size, m)
        chunk = acc[i:end].float()
        combined = row_scale[i:end].to(device=chunk.device, dtype=torch.float32) * scales
        parts.append((chunk * combined).to(source.dtype))
    output = torch.cat(parts, dim=0)
    if bias is not None:
        output = output + bias.to(device=output.device, dtype=output.dtype).reshape(1, -1)
    return output.reshape(*source.shape[:-1], weight.shape[0])


def try_shared_int8_linear(source, weight, weight_scale, bias, *, convrot=False, transposed=False):
    """Reuse rows only when the ordinary linear selects an audited backend."""
    active = _ACTIVE.get()
    native_cache = active[1].get(_NATIVE_CACHE) if active is not None else None
    if (active is None or active[0] is not source or not source.is_cuda
            or (convrot and native_cache is None) or transposed or source.ndim < 2 or not source.is_contiguous()
            or source.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or weight.dtype != torch.int8 or weight.ndim != 2 or not weight.is_contiguous()
            or weight.device != source.device or weight_scale.device != source.device
            or source.shape[-1] != weight.shape[-1] or not source.numel() or not weight.numel()
            or weight_scale.numel() not in (1, weight.shape[0])
            or (bias is not None and (bias.ndim != 1 or bias.numel() != weight.shape[0]))):
        return None
    eager = _eager_int8_backend()
    if eager is None:
        return None
    from comfy_kitchen import registry
    from comfy_kitchen.exceptions import NoCapableBackendError

    kwargs = dict(x=source, weight=weight, weight_scale=weight_scale, bias=bias,
                  out_dtype=source.dtype, convrot=convrot, convrot_groupsize=256,
                  input_act=None, input_act_weight=None, input_act_eps=0.0,
                  residual=None, residual_scale=None)
    try:
        selected = registry.get_capable_backend("int8_linear", kwargs)
        if selected != "eager" and (selected != "cuda" or native_cache is None):
            return None
        kwargs["x"] = source.reshape(-1, source.shape[-1])
        if registry.get_capable_backend("int8_linear", kwargs) != selected:
            return None
    except NoCapableBackendError:
        return None
    if selected == "cuda" and native_cache is not None:
        from backend.native_int8_rows import _cuda_backend, try_native_cutlass
        cuda = _cuda_backend()
        if (cuda is None or weight_scale.dtype != torch.float32
                or not weight_scale.is_contiguous() or weight_scale.data_ptr() % 16):
            return None
        with record_region("forge::shared_int8_cuda_linear"):
            return try_native_cutlass(source, weight, weight_scale, bias, convrot, native_cache, cuda)
    if selected != "eager" or convrot:
        return None
    with record_region("forge::shared_int8_eager_linear"):
        return _eager_linear_from_shared_rows(source, weight, weight_scale, bias, active[1],
                                              eager.quantize_int8_rowwise, eager.mm_int8,
                                              stream_key=torch.cuda.current_stream(source.device).cuda_stream)
