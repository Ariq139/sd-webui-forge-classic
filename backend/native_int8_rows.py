"""Scoped reuse of the pinned CUDA INT8 producer and CUTLASS epilogue.

This is not a new GEMM or a packed-weight replacement. Only the ordinary
SM86 CUTLASS route of Comfy-Kitchen 0.2.40 is eligible.
"""
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch


MAX_ROW_BYTES = 32 * 1024**2


@lru_cache(maxsize=1)
def _cuda_backend():
    try:
        if version("comfy-kitchen") != "0.2.40":
            return None
        from comfy_kitchen.backends import cuda
        required = ("quantize_int8_rowwise", "_convrot_fused_shared_memory_fits",
                    "_CONVROT_FUSED_MAX_K", "_DISABLE_CUTLASS_INT8",
                    "_cuda_device_supports_cutlass_int8_dequant",
                    "_cuda_device_should_use_turing_kernels", "_prefer_turing_fused_int8",
                    "_prefer_cublas_int8_fallback", "_int8_weight_scale_arg",
                    "_gemm_vector_arg", "_empty_cuda_tensor", "_wrap_for_dlpack",
                    "_input_act_code", "DTYPE_TO_CODE", "_C")
        if not all(hasattr(cuda, name) for name in required):
            return None
        if not all(hasattr(cuda._C, name) for name in
                   ("cutlass_int8_dequant", "quantize_int8_rowwise_convrot64")):
            return None
        return cuda
    except (ImportError, PackageNotFoundError):
        return None


def row_key(source, convrot, stream):
    try:
        revision = source._version
    except RuntimeError:
        revision = None  # Only used in an audited, read-only forward group.
    return (stream, convrot, revision, source.data_ptr(), source.dtype,
            source.device, tuple(source.shape), tuple(source.stride()))


def shared_rows(source, convrot, stream, cache, quantize, *, budget=None):
    key = row_key(source, convrot, stream)
    if key in cache:
        return cache[key]
    m = source.numel() // source.shape[-1]
    size = source.numel() + 4 * m
    budget = MAX_ROW_BYTES if budget is None else min(budget, MAX_ROW_BYTES)
    if size > budget or (cache and size + sum(
            q.numel() * q.element_size() + s.numel() * s.element_size()
            for q, s in cache.values()) > budget):
        return None
    value = quantize(source.reshape(m, source.shape[-1]), convrot)
    cache[key] = value
    return value


def _native_rows(cuda, flat, convrot, stream):
    if not convrot:
        return cuda.quantize_int8_rowwise(flat)
    # Same fused ConvRot-256 producer, no activation or stochastic rounding,
    # as the pinned int8_linear implementation. Do not substitute the older
    # non-fused rotation kernel, which has a different rounding contract.
    q = torch.empty_like(flat, dtype=torch.int8)
    scales = torch.empty((flat.shape[0], 1), device=flat.device, dtype=torch.float32)
    empty = cuda._empty_cuda_tensor(flat.device, flat.dtype)
    wrap = cuda._wrap_for_dlpack
    cuda._C.quantize_int8_rowwise_convrot64(
        wrap(flat), wrap(q), wrap(scales), 256, False,
        cuda._input_act_code(None), 0, wrap(empty), 0.0, stream)
    return q, scales


def cutlass_eligible(source, m, n, k, cuda):
    return (source.is_cuda and source.dtype in (torch.float16, torch.bfloat16)
            and torch.cuda.get_device_capability(source.device) == (8, 6)
            and m > 2 and k > 0 and n > 0 and k % 16 == n % 16 == 0
            and not cuda._DISABLE_CUTLASS_INT8
            and cuda._cuda_device_supports_cutlass_int8_dequant(source)
            and not cuda._prefer_cublas_int8_fallback(m, n, k, source.get_device())
            and not (cuda._prefer_turing_fused_int8(m, n, k)
                     and cuda._cuda_device_should_use_turing_kernels(source.get_device())))


def cutlass_rows(source, weight, scale, bias, rows, cuda, *, destination_factory=None):
    q, row_scale = rows
    m, n = q.shape[0], weight.shape[0]
    stream = torch.cuda.current_stream(source.device).cuda_stream
    weight_scale = cuda._int8_weight_scale_arg(scale, source.device)
    if weight_scale.numel() != n:
        weight_scale = weight_scale.reshape(1).expand(n).contiguous()
    bias_arg = (cuda._gemm_vector_arg(bias, source.device, source.dtype) if bias is not None
                else cuda._empty_cuda_tensor(source.device, source.dtype))
    out = destination_factory() if destination_factory is not None else None
    if (out is None or out.shape != (m, n) or out.dtype != source.dtype
            or out.device != source.device or not out.is_contiguous() or out.data_ptr() % 16):
        out = torch.empty((m, n), device=source.device, dtype=source.dtype)
    wrap = cuda._wrap_for_dlpack
    used = cuda._C.cutlass_int8_dequant(
        wrap(q), wrap(weight), wrap(row_scale), wrap(weight_scale), wrap(bias_arg),
        wrap(out), cuda.DTYPE_TO_CODE[source.dtype], stream)
    return out.reshape(*source.shape[:-1], n) if used else None


def try_native_cutlass(source, weight, scale, bias, convrot, cache, cuda, *, destination_factory=None, budget=None):
    if (not source.is_cuda or source.dtype not in (torch.float16, torch.bfloat16)
            or not source.is_contiguous() or source.data_ptr() % 16 or weight.data_ptr() % 16):
        return None
    n, k = weight.shape
    if not k:
        return None
    m = source.numel() // k
    if not cutlass_eligible(source, m, n, k, cuda):
        return None
    flat = source.reshape(m, k)
    if convrot and (k < 256 or k % 256 or k > cuda._CONVROT_FUSED_MAX_K
                    or not cuda._convrot_fused_shared_memory_fits(flat, k, 256)):
        return None
    stream = torch.cuda.current_stream(source.device).cuda_stream
    rows = shared_rows(source, convrot, stream, cache,
                       lambda x, rotated: _native_rows(cuda, x, rotated, stream), budget=budget)
    if rows is None:
        return None
    out = cutlass_rows(source, weight, scale, bias, rows, cuda, destination_factory=destination_factory)
    if out is None:
        cache.pop(row_key(source, convrot, stream), None)
    return out
