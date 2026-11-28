"""Optional Triton producer reading two segments without a dense concatenation."""
import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError


@triton.jit
def _quantize(A, B, Q, S, KA: tl.constexpr, KB: tl.constexpr, MAXVAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    va = tl.load(A + row * KA + col, mask=col < KA, other=0).to(tl.float32)
    vb = tl.load(B + row * KB + col - KA, mask=(col >= KA) & (col < KA + KB), other=0).to(tl.float32)
    v = tl.where(col < KA, va, vb)
    maximum = tl.max(tl.where(v == v, tl.abs(v), 0.0), axis=0)
    scale = tl.maximum(tl.minimum(maximum, MAXVAL) * (1.0 / 127.0), 1e-30)
    tl.store(S + row, scale)
    denom = scale.to(A.dtype.element_ty).to(tl.float32)
    scaled = tl.div_rn(v, denom).to(A.dtype.element_ty).to(tl.float32)
    scaled = tl.where(scaled == scaled, scaled, -128.0)
    scaled = tl.minimum(tl.maximum(scaled, -128.0), 127.0)
    lower = tl.floor(scaled)
    fraction = scaled - lower
    rounded = tl.where((fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0)), lower + 1.0, lower)
    tl.store(Q + row * (KA + KB) + col, rounded.to(tl.int8), mask=col < KA + KB)


_FAILED = set()


def launch(a, b):
    m, ka = a.shape
    kb = b.shape[-1]
    key = (a.device, a.dtype, ka, kb)
    if key in _FAILED or not hasattr(tl, "div_rn"):
        return None
    q = torch.empty((m, ka + kb), dtype=torch.int8, device=a.device)
    scale = torch.empty((m, 1), dtype=torch.float32, device=a.device)
    block = triton.next_power_of_2(ka + kb)
    try:
        _quantize[(m,)](a, b, q, scale, ka, kb, torch.finfo(a.dtype).max, block,
                        num_warps=8 if block >= 4096 else 4, enable_fp_fusion=False)
    except CompilationError:
        _FAILED.add(key)
        return None
    return q, scale
