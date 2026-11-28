"""Avoid the floating-point concatenation before an unrotated INT8 linear."""
import torch


def segmented_reference(a, b):
    """Reference for the pinned CUDA quantizer's scale, division and RN rounding."""
    af, bf = a.float(), b.float()
    maxima = [torch.nan_to_num(t.abs(), nan=0.0, posinf=torch.finfo(a.dtype).max).amax(-1, keepdim=True)
              for t in (af, bf)]
    scale = (torch.maximum(*maxima).clamp_max(torch.finfo(a.dtype).max) * (1.0 / 127.0)).clamp_min(1e-30)
    denom = scale.to(a.dtype).float()
    def quant(t):
        rounded = (t / denom).to(a.dtype).float().round()
        return torch.nan_to_num(rounded, nan=-128.0, posinf=127.0, neginf=-128.0).clamp(-128, 127).to(torch.int8)
    return torch.cat((quant(af), quant(bf)), dim=-1), scale


def try_segmented_int8_linear(linear, a, b):
    from backend.single_gpu_fastpaths import single_gpu_flag
    if (not single_gpu_flag("segmented_int8_linear") or type(a) is not torch.Tensor
            or type(b) is not torch.Tensor or not a.is_cuda or a.device != b.device
            or a.dtype != b.dtype or a.dtype not in (torch.bfloat16, torch.float16)
            or a.ndim < 2 or a.shape[:-1] != b.shape[:-1] or not a.is_contiguous()
            or not b.is_contiguous() or not a.numel() or not b.numel()
            or torch.is_grad_enabled() or torch.is_autocast_enabled(a.device.type)
            or torch.compiler.is_compiling() or torch.cuda.get_device_capability(a.device) != (8, 6)):
        return None
    from backend.quant_ops import QUANT_ALGOS, QuantizedTensor, TensorWiseINT8Layout
    from backend.sampling.invariant_cache import cache_dependencies_safe
    if (type(linear).__module__ != "backend.operations_mixed_precision"
            or type(linear).__qualname__ != "mixed_precision_ops.<locals>.MixedPrecisionOps.Linear"
            or "forward" in vars(linear) or "_forward" in vars(linear)
            or not cache_dependencies_safe(linear) or linear.weight_function or linear.bias_function
            or linear._full_precision_mm or getattr(linear, "forge_force_cast_weights", False)
            or getattr(linear, "input_scale", None) is not None
            or QUANT_ALGOS.get(getattr(linear, "quant_format", None), {}).get("quantize_input", True)):
        return None
    w = linear.weight
    if (type(w) is not QuantizedTensor or w._layout_cls != "TensorWiseINT8Layout"
            or type(w._params) is not TensorWiseINT8Layout.Params or w.device != a.device
            or w.dtype != a.dtype or not w._params.is_weight or w._params.transposed or w._params.convrot):
        return None
    raw, scale = w._qdata, w._params.scale
    k = a.shape[-1] + b.shape[-1]
    m = a.numel() // a.shape[-1]
    n = linear.out_features
    if (k != linear.in_features or m <= 2 or k % 16 or n % 16 or k > 32768
            or raw.dtype != torch.int8 or raw.device != a.device or raw.shape != (n, k)
            or not raw.is_contiguous() or raw.data_ptr() % 16
            or scale.dtype != torch.float32 or scale.device != a.device or not scale.is_contiguous()
            or scale.data_ptr() % 16 or scale.numel() not in (1, n)
            or (linear.bias is not None and (linear.bias.device != a.device or linear.bias.dtype != a.dtype
                                            or linear.bias.shape != (n,)))):
        return None
    from backend.native_int8_rows import _cuda_backend, cutlass_eligible, cutlass_rows
    cuda = _cuda_backend()
    if cuda is None or not cutlass_eligible(a, m, n, k, cuda):
        return None
    # In the pinned dependency all competing INT8 dispatch constraints use
    # device/dtype/rank/shape, not stride. This proxy avoids allocating the cat.
    from comfy_kitchen import registry
    from comfy_kitchen.exceptions import NoCapableBackendError
    proxy = a[..., :1].expand(*a.shape[:-1], k)
    kwargs = dict(x=proxy, weight=raw, weight_scale=scale, bias=linear.bias,
                  out_dtype=a.dtype, convrot=False, convrot_groupsize=256,
                  input_act=None, input_act_weight=None, input_act_eps=0.0,
                  residual=None, residual_scale=None)
    try:
        if registry.get_capable_backend("int8_linear", kwargs) != "cuda":
            return None
        kwargs["x"] = proxy.reshape(m, k) if a.ndim == 2 else a.reshape(m, -1)[:, :1].expand(m, k)
        if registry.get_capable_backend("int8_linear", kwargs) != "cuda":
            return None
    except NoCapableBackendError:
        return None
    if torch.cuda.mem_get_info(a.device)[0] < 384 * 1024**2 + m * (k + 4 + n * a.element_size()):
        return None
    try:
        from backend.segmented_int8_kernel import launch
    except ImportError:
        return None
    rows = launch(a.reshape(m, -1), b.reshape(m, -1))
    if rows is None:
        return None
    from backend.performance import record_region
    with record_region("forge::segmented_int8_linear"):
        out = cutlass_rows(a, raw, scale, linear.bias, rows, cuda)
    return out
