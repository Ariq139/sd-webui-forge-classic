"""Load-time CPU INT8 coalescing; combined GPU execution is transient."""
from collections import Counter

import torch

from backend.projection_packing import coalesce_rows, joined_rows, MAX_PACK_BYTES


def _operand(layer):
    from backend.quant_ops import QUANT_ALGOS, QuantizedTensor, TensorWiseINT8Layout
    from backend.sampling.invariant_cache import cache_dependencies_safe
    if (type(layer).__module__ != "backend.operations_mixed_precision"
            or type(layer).__qualname__ != "mixed_precision_ops.<locals>.MixedPrecisionOps.Linear"
            or "forward" in vars(layer) or "_forward" in vars(layer)
            or not cache_dependencies_safe(layer)
            or layer.weight_function or layer.bias_function or layer._full_precision_mm
            or getattr(layer, "forge_force_cast_weights", False)
            or getattr(layer, "input_scale", None) is not None
            or QUANT_ALGOS.get(getattr(layer, "quant_format", None), {}).get("quantize_input", True)):
        return None
    w = layer.weight
    if (type(w) is not QuantizedTensor or w.device.type != "cpu"
            or w._layout_cls != "TensorWiseINT8Layout" or type(w._params) is not TensorWiseINT8Layout.Params):
        return None
    p = w._params
    q, s = w._qdata, p.scale
    if (not p.is_weight or p.transposed or p.convrot_groupsize != 256
            or p.orig_dtype not in (torch.bfloat16, torch.float16)
            or type(q) is not torch.Tensor or q.dtype != torch.int8 or q.device.type != "cpu"
            or not q.is_contiguous() or tuple(q.shape) != (layer.out_features, layer.in_features)
            or not all(n > 0 and n % 16 == 0 for n in q.shape)
            or type(s) is not torch.Tensor or s.device.type != "cpu" or s.dtype != torch.float32
            or not s.is_contiguous() or s.numel() not in (1, q.shape[0])):
        return None
    bias = layer.bias
    if bias is not None and (type(bias) is not torch.nn.Parameter or bias.device.type != "cpu"
                             or bias.dtype != p.orig_dtype or tuple(bias.shape) != (q.shape[0],)):
        return None
    return w, q, s, bias


def _pair(first, second):
    a, b = _operand(first), _operand(second)
    if a is None or b is None:
        return None
    pa, pb = a[0]._params, b[0]._params
    if (a[1].shape != b[1].shape or pa.orig_dtype != pb.orig_dtype
            or pa.convrot != pb.convrot or (a[3] is None) != (b[3] is None)):
        return None
    return a, b


def pack_offloaded_gate_up(patcher, offloaded):
    from backend.single_gpu_fastpaths import single_gpu_flag
    from backend.args import args
    if (not single_gpu_flag("pack_int8_gate_up") or getattr(args, "pin_quantized_weights", False)
            or patcher.force_cast_weights or patcher.object_patches or patcher.weight_wrapper_patches):
        return
    from backend.nn.lumina import FeedForward, NextDiT
    from backend.nn.flux import YakMLP, IntegratedFluxTransformer2DModel
    from backend.nn.krea import SwiGLU, SingleStreamDiT
    from backend.quantized_storage import _layout_types
    from backend.modules.k_model import KModel
    from backend.sampling.invariant_cache import cache_dependencies_safe
    if (type(patcher.model) is not KModel
            or type(patcher.model.diffusion_model) not in (IntegratedFluxTransformer2DModel, NextDiT, SingleStreamDiT)
            or not cache_dependencies_safe(patcher.model.diffusion_model) or _layout_types() is None):
        return
    offloaded_names = {entry[1] for entry in offloaded}
    pairs = {FeedForward: ("w1", "w3"), YakMLP: ("gate_proj", "up_proj"), SwiGLU: ("gate", "up")}
    named = list(patcher.model.named_parameters(remove_duplicate=False))
    identities = Counter(id(p) for _, p in named)
    storage_counts = Counter(p._qdata.untyped_storage().data_ptr() for _, p in named
                             if type(p) is _layout_types()[0] and p.device.type == "cpu")
    del named
    for name, parent in patcher.model.named_modules():
        names = pairs.get(type(parent))
        if names is None:
            continue
        full_names = [f"{name}.{n}" if name else n for n in names]
        keys = [f"{n}.{key}" for n in full_names for key in ("weight", "bias")]
        if (not all(n in offloaded_names for n in full_names)
                or any(key in patcher.pinned or key in patcher.patches or key in patcher.backup for key in keys)):
            continue
        first, second = (getattr(parent, n) for n in names)
        pair = _pair(first, second)
        if pair is None:
            continue
        a, b = pair
        if joined_rows(a[1], b[1]) is not None:
            continue
        if any(identities[id(w)] != 1 or storage_counts[q.untyped_storage().data_ptr()] != 1
               for w, q, _, _ in pair):
            continue
        import psutil
        if psutil.virtual_memory().available < 1024**3 + a[1].numel() + b[1].numel():
            continue
        views = coalesce_rows(a[1], b[1])
        if views is None:
            continue
        new = [torch.nn.Parameter(type(w)(view, w._layout_cls, w._params), requires_grad=False)
               for (w, _, _, _), view in zip(pair, views)]
        if joined_rows(new[0]._qdata, new[1]._qdata) is None:
            continue
        # Native Parameter detach clones scale metadata, not the qdata view.
        # No packed GPU weight is retained, and the named parameter keys remain.
        first.weight, second.weight = new


def try_packed_gate_up(source, first, second):
    from backend.single_gpu_fastpaths import single_gpu_flag
    from backend.args import args
    if (not single_gpu_flag("pack_int8_gate_up") or getattr(args, "pin_quantized_weights", False)
            or type(source) is not torch.Tensor or not source.is_cuda or not source.is_contiguous()
            or source.ndim < 2 or not source.shape[-1] or torch.is_grad_enabled()
            or torch.is_autocast_enabled(source.device.type) or torch.compiler.is_compiling()
            or torch.cuda.get_device_capability(source.device) != (8, 6)):
        return None
    pair = _pair(first, second)
    if pair is None:
        return None
    a, b = pair
    raw = joined_rows(a[1], b[1])
    m = source.numel() // source.shape[-1]
    n, k = a[1].shape
    if (raw is None or raw.numel() > MAX_PACK_BYTES or source.shape[-1] != k or m <= 2
            or source.dtype != a[0]._params.orig_dtype
            or torch.cuda.mem_get_info(source.device)[0] < 512 * 1024**2 + raw.numel() + m * (k + 4*n + 4)):
        return None
    from backend.native_int8_rows import _cuda_backend, try_native_cutlass
    cuda = _cuda_backend()
    if cuda is None:
        return None
    from comfy_kitchen import registry
    from comfy_kitchen.exceptions import NoCapableBackendError
    convrot = a[0]._params.convrot
    weight = raw.to(source.device)
    scale = torch.cat([s.reshape(-1).expand(n) for _, _, s, _ in pair]).to(source.device)
    bias = None if a[3] is None else torch.cat((a[3], b[3])).to(source.device)
    try:
        for w, s, z in ((weight[:n], scale[:n], None if bias is None else bias[:n]),
                        (weight[n:], scale[n:], None if bias is None else bias[n:]),
                        (weight, scale, bias)):
            kw = dict(x=source, weight=w, weight_scale=s, bias=z, out_dtype=source.dtype,
                      convrot=convrot, convrot_groupsize=256, input_act=None, input_act_weight=None,
                      input_act_eps=0.0, residual=None, residual_scale=None)
            if registry.get_capable_backend("int8_linear", kw) != "cuda":
                return None
    except NoCapableBackendError:
        return None
    from backend.performance import record_region
    with record_region("forge::packed_gate_up"):
        out = try_native_cutlass(source, weight, scale, bias, convrot, {}, cuda)
    return None if out is None else out.split(n, dim=-1)
