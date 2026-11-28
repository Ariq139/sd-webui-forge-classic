"""Single-job, single-slot CUDA graph replay for audited resident denoisers."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import weakref

import torch

_SLOT = ContextVar("forge_resident_graph_slot", default=None)
_CANDIDATE = ContextVar("forge_resident_graph_candidate", default=False)
_HEADROOM = 512 * 1024**2


def tensor_stamp(value, *, storage=False):
    if type(value) not in (torch.Tensor, torch.nn.Parameter) or value.layout != torch.strided:
        return None
    info = (value.device, value.dtype, tuple(value.shape), tuple(value.stride()))
    if not storage:
        return info
    try:
        version = value._version
    except RuntimeError:
        # Versionless inference weights cannot safely survive capture/replay.
        return None
    return (id(value), value.data_ptr(), version, info)


def input_signature(values):
    items = []
    for name, value in sorted(values.items()):
        if value is None:
            items.append((name, None))
        elif type(value) is torch.Tensor:
            stamp = tensor_stamp(value)
            if stamp is None or not value.is_contiguous():
                return None
            items.append((name, stamp))
        elif type(value) in (list, tuple) and all(type(x) in (int, bool, str, float) for x in value):
            items.append((name, type(value), tuple(value)))
        else:
            return None
    return tuple(items)


def static_inputs(values):
    return {k: v.detach().clone() if type(v) is torch.Tensor else list(v) if type(v) is list else v
            for k, v in values.items()}


def update_static(destination, values):
    for k, v in values.items():
        if type(v) is torch.Tensor:
            destination[k].copy_(v)


class ReplaySlot:
    """A single graph, never a shape-indexed collection of graph pools."""
    def __init__(self, budget):
        self.budget = budget
        self.disabled = False
        self.key = self.stream = self.graph = self.inputs = self.output = None
        self.owner_refs = ()

    def clear(self):
        try:
            if self.stream is not None:
                self.stream.synchronize()
        finally:
            self.key = self.stream = self.graph = self.inputs = self.output = None
            self.owner_refs = ()

    def disable(self):
        self.disabled = True
        self.clear()

    def run(self, key, device, values, invoke, *, owners=()):
        if self.disabled:
            return None
        if self.key is not None and (key != self.key or any(ref() is None for ref in self.owner_refs)):
            # Do not accumulate graph pools or recapture after a model/shape
            # change. Dead weak owners detect allocator/Python identity reuse
            # without holding offloaded model weights alive.
            self.disable()
            return None
        if self.key is None:
            self.key = key
            self.owner_refs = owners
        if self.graph is None and torch.cuda.mem_get_info(device)[0] < self.budget + _HEADROOM:
            self.clear()
            self.disabled = True
            return None
        current = torch.cuda.current_stream(device)
        if self.stream is None:
            # Warm up exactly once on the stream that will perform capture.
            self.stream = torch.cuda.Stream(device=device)
            baseline = torch.cuda.memory_allocated(device)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                result = invoke(values)
            current.wait_stream(self.stream)
            result.record_stream(current)
            # Do not reset the user's allocator peak counters. An older larger
            # peak can conservatively disable this experiment.
            if torch.cuda.max_memory_allocated(device) - baseline > self.budget:
                self.clear()
                self.disabled = True
            return result
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            if self.graph is None:
                baseline = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                self.inputs = static_inputs(values)
                self.graph = torch.cuda.CUDAGraph()
                # Capture errors propagate. Retrying an invalidated CUDA stream
                # or a device error as an eager call is not safe.
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.output = invoke(self.inputs)
                self.graph.replay()
                over_budget = max(torch.cuda.memory_allocated(device) - baseline,
                                  torch.cuda.memory_reserved(device) - reserved) > self.budget
            else:
                update_static(self.inputs, values)
                self.graph.replay()
                over_budget = False
            result = self.output.clone()
        current.wait_stream(self.stream)
        result.record_stream(current)
        if over_budget:
            self.clear()
            self.disabled = True
        return result


@contextmanager
def sampling_graph_scope():
    from backend.args import args
    from backend.single_gpu_fastpaths import single_gpu_flag
    budget = max(0, getattr(args, "resident_denoiser_graph_mib", 0)) * 1024**2
    slot = ReplaySlot(budget) if budget and single_gpu_flag("resident_denoiser_graph_mib") else None
    token = _SLOT.set(slot)
    try:
        yield
    except BaseException:
        if slot is not None:
            try:
                slot.clear()
            except Exception:
                logging.getLogger(__name__).exception("Failed to release CUDA graph after sampling error")
        raise
    else:
        if slot is not None:
            slot.clear()
    finally:
        _SLOT.reset(token)


def discard_resident_graph():
    slot = _SLOT.get()
    if slot is not None and slot.key is not None:
        slot.disable()


def graph_scope_active():
    slot = _SLOT.get()
    return slot is not None and not slot.disabled


@contextmanager
def graph_candidate(enabled):
    token = _CANDIDATE.set(enabled)
    try:
        yield
    finally:
        _CANDIDATE.reset(token)


def model_stamp(model, device):
    from backend.quant_ops import QuantizedTensor, TensorWiseINT8Layout
    from backend.sampling.invariant_cache import cache_dependencies_safe
    if not cache_dependencies_safe(model):
        return None
    stamps, owners = [], []
    for child in model.modules():
        namespace = type(child).__module__
        if (not (namespace in ("backend.nn.flux", "backend.nn.lumina", "backend.operations", "backend.operations_mixed_precision")
                 or namespace.startswith("torch.nn.modules."))
                or any(name in vars(child) for name in ("forward", "_forward", "forward_orig", "process_img"))
                or getattr(child, "weight_function", None) or getattr(child, "bias_function", None)
                or getattr(child, "forge_force_cast_weights", False) or getattr(child, "weight_scale", None) is not None):
            return None
        stamps.append((id(child), type(child), type(child).forward))
        owners.append(weakref.ref(child))
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        if value.device != device or value.requires_grad:
            return None
        owners.append(weakref.ref(value))
        if type(value) is QuantizedTensor:
            p = value._params
            if (value._layout_cls != "TensorWiseINT8Layout" or type(p) is not TensorWiseINT8Layout.Params
                    or not p.is_weight or p.transposed or p.orig_dtype not in (torch.float16, torch.bfloat16)
                    or value._qdata.device != device or p.scale.device != device):
                return None
            raw, scale = tensor_stamp(value._qdata, storage=True), tensor_stamp(p.scale, storage=True)
            if raw is None or scale is None:
                return None
            owners.extend((weakref.ref(value._qdata), weakref.ref(p.scale)))
            stamp = (id(value), raw, scale, p.orig_dtype, tuple(p.orig_shape), p.convrot, p.convrot_groupsize)
        else:
            stamp = tensor_stamp(value, storage=True)
            if stamp is None:
                return None
        stamps.append((name, stamp))
    return tuple(stamps), tuple(owners)


_SAFE_OPTIONS = {"cond_or_uncond", "sigmas", "cond_mark", "cond_indices", "uncond_indices", "sampling_sigmas"}
_INCOMPATIBLE_FLAGS = ("invariant_cache_mib", "weight_prefetch_mib", "shared_activation_quantization",
                       "fused_denoiser_pointwise", "native_int8_mlp_activation", "flux_fused_qk_rope",
                       "native_lumina_gqa", "native_krea_gqa", "target_only_output_head",
                       "native_token_assembly", "segmented_int8_linear", "pack_int8_gate_up",
                       "fused_flux_residual", "nunchaku_rope_cache", "assembled_condition_cache")


def try_resident_graph(model, x, t, context, options, extra):
    slot = _SLOT.get()
    if slot is None or slot.disabled:
        return None

    def decline():
        discard_resident_graph()
        return None

    if not _CANDIDATE.get():
        return decline()
    from backend.args import args, dynamic_args
    from backend import attention
    from backend.performance import profiler_enabled, record_region
    from backend.nn.flux import IntegratedFluxTransformer2DModel
    from backend.nn.lumina import NextDiT
    if (type(model) not in (IntegratedFluxTransformer2DModel, NextDiT)
            or type(x) is not torch.Tensor or not x.is_cuda or x.ndim != 4 or x.shape[0] != 1
            or x.dtype not in (torch.float16, torch.bfloat16) or torch.is_grad_enabled()
            or torch.is_autocast_enabled(x.device.type) or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing() or torch.cuda.get_device_capability(x.device) != (8, 6)
            or profiler_enabled() or dynamic_args.ref_latents or dynamic_args.context_handler is not None
            or getattr(model, "_anima_shard_devices", None) or attention.attention_function is not attention.attention_pytorch
            or any(getattr(args, flag, False) for flag in _INCOMPATIBLE_FLAGS)
            or type(options) is not dict or options.keys() - _SAFE_OPTIONS or extra.keys() - {"y", "guidance"}):
        return decline()
    values = dict(x=x, t=t, context=context, **extra)
    values.update({"option:" + k: v for k, v in options.items()})
    if any(type(v) is torch.Tensor and v.device != x.device for v in values.values()):
        return decline()
    signature = input_signature(values)
    if signature is None:
        return decline()
    weights = model_stamp(model, x.device)
    if weights is None:
        return decline()
    stamps, owners = weights
    key = (id(model), stamps, signature)

    def invoke(inputs):
        kwargs = {k: inputs[k] for k in extra}
        transformer_options = {k: inputs["option:" + k] for k in options}
        return model(inputs["x"], inputs["t"], context=inputs["context"], control=None,
                     transformer_options=transformer_options, **kwargs)

    with record_region("forge::resident_denoiser_graph"):
        return slot.run(key, x.device, values, invoke, owners=owners)
