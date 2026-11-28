"""Small, opt-in native single-GPU sampling fast paths."""
import torch


def single_gpu_flag(name):
    from backend.args import args
    return (getattr(args, name, False)
            and not any(getattr(args, key, None) is not None
                        for key in ("multi_gpu_anima", "parallel_cfg", "parallel_batch")))


def schedule_options_safe(options):
    return isinstance(options, dict) and options.keys() <= {"sampling_sigmas"}


def uniform_unpatchify(x, img_size, cap_size, patch, channels):
    if (x.ndim != 3 or not img_size or len(img_size) != x.shape[0]
            or len(cap_size) != x.shape[0]
            or any(size != img_size[0] for size in img_size)
            or any(size != cap_size[0] for size in cap_size)):
        return None
    h, w = img_size[0]
    begin = cap_size[0]
    end = begin + (h // patch) * (w // patch)
    if h % patch or w % patch or begin < 0 or end > x.shape[1]:
        return None
    return (x[:, begin:end].reshape(x.shape[0], h // patch, w // patch,
                                   patch, patch, channels)
            .permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], channels, h, w).contiguous())


def requested_layer_stop(request, layer_count):
    """Llama2_ collects list entries BEFORE layer i, scalar entries AFTER it."""
    if isinstance(request, list):
        if request and all(type(i) is int and 0 <= i < layer_count for i in request):
            return max(request), True
    elif type(request) is int:
        index = request if request >= 0 else layer_count + request
        if 0 <= index < layer_count - 1:
            return index, False
    return None


def versioned_schedule(sigmas):
    # Inference tensors lack version counters. Preserve device/dtype and make
    # a normal, versioned schedule so callback writes invalidate snapshots.
    if (single_gpu_flag("static_sigma_control") and sigmas.is_cuda
            and sigmas.is_inference()):
        with torch.inference_mode(False), torch.no_grad():
            return sigmas.clone()
    return sigmas


class SigmaControl:
    """One CPU snapshot per unchanged schedule; never change GPU arithmetic."""
    def __init__(self, sigmas):
        self.sigmas = sigmas
        self.values = None
        self.version = None
        self.thresholds = {}
        self.enabled = (single_gpu_flag("static_sigma_control") and sigmas.is_cuda
                        and type(sigmas) is torch.Tensor and sigmas.ndim == 1)

    def value(self, index):
        if not self.enabled:
            return self.sigmas[index]
        try:
            version = self.sigmas._version
        except RuntimeError:
            return self.sigmas[index]
        if self.values is None or version != self.version:
            self.values = self.sigmas.detach().cpu().tolist()
            self.version = version
        return self.values[index]

    def in_range(self, index, lower, upper):
        value = self.value(index)
        if isinstance(value, torch.Tensor):
            return lower <= value <= upper
        # Torch comparisons cast Python scalars to the schedule dtype.
        # Preserve that behavior at non-representable churn boundaries.
        key = (lower, upper)
        bounds = self.thresholds.get(key)
        if bounds is None:
            bounds = tuple(torch.tensor(bound, dtype=self.sigmas.dtype, device="cpu").item()
                           for bound in key)
            self.thresholds[key] = bounds
        return bounds[0] <= value <= bounds[1]


def reference_signature(refs):
    """Unversioned or subclassed references retain the original path."""
    result = []
    for ref in refs:
        if type(ref) is not torch.Tensor:
            return None
        try:
            version = ref._version
        except RuntimeError:
            return None
        result.append((id(ref), version, ref.device, ref.dtype, tuple(ref.shape), tuple(ref.stride())))
    return tuple(result)


def dense_tokenwise_head(module):
    """Only ordinary dense weights and native tokenwise modules may be sliced."""
    from backend.operations import ForgeOperations
    from backend.sampling.invariant_cache import cache_dependencies_safe
    if not cache_dependencies_safe(module):
        return False
    allowed = (torch.nn.Linear, ForgeOperations.Linear, torch.nn.LayerNorm,
               ForgeOperations.LayerNorm, torch.nn.RMSNorm, ForgeOperations.RMSNorm,
               torch.nn.SiLU, torch.nn.Sequential)
    for child in module.modules():
        if child is not module and type(child) not in allowed:
            return False
        if (getattr(child, "weight_function", None) or getattr(child, "bias_function", None)
                or getattr(child, "weight_scale", None) is not None
                or getattr(child, "forge_force_cast_weights", False)):
            return False
        if any(type(p) is not torch.nn.Parameter or not p.is_floating_point()
               for p in child.parameters(recurse=False)):
            return False
    return True


def version_reference_latents(refs):
    """Create versioned immutable job inputs once, not once per denoiser call."""
    if not single_gpu_flag("flux_reference_cache"):
        return list(refs)
    result = []
    for ref in refs:
        if type(ref) is torch.Tensor and ref.is_inference():
            with torch.inference_mode(False), torch.no_grad():
                ref = ref.clone()
        result.append(ref)
    return result
