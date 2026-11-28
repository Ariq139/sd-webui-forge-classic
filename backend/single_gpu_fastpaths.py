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
