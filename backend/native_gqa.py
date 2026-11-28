"""Compact GQA only when native Flash SDPA is explicitly eligible."""
import torch

_SAFE_OPTIONS = frozenset({"cond_or_uncond", "sigmas", "cond_mark", "cond_indices", "uncond_indices",
                           "sampling_sigmas", "total_blocks", "block_type", "block_index"})


def compatible_gqa(q, k, v, mask, options):
    return (mask is None and isinstance(options, dict) and options.keys() <= _SAFE_OPTIONS
            and all(type(t) is torch.Tensor and t.ndim == 4 for t in (q, k, v))
            and q.dtype == k.dtype == v.dtype and q.dtype in (torch.float16, torch.bfloat16)
            and q.device == k.device == v.device
            and k.shape == v.shape and q.shape[0] == k.shape[0]
            and q.shape[2:] == k.shape[2:] and 0 < k.shape[1] < q.shape[1]
            and q.shape[1] % k.shape[1] == 0
            and all(t.stride(-1) == 1 for t in (q, k, v)))


def try_flash_gqa(q, k, v, mask, options, attention):
    if (not compatible_gqa(q, k, v, mask, options) or not q.is_cuda
            or torch.is_grad_enabled() or torch.is_autocast_enabled(q.device.type)):
        return None
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, True)
        if (not torch.backends.cuda.flash_sdp_enabled()
                or not torch.backends.cuda.can_use_flash_attention(params)):
            return None
    except (AttributeError, ImportError, TypeError):
        return None
    # Explicit opt-in. Do not accidentally dispatch a math kernel which
    # materializes the very KV expansion this path is intended to avoid.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return attention(q, k, v, q.shape[1], mask=None, skip_reshape=True,
                         transformer_options=options, enable_gqa=True)
