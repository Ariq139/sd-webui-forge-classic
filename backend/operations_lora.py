"""Narrow activation-space execution for standard online linear LoRAs."""
import math
import torch
import torch.nn.functional as functional
from backend import memory_management

_activation_lora_logged = False


def activation_space_lora_patches(weight_functions, bias_functions, in_features, out_features):
    global _activation_lora_logged
    """Return validated adapters, or None so callers can use legacy merging."""
    from backend.args import args

    if not args.activation_space_lora or not weight_functions or bias_functions:
        return None

    from backend.patcher.base import OnlineLoRAPatch
    from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter

    result = []
    for weight_function in weight_functions:
        if type(weight_function) is not OnlineLoRAPatch or len(weight_function.patch) != 1:
            return None
        patch = weight_function.patch[0]
        if not isinstance(patch, (tuple, list)) or len(patch) != 5:
            return None
        strength, adapter, strength_model, offset, function = patch
        if (type(adapter) is not LoRAAdapter or strength_model != 1.0 or offset is not None
                or function is not None):
            return None
        up, down, alpha, mid, dora_scale, reshape = adapter.weights
        if (mid is not None or dora_scale is not None or reshape is not None
                or not isinstance(up, torch.Tensor) or not isinstance(down, torch.Tensor)
                or up.ndim != 2 or down.ndim != 2 or down.shape[0] <= 0
                or down.shape[1] != in_features or up.shape[0] != out_features
                or up.shape[1] != down.shape[0]):
            return None
        if alpha is not None and (not isinstance(alpha, (int, float)) or not math.isfinite(alpha)):
            return None
        if not isinstance(strength, (int, float)) or not math.isfinite(strength):
            return None
        scale = strength * (alpha / down.shape[0] if alpha is not None else 1.0)
        if not math.isfinite(scale):
            return None
        result.append((up, down, scale))
    if result and not _activation_lora_logged:
        memory_management.logger.info(
            "Activation-space LoRA is active for compatible linear adapters; unsupported patches use the standard path"
        )
        _activation_lora_logged = True
    return result


def apply_activation_space_lora(output, inputs, patches):
    if not patches:
        return output
    for up, down, scale in patches:
        if scale == 0:
            continue
        down = memory_management.cast_to_device(down, inputs.device, inputs.dtype)
        up = memory_management.cast_to_device(up, inputs.device, inputs.dtype)
        hidden = functional.linear(inputs, down)
        delta = functional.linear(hidden, up)
        if scale != 1:
            delta = delta * scale
        output = output + delta.to(dtype=output.dtype)
    return output
