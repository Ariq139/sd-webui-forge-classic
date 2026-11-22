"""Independent model replicas for experimental parallel inference."""

import copy

import torch


class UnsupportedReplica(ValueError):
    """A model cannot safely be replicated by the experimental loader."""


def clone_unet_patcher(unet, secondary):
    """Create a CPU replica with independent patches and model state."""
    from backend.operations_gguf import ParameterGGUF
    from backend import memory_management, utils
    from backend.operations import ForgeOperationsFP8, _get_disabled_quants
    from backend.patcher.base import OnlineLoRAPatch
    from backend.patcher.unet import UnetPatcher
    from backend.quant_ops import QuantizedTensor

    if type(unet) is not UnetPatcher or unet.object_patches or unet.backup:
        raise UnsupportedReplica("Model patcher or its current patch state is unsupported")
    if any(patch[-1] is not None for patches in unet.patches.values() for patch in patches):
        raise UnsupportedReplica("Custom LoRA patch functions are unsupported")
    if any(type(wrapper) is not OnlineLoRAPatch or any(patch[-1] is not None for patch in wrapper.patch)
           for wrappers in unet.weight_wrapper_patches.values() for wrapper in wrappers):
        raise UnsupportedReplica("Custom online weight wrappers are unsupported")
    if unet.controlnet_linked_list is not None or unet.extra_model_patchers_during_sampling:
        raise UnsupportedReplica("Control networks and additional models are unsupported")
    options = unet.model_options
    if set(options) - {"transformer_options"} or set(options.get("transformer_options", {})) - {"sampling_sigmas"}:
        raise UnsupportedReplica("Model extensions are unsupported")
    tensors = list(unet.model.parameters()) + list(unet.model.buffers())
    if any(t.device.type != "cpu" for t in tensors) or unet.loaded_size():
        raise UnsupportedReplica("The source model must be offloaded before cloning")
    if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in unet.model.modules()):
        raise UnsupportedReplica("Module hooks are unsupported")
    if any(isinstance(module, ForgeOperationsFP8.Linear) for module in unet.model.modules()) and not memory_management.supports_fp8_compute(secondary):
        raise UnsupportedReplica("The secondary device does not support the model's fast FP8 operations")
    uses_bf16 = unet.model.computation_dtype == torch.bfloat16 or any(
        getattr(module, "factory_kwargs", {}).get("dtype") == torch.bfloat16
        for module in unet.model.modules()
    )
    if uses_bf16:
        with torch.cuda.device(secondary):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise UnsupportedReplica("The secondary device does not support BF16")
    # Replicas are first deep-copied on the host. A large GPU alone is not
    # sufficient if the system cannot safely hold another model in RAM.
    patch_bytes = utils.nested_compute_size(unet.patches, element_size=4)
    patch_bytes += sum(utils.nested_compute_size(wrapper.patch, element_size=4)
                       for wrappers in unet.weight_wrapper_patches.values() for wrapper in wrappers)
    host_required = unet.model_size() + patch_bytes + 512 * 1024 ** 2
    if memory_management.get_free_memory(torch.device("cpu")) < host_required:
        raise UnsupportedReplica("Not enough available system memory to copy the model and LoRA state")
    try:
        memo = {}
        for tensor in tensors:
            if isinstance(tensor, ParameterGGUF):
                clone = tensor.to(device="cpu", copy=True)
                clone.gguf_cls = copy.deepcopy(tensor.gguf_cls)
                memo[id(tensor)] = clone
            elif isinstance(tensor, QuantizedTensor):
                clone = tensor.to(device="cpu", copy=True)
                memo[id(tensor)] = torch.nn.Parameter(clone, requires_grad=False) if isinstance(tensor, torch.nn.Parameter) else clone
        model = copy.deepcopy(unet.model, memo)
        disabled = _get_disabled_quants(secondary)
        for module in model.modules():
            if getattr(module, "quant_format", None) in disabled:
                module._full_precision_mm = True
        replica = UnetPatcher(model, load_device=secondary, offload_device=torch.device("cpu"))
        replica.patches = copy.deepcopy(unet.patches)
        replica.weight_wrapper_patches = copy.deepcopy(unet.weight_wrapper_patches)
        replica.patches_uuid = unet.patches_uuid
        replica.force_cast_weights = unet.force_cast_weights
    except Exception as error:
        raise UnsupportedReplica("Model or LoRA state cannot be copied independently") from error
    return replica


class ReplicaPair:
    """Own two independent inference replicas created from a CPU model.

    This loader deliberately accepts only ordinary floating-point parameters.
    Quantized parameters, shared extension state and Forge patchers need their
    own explicit lifecycle before they can be supported here.
    """

    def __init__(self, model, devices, reserve_bytes=0):
        self.models = []
        self.devices = tuple(torch.device(device) for device in devices)
        if len(self.devices) != 2 or self.devices[0] == self.devices[1]:
            raise UnsupportedReplica("Two distinct CUDA devices are required")
        if reserve_bytes < 0:
            raise ValueError("reserve_bytes must be non-negative")
        for device in self.devices:
            if device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
                raise UnsupportedReplica("Devices must be available, indexed CUDA devices")

        tensors = list(model.parameters()) + list(model.buffers())
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise UnsupportedReplica("The source model must be entirely on CPU")
        if any(type(param) is not torch.nn.Parameter or not param.is_floating_point() for param in model.parameters()):
            raise UnsupportedReplica("Quantized and custom parameters are not supported")
        if any(module._forward_hooks or module._forward_pre_hooks or module._backward_hooks for module in model.modules()):
            raise UnsupportedReplica("Module hooks are not supported")
        size = sum(t.numel() * t.element_size() for t in tensors)
        for device in self.devices:
            free, _ = torch.cuda.mem_get_info(device)
            if free < size + reserve_bytes:
                raise UnsupportedReplica(f"Insufficient free memory for a full replica on {device}")

        try:
            for device in self.devices:
                replica = copy.deepcopy(model).eval().requires_grad_(False)
                self.models.append(replica)
                replica.to(device)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.models.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
