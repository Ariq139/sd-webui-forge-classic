"""Opt-in resident sharding for supported denoisers."""

import torch
import uuid

from backend import memory_management as mm
from backend.args import args
from backend.multi_gpu.planner import Budget, InsufficientDeviceMemory, contiguous_split, contiguous_split_with_offload
from backend.multi_gpu.residency import Residency
from backend.patcher.base import key_param_name_to_key, low_vram_patch_estimate_vram, wipe_lowvram_weight
from backend.patcher.unet import UnetPatcher


class ShardedModelPatcher(UnetPatcher):
    def add_weight_wrapper(self, name, function):
        super().add_weight_wrapper(name, function)
        self.patches_uuid = uuid.uuid4()

    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)
        self.secondary_device = torch.device("cuda", args.multi_gpu_anima)
        if self.load_device.type != "cuda" or self.load_device.index is None:
            raise ValueError("Multi-GPU requires an indexed CUDA primary device")
        if self.secondary_device == self.load_device or self.secondary_device.index >= torch.cuda.device_count():
            raise ValueError("Select a distinct, available secondary CUDA device")
        if self.offload_device.type != "cpu":
            raise ValueError("Multi-GPU requires CPU model offloading")
        if self.model.computation_dtype == torch.bfloat16:
            for device in (self.load_device, self.secondary_device):
                with torch.cuda.device(device):
                    if not torch.cuda.is_bf16_supported(including_emulation=False):
                        raise ValueError(f"{device} does not support BF16; select FP16 computation")
        if not hasattr(self.model, "_anima_shard"):
            self.model.to(self.offload_device)
            self.current_device = self.offload_device
            self.model._anima_shard = None
        self._planned_split = None
        self._planned_offloaded = frozenset()
        self._planned_staging = (0, 0)
        self._offload_retry_margin = 0

    def enable_cpu_offload_retry(self):
        """Replan a failed placement with extra GPU headroom."""
        if not self._supports_cpu_offload() or self.model._anima_shard:
            return False
        self._planned_split = None
        self._planned_offloaded = frozenset()
        self._planned_staging = (0, 0)
        self._offload_retry_margin = 512 * 1024 ** 2
        return True
    def _supports_cpu_offload(self):
        # Only enable staging for architectures whose sharded forward calls
        # each stage as a module. A replacement callback may bypass those calls.
        supported = {
            ("backend.nn.anima", "Anima"),
            ("backend.nn.krea", "SingleStreamDiT"),
            ("backend.nn.flux", "IntegratedFluxTransformer2DModel"),
            ("backend.nn.chroma", "IntegratedChromaTransformer2DModel"),
            ("backend.nn.unet", "IntegratedUNet2DConditionModel"),
            ("backend.nn.moss_image", "MOSSImage"),
            ("backend.nn.qwen", "QwenImageTransformer2DModel"),
            ("backend.nn.qwen21", "QwenImage21Transformer2DModel"),
            ("backend.nn.wan", "WanModel"),
            ("backend.nn.lumina", "NextDiT"),
            ("backend.nn.pixeldit.pid", "PidNet"),
        }
        model_type = type(self.model.diffusion_model)
        options = self.model_options.get("transformer_options", {})
        return ((model_type.__module__, model_type.__name__) in supported
                and not any(options.get(key) for key in ("patches_replace", "patches", "block_modifiers")))
    def _check_offload_aliases(self, stages, split, offloaded):
        # Moving a shared tensor with one stage would also move it for another
        # owner. Refuse such placements instead of silently changing its device.
        seen = {}
        tensors = list(self.model.named_parameters(remove_duplicate=False))
        tensors.extend(self.model.named_buffers(remove_duplicate=False))
        for name, tensor in tensors:
            location = self._device_for_name(name, split, stages)
            for index in offloaded:
                prefix = "diffusion_model." + stages[index][0]
                if name == prefix or name.startswith(prefix + "."):
                    location = self.offload_device
                    break
            previous = seen.setdefault(id(tensor), location)
            if previous != location:
                raise InsufficientDeviceMemory("Shared model tensors cannot be placed across offloaded stages")

    def _remove_offload_hooks(self):
        for handle in getattr(self.model, "_multi_gpu_offload_hooks", ()):
            handle.remove()
        self.model._multi_gpu_offload_hooks = []

    def _install_offload_hooks(self, stages, split, offloaded):
        self._remove_offload_hooks()
        handles = self.model._multi_gpu_offload_hooks
        for index in offloaded:
            block = stages[index][1]
            device = self.load_device if index < split else self.secondary_device

            def load_block(module, inputs, target=device):
                try:
                    with torch.cuda.device(target):
                        module.to(target)
                except BaseException:
                    try:
                        module.to(torch.device("cpu"))
                    except Exception:
                        mm.logger.exception("Failed to restore an offloaded block after a load error")
                    raise

            def unload_block(module, inputs, output, target=device):
                with torch.cuda.device(target):
                    module.to(torch.device("cpu"))

            handles.append(block.register_forward_pre_hook(load_block))
            handles.append(block.register_forward_hook(unload_block, always_call=True))

    def _stages(self):
        dit = self.model.diffusion_model
        if hasattr(dit, "double_blocks") and hasattr(dit, "single_blocks"):
            return [(f"double_blocks.{i}", block) for i, block in enumerate(dit.double_blocks)] + [(f"single_blocks.{i}", block) for i, block in enumerate(dit.single_blocks)]
        if hasattr(dit, "input_blocks") and hasattr(dit, "output_blocks"):
            return [(f"input_blocks.{i}", block) for i, block in enumerate(dit.input_blocks)] + [("middle_block", dit.middle_block)] + [(f"output_blocks.{i}", block) for i, block in enumerate(dit.output_blocks)]
        for name in ("transformer_blocks", "layers", "pixel_blocks"):
            if hasattr(dit, name):
                return [(f"{name}.{i}", block) for i, block in enumerate(getattr(dit, name))]
        return [(f"blocks.{i}", block) for i, block in enumerate(dit.blocks)]

    def _device_for_name(self, name, split, stages):
        prefix = "diffusion_model."
        if name.startswith(prefix):
            relative = name[len(prefix):]
            for index, (path, _) in enumerate(stages):
                if relative == path or relative.startswith(path + "."):
                    return self.load_device if index < split else self.secondary_device
        return self.load_device

    def resident_devices(self):
        return {self.load_device, self.secondary_device} if self.model._anima_shard else {self.load_device}

    def _runtime_scratch(self):
        wrapper_reserve = max((low_vram_patch_estimate_vram(self.model, key) for key in self.weight_wrapper_patches), default=0)
        cast_reserve = max(
            (param.numel() * self.model.computation_dtype.itemsize * 2
             for module in self.model.modules()
             if self.force_cast_weights or getattr(module, "parameters_manual_cast", False) or getattr(module, "_full_precision_mm", False)
             for param in module.parameters(recurse=False)),
            default=0,
        )
        return max(wrapper_reserve, cast_reserve)

    def prepare_sharded_load(self, keep_loaded, force_patch_weights=False, reserve=0):
        from backend.nn.krea import SingleStreamDiT
        if self.object_patches:
            raise RuntimeError("Multi-GPU does not support object patches")
        if self.model._anima_shard:
            reason = None
            if force_patch_weights:
                reason = "forced weight patching"
            elif self.model.current_weight_patches_uuid != self.patches_uuid:
                reason = "weight patch revision changed"
            elif getattr(self.model, "_multi_gpu_force_cast_weights", None) != self.force_cast_weights:
                reason = "weight casting changed"
            elif getattr(self.model, "_multi_gpu_offload_hooks", ()) and not self._supports_cpu_offload():
                reason = "CPU offloading is incompatible with the current transformer patches"
            residency = getattr(self.model, "_anima_residency", None)
            if reason is None:
                try:
                    if residency is None:
                        raise ValueError("Missing multi-GPU residency record")
                    residency.validate(self.model)
                except ValueError as error:
                    reason = f"invalid residency: {error}"
            if reason is None:
                scratch = self._runtime_scratch()
                staging = getattr(self.model, "_multi_gpu_offload_staging", (0, 0))
                for device, stage_bytes in zip((self.load_device, self.secondary_device), staging):
                    free = mm.get_free_memory(device)
                    required = reserve + scratch + stage_bytes + self._offload_retry_margin
                    if free < required:
                        # Reclaim other models before discarding a valid
                        # sharded placement. The active model stays protected.
                        mm.free_memory(required, device, keep_loaded=keep_loaded)
                        free = mm.get_free_memory(device)
                    if free < required:
                        reason = f"insufficient headroom on {device}: {free} bytes free, {required} required"
                        break
            if reason is not None:
                mm.logger.debug("Rebuilding sharded model: %s", reason)
                self.detach()
            else:
                mm.logger.debug("Reusing sharded model with verified residency on %s and %s", self.load_device, self.secondary_device)
        if isinstance(self.model.diffusion_model, SingleStreamDiT) and not self.model._anima_shard:
            # Krea's text encoder can otherwise occupy several GiB while the
            # denoiser is resident. Release it before budgeting both cards.
            mm.free_memory(float("inf"), self.load_device, keep_loaded=keep_loaded)

    def device_memory_requirements(self, reserve, allow_offload=False):
        runtime_reserve = self._runtime_scratch()
        if self.model._anima_shard:
            staging = getattr(self.model, "_multi_gpu_offload_staging", (0, 0))
            return {self.load_device: runtime_reserve + staging[0] + self._offload_retry_margin,
                    self.secondary_device: runtime_reserve + staging[1] + self._offload_retry_margin}
        dit = self.model.diffusion_model
        sizes = [mm.module_size(block) for _, block in self._stages()]
        fixed = mm.module_size(self.model) - sum(sizes)
        primary_free = mm.get_free_memory(self.load_device)
        secondary_free = mm.get_free_memory(self.secondary_device)
        patch_reserve = max((low_vram_patch_estimate_vram(self.model, key) for key in self.patches), default=0)
        scratch = max(runtime_reserve, patch_reserve)
        primary_budget = Budget(primary_free, int(reserve) + scratch + self._offload_retry_margin)
        secondary_budget = Budget(secondary_free, int(reserve) + scratch + self._offload_retry_margin)
        try:
            if self._offload_retry_margin:
                raise InsufficientDeviceMemory("Retrying with CPU-offloaded stages")
            split = contiguous_split(sizes, primary_budget, secondary_budget, first_fixed=fixed, policy=args.multi_gpu_split)
            offloaded, staging = frozenset(), (0, 0)
        except InsufficientDeviceMemory:
            if not (allow_offload or self._offload_retry_margin) or not self._supports_cpu_offload():
                raise
            split, offloaded, staging = contiguous_split_with_offload(sizes, primary_budget, secondary_budget, first_fixed=fixed, policy=args.multi_gpu_split)
            if self._offload_retry_margin and not offloaded:
                raise InsufficientDeviceMemory("CPU offloading could not increase the available GPU headroom")
            if offloaded:
                self._check_offload_aliases(self._stages(), split, offloaded)
            # Merged patches can leave an original CPU backup alongside the
            # offloaded patched weights. Reserve space for that second copy.
            if offloaded and self.patches:
                host_required = sum(sizes[index] for index in offloaded) + 512 * 1024 ** 2
                if mm.get_free_memory(torch.device("cpu")) < host_required:
                    raise InsufficientDeviceMemory("Not enough system memory for offloaded patched blocks")
        self._planned_split = split
        self._planned_offloaded = offloaded
        self._planned_staging = staging
        return {self.load_device: fixed + sum(size for i, size in enumerate(sizes[:split]) if i not in offloaded) + scratch + staging[0] + self._offload_retry_margin,
                self.secondary_device: sum(sizes[i] for i in range(split, len(sizes)) if i not in offloaded) + scratch + staging[1] + self._offload_retry_margin}

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        active = self.model._anima_shard
        if self.object_patches:
            raise RuntimeError("Multi-GPU does not support object patches")
        if active and self.model.current_weight_patches_uuid == self.patches_uuid and getattr(self.model, "_multi_gpu_force_cast_weights", None) == self.force_cast_weights and not force_patch_weights:
            return 0
        if active:
            self.detach()
        if self._planned_split is None:
            self.device_memory_requirements(mm.minimum_inference_memory())
        split = self._planned_split
        dit = self.model.diffusion_model
        stages = self._stages()
        offloaded = self._planned_offloaded

        def is_offloaded(name):
            prefix = "diffusion_model."
            if not name.startswith(prefix):
                return False
            relative = name[len(prefix):]
            return any(relative == stages[i][0] or relative.startswith(stages[i][0] + ".") for i in offloaded)

        try:
            self.patch_model(load_weights=False)
            installed_wrappers = set()
            load_list = self._load_list()
            # The base loader lists leaf modules. Some denoisers also keep
            # parameters directly on a parent (for example positional embeddings).
            parents = []
            for name, module in self.model.named_modules():
                if any(module.children()):
                    params = dict(module.named_parameters(recurse=False))
                    parents.append((name, module))
                    if params:
                        load_list.append((0, 0, name, module, params))
            for _, _, name, module, params in load_list:
                device = self._device_for_name(name, split, stages)
                if hasattr(module, "parameters_manual_cast"):
                    wipe_lowvram_weight(module)
                    if self.force_cast_weights:
                        module.prev_parameters_manual_cast = module.parameters_manual_cast
                        module.parameters_manual_cast = True
                module.forge_force_cast_weights = self.force_cast_weights
                for param in params:
                    key = key_param_name_to_key(name, param)
                    self.unpin_weight(key)
                    with torch.cuda.device(device):
                        self.patch_weight_to_device(key, device_to=device)
                    if key in self.weight_wrapper_patches:
                        function_name = f"{param}_function"
                        if not hasattr(module, function_name) or not hasattr(module, "parameters_manual_cast"):
                            raise RuntimeError(f"Online weight wrapper is unsupported for {key}")
                        if not hasattr(module, "prev_parameters_manual_cast"):
                            module.prev_parameters_manual_cast = module.parameters_manual_cast
                        module.parameters_manual_cast = True
                        setattr(module, function_name, list(self.weight_wrapper_patches[key]))
                        installed_wrappers.add(key)
                module.forge_patched_weights = True
                if is_offloaded(name):
                    module._apply(lambda tensor: tensor.to(self.offload_device), recurse=False)
            for name, module in parents:
                device = self.offload_device if is_offloaded(name) else self._device_for_name(name, split, stages)
                module._apply(lambda tensor: tensor.to(device), recurse=False)
            missing = self.weight_wrapper_patches.keys() - installed_wrappers
            if missing:
                raise RuntimeError(f"Online weight wrappers have no supported target: {sorted(missing)}")
            for name, child in self.model.named_children():
                if name != "diffusion_model":
                    child.to(self.load_device)
            stage_roots = {name.split(".", 1)[0] for name, _ in stages}
            for name, child in dit.named_children():
                if name not in stage_roots:
                    child.to(self.load_device)
            for index, (_, block) in enumerate(stages):
                block.to(self.offload_device if index in offloaded else (self.load_device if index < split else self.secondary_device))
            torch.cuda.synchronize(self.load_device)
            torch.cuda.synchronize(self.secondary_device)
            expected = {}
            for name, _ in list(self.model.named_parameters(remove_duplicate=False)) + list(self.model.named_buffers(remove_duplicate=False)):
                expected[name] = self.offload_device if is_offloaded(name) else self._device_for_name(name, split, stages)
            residency = Residency(expected)
            residency.validate(self.model)
            self._install_offload_hooks(stages, split, offloaded)
            dit._anima_shard_devices = (self.load_device, self.secondary_device, split)
            self.model._anima_shard = (self.load_device, self.secondary_device, split)
            self.model._anima_residency = residency
            self.model._multi_gpu_offload_staging = self._planned_staging
            self.model._multi_gpu_force_cast_weights = self.force_cast_weights
            offload_size = sum(mm.module_size(stages[index][1]) for index in offloaded)
            self.model.model_loaded_weight_memory = max(0, self.model_size() - offload_size)
            self.model.model_lowvram = bool(offloaded)
            self.model.current_weight_patches_uuid = self.patches_uuid
            self.current_device = self.load_device
            mm.logger.info("%s multi-GPU: stages 0-%d on %s, %d-%d on %s", type(dit).__name__, split - 1, self.load_device, split, len(stages) - 1, self.secondary_device)
            if offloaded:
                mm.logger.info("CPU offloading enabled for %d multi-GPU stages (%.2f GiB)", len(offloaded), offload_size / 1024 ** 3)
            return self.model.model_loaded_weight_memory
        except BaseException:
            self._remove_offload_hooks()
            try:
                self.unpatch_model(self.offload_device)
            except Exception:
                # Some tensors may still live on either device. Keep a
                # conservative residency record so the manager can retry.
                mm.logger.exception("Failed to roll back a partial multi-GPU load")
                self.model._anima_shard = (self.load_device, self.secondary_device, split)
                self.model.model_loaded_weight_memory = self.model_size()
            self.model._anima_residency = None
            self.model._multi_gpu_force_cast_weights = None
            if hasattr(dit, "_anima_shard_devices"):
                del dit._anima_shard_devices
            raise

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        loaded = self.loaded_size()
        self.detach()
        return loaded

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights and device_to is not None:
            self._remove_offload_hooks()
            for module in self.model.modules():
                wipe_lowvram_weight(module)
            if hasattr(self.model.diffusion_model, "_anima_shard_devices"):
                del self.model.diffusion_model._anima_shard_devices
            self.model._anima_shard = None
            self.model._anima_residency = None
            self.model._multi_gpu_force_cast_weights = None
            self.model._multi_gpu_offload_staging = (0, 0)
        return super().unpatch_model(device_to, unpatch_weights)

    def detach(self, unpatch_all=True):
        if not unpatch_all:
            return super().detach(unpatch_all=False)
        previous_shard = self.model._anima_shard
        dit = self.model.diffusion_model
        if hasattr(dit, "_anima_shard_devices"):
            del dit._anima_shard_devices
        self.model._anima_shard = None
        self._planned_split = None
        self._planned_offloaded = frozenset()
        self._planned_staging = (0, 0)
        self._offload_retry_margin = 0
        try:
            self.model_patches_to(self.offload_device)
        except BaseException:
            try:
                # A broken extension must not prevent model weights from
                # being moved off the GPUs during recovery.
                self.unpatch_model(self.offload_device)
            except Exception:
                mm.logger.exception("Failed to unload multi-GPU weights after callback cleanup failed")
            if previous_shard:
                self.model._anima_shard = previous_shard
                self.model._anima_residency = None
            raise
        try:
            self.unpatch_model(self.offload_device)
            return self.model
        except BaseException:
            if previous_shard:
                self.model._anima_shard = previous_shard
                self.model._anima_residency = None
            raise


ShardedAnimaPatcher = ShardedModelPatcher
