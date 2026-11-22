"""Opt-in resident sharding for supported denoisers."""

import torch
import uuid

from backend import memory_management as mm
from backend.args import args
from backend.multi_gpu.planner import Budget, contiguous_split
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
                for device in (self.load_device, self.secondary_device):
                    free = mm.get_free_memory(device)
                    if free < reserve + scratch:
                        # Reclaim other models before discarding a valid
                        # sharded placement. The active model stays protected.
                        mm.free_memory(reserve + scratch, device, keep_loaded=keep_loaded)
                        free = mm.get_free_memory(device)
                    if free < reserve + scratch:
                        reason = f"insufficient headroom on {device}: {free} bytes free, {reserve + scratch} required"
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

    def device_memory_requirements(self, reserve):
        runtime_reserve = self._runtime_scratch()
        if self.model._anima_shard:
            return {self.load_device: runtime_reserve, self.secondary_device: runtime_reserve}
        dit = self.model.diffusion_model
        sizes = [mm.module_size(block) for _, block in self._stages()]
        fixed = mm.module_size(self.model) - sum(sizes)
        primary_free = mm.get_free_memory(self.load_device)
        secondary_free = mm.get_free_memory(self.secondary_device)
        patch_reserve = max((low_vram_patch_estimate_vram(self.model, key) for key in self.patches), default=0)
        scratch = max(runtime_reserve, patch_reserve)
        split = contiguous_split(sizes, Budget(primary_free, int(reserve) + scratch), Budget(secondary_free, int(reserve) + scratch), first_fixed=fixed, policy=args.multi_gpu_split)
        self._planned_split = split
        return {self.load_device: fixed + sum(sizes[:split]) + scratch, self.secondary_device: sum(sizes[split:]) + scratch}

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
            for name, module in parents:
                device = self._device_for_name(name, split, stages)
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
                block.to(self.load_device if index < split else self.secondary_device)
            torch.cuda.synchronize(self.load_device)
            torch.cuda.synchronize(self.secondary_device)
            expected = {}
            for name, _ in list(self.model.named_parameters(remove_duplicate=False)) + list(self.model.named_buffers(remove_duplicate=False)):
                expected[name] = self._device_for_name(name, split, stages)
            residency = Residency(expected)
            residency.validate(self.model)
            dit._anima_shard_devices = (self.load_device, self.secondary_device, split)
            self.model._anima_shard = (self.load_device, self.secondary_device, split)
            self.model._anima_residency = residency
            self.model._multi_gpu_force_cast_weights = self.force_cast_weights
            self.model.model_loaded_weight_memory = self.model_size()
            self.model.model_lowvram = False
            self.model.current_weight_patches_uuid = self.patches_uuid
            self.current_device = self.load_device
            mm.logger.info("%s multi-GPU: stages 0-%d on %s, %d-%d on %s", type(dit).__name__, split - 1, self.load_device, split, len(stages) - 1, self.secondary_device)
            return self.model.model_loaded_weight_memory
        except BaseException:
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
            for module in self.model.modules():
                wipe_lowvram_weight(module)
            if hasattr(self.model.diffusion_model, "_anima_shard_devices"):
                del self.model.diffusion_model._anima_shard_devices
            self.model._anima_shard = None
            self.model._anima_residency = None
            self.model._multi_gpu_force_cast_weights = None
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
