import math
import sys

import torch

from backend import args, attention, memory_management
from backend.modules.k_prediction import k_prediction_from_diffusers_scheduler


def _release_shards_after_oom(shard):
    # MOSS keeps per-block reference keys and values outside the model. They
    # must be dropped before reclaiming VRAM; the next attempt can rebuild them.
    args.dynamic_args.moss_ref_kv = None
    for device in shard[:2]:
        try:
            memory_management.free_memory(float("inf"), device)
        except Exception:
            memory_management.logger.exception("Failed to release models on %s after OOM", device)
    try:
        memory_management.soft_empty_cache(force=True)
    except Exception:
        memory_management.logger.exception("Failed to clear the CUDA cache after OOM")


class KModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, diffusers_scheduler, config, k_predictor):
        super().__init__()

        self.config = config

        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        _store = f"storage: {self.storage_dtype}"
        _compute = f"computation: {self.computation_dtype}"

        if args.dynamic_args.ops.endswith("FP8"):
            _compute += f" + {torch.float8_e4m3fn}"
        if args.dynamic_args.ops.startswith("Mixed"):
            _compute = "computation: Mixed"

        memory_management.logger.info(f"Diffusion Model: {{{_store}, {_compute}}}")

        self.diffusion_model = model
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)

        if diffusers_scheduler is not None:
            self.predictor = k_prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = k_predictor

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        shard = getattr(self.diffusion_model, "_anima_shard_devices", None)
        if shard:
            from backend.multi_gpu.patches import device_aware_options, restore_patch_devices
            sources = transformer_options.get("_multi_gpu_patch_sources", ())
            original_options = transformer_options
            transformer_options = transformer_options.copy()
            transformer_options.pop("_multi_gpu_patch_sources", None)
            transformer_options = device_aware_options(transformer_options)
        try:
            sigma = t
            xc = self.predictor.calculate_input(sigma, x)
            if c_concat is not None:
                xc = torch.cat([xc] + [c_concat], dim=1)

            context = c_crossattn
            dtype = self.computation_dtype

            xc = xc.to(dtype)
            t = self.predictor.timestep(t).float()
            from backend.single_gpu_fastpaths import single_gpu_flag
            from backend.sampling.invariant_cache import cached_invariant, invariants_enabled, cache_dependencies_safe
            typed_cache = False
            if (single_gpu_flag("assembled_condition_cache") and type(self) is KModel
                    and c_concat is None and control is None and not shard
                    and kwargs.keys() <= {"y", "guidance"}
                    and type(context) is torch.Tensor
                    and invariants_enabled(self.diffusion_model, context, transformer_options)):
                from backend.nn.anima import Anima
                from backend.nn.flux import IntegratedFluxTransformer2DModel
                from backend.nn.lumina import NextDiT
                # Krea text fusion has in-place residuals; keep its casts separate.
                typed_cache = (type(self.diffusion_model) in (Anima, IntegratedFluxTransformer2DModel, NextDiT)
                               and cache_dependencies_safe(self.diffusion_model))

            def cast_condition(value, name):
                if typed_cache and type(value) is torch.Tensor and value.dtype != dtype:
                    return cached_invariant(self.diffusion_model, "typed_condition_" + name, value,
                                            transformer_options, lambda: value.to(dtype), geometry=(dtype,))
                return value.to(dtype)

            context = cast_condition(context, "context")
            extra_conds = {}
            for o in kwargs:
                extra = kwargs[o]
                if hasattr(extra, "dtype"):
                    if extra.dtype != torch.int and extra.dtype != torch.long:
                        extra = cast_condition(extra, o)
                extra_conds[o] = extra

            graph_output = None
            if getattr(args.args, "resident_denoiser_graph_mib", 0) > 0:
                from backend.resident_denoiser_graph import discard_resident_graph, try_resident_graph
                if c_concat is None and control is None and not shard:
                    graph_output = try_resident_graph(self.diffusion_model, xc, t, context, transformer_options, extra_conds)
                else:
                    discard_resident_graph()
            if graph_output is None:
                graph_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
            model_output = graph_output.float()
            return self.predictor.calculate_denoised(sigma, model_output, x)
        finally:
            if shard:
                failure = sys.exc_info()[1]
                out_of_memory = failure is not None and memory_management.is_oom(failure)
                cleanup_device = torch.device("cpu") if out_of_memory else shard[0]
                cleanup_oom = False
                try:
                    restore_patch_devices(transformer_options, cleanup_device, (original_options, *sources))
                except Exception as cleanup_error:
                    cleanup_oom = memory_management.is_oom(cleanup_error)
                    if failure is None:
                        if cleanup_oom:
                            _release_shards_after_oom(shard)
                        raise
                    memory_management.logger.exception("Failed to restore multi-GPU extension callbacks after an error")
                if out_of_memory or cleanup_oom:
                    # A partially completed sampler cannot safely be retried,
                    # but its resident models should not block the next job.
                    _release_shards_after_oom(shard)

    def memory_required(self, input_shape: list[int]) -> float:
        """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/model_base.py#L354"""
        input_shapes = [input_shape]
        area = sum(map(lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes))

        if attention.attention_function is attention.attention_basic:
            size = 8
            multiplier = 0.02
        elif args.dynamic_args.ops.startswith("Mixed"):
            size = 1
            multiplier = 0.02
        elif memory_management.xformers_enabled() or memory_management.pytorch_attention_flash_attention():
            size = memory_management.dtype_size(self.computation_dtype)
            multiplier = 0.01
        else:
            size = 0.15
            multiplier = 1.0

        return (area * size * multiplier * self.config.memory_usage_factor) * (1024 * 1024)
