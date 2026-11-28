# Referenced some codes from ComfyUI, with some parts rewritten
# to support different special control methods in Forge
# Copyright Forge 2024

import collections
import contextlib
import math
import weakref
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.patcher.unet import UnetPatcher
    from modules.sd_samplers_cfg_denoiser import CFGDenoiser, CFGDenoiserParams

import torch

from backend import memory_management, utils
from backend.args import args, dynamic_args
from backend.sampling.condition import (
    Condition,
    ConditionCrossAttn,
    compile_conditions,
    compile_weighted_conditions,
    repeat_to_batch_size,
)
from backend.sampling.krea_cache import (
    clear_current_krea_cache,
    krea_text_fusion_condition,
    krea_text_fusion_revision,
)
from backend.performance import record_region
from backend.sampling.invariant_cache import clear_invariants, invariant_condition, invariant_revision
from modules import shared


_parallel_cfg_cache = None


def _discard_parallel_cfg_cache():
    global _parallel_cfg_cache
    cached = _parallel_cfg_cache
    if cached is None:
        return True
    replica = cached[1]
    for loaded in list(memory_management.current_loaded_models):
        if loaded.model is replica:
            try:
                loaded.model_unload()
            except Exception:
                memory_management.logger.exception("Failed to release the cached parallel CFG replica")
                return False
            memory_management.current_loaded_models.remove(loaded)
    _parallel_cfg_cache = None
    return True


def _clear_parallel_cfg_cache(reference):
    if _parallel_cfg_cache is not None and _parallel_cfg_cache[0] is reference:
        _discard_parallel_cfg_cache()


def _invalidate_parallel_cfg_cache(model):
    if _parallel_cfg_cache is not None and _parallel_cfg_cache[0]() is model:
        _discard_parallel_cfg_cache()


def _parallel_cfg_patcher_eligible(unet):
    return not (unet.object_patches or unet.controlnet_linked_list is not None
                or unet.extra_model_patchers_during_sampling or unet.extra_concat_condition is not None)


def _parallel_batch_callbacks_safe():
    from modules import script_callbacks

    for entry in script_callbacks.callback_map["callbacks_cfg_denoiser"]:
        predicate = getattr(getattr(entry, "callback", None), "_forge_parallel_batch_safe", None)
        if not callable(predicate):
            return False
        try:
            if not predicate():
                return False
        except Exception:
            memory_management.logger.exception("Parallel batch callback safety check failed")
            return False
    return True


def _parallel_batch_model_supported(unet):
    from backend.nn.anima import Anima
    from backend.nn.chroma import IntegratedChromaTransformer2DModel
    from backend.nn.flux import IntegratedFluxTransformer2DModel
    from backend.nn.krea import SingleStreamDiT
    from backend.nn.moss_image import MOSSImage
    from backend.nn.qwen import QwenImageTransformer2DModel
    from backend.nn.qwen21 import QwenImage21Transformer2DModel
    from backend.nn.unet import IntegratedUNet2DConditionModel

    supported = (Anima, SingleStreamDiT, IntegratedFluxTransformer2DModel, IntegratedChromaTransformer2DModel,
                 IntegratedUNet2DConditionModel, MOSSImage, QwenImageTransformer2DModel, QwenImage21Transformer2DModel)
    return type(getattr(unet.model, "diffusion_model", None)) in supported


def condition_cache_allowed(model_options):
    return not any(
        model_options.get(option)
        for option in ("conditioning_modifiers", "sampler_pre_cfg_function", "model_function_wrapper")
    )


def get_area_and_mult(conds, x_in, timestep_in, condition_cache=None, direct_full_area=False):
    batch_size = x_in.shape[0]
    batch_indices = conds.get("batch_indices")
    if batch_indices is not None:
        x_in = x_in[batch_indices]

    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    if "timestep_start" in conds:
        timestep_start = conds["timestep_start"]
        if timestep_in[0] > timestep_start:
            return None
    if "timestep_end" in conds:
        timestep_end = conds["timestep_end"]
        if timestep_in[0] < timestep_end:
            return None
    if "area" in conds:
        area = conds["area"]
    if "strength" in conds:
        strength = conds["strength"]

    input_x = x_in[:, :, area[2] : area[0] + area[2], area[3] : area[1] + area[3]]

    if direct_full_area:
        mult = None
    elif "mask" in conds:
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds["mask"]
        if batch_indices is not None and mask.shape[0] == batch_size:
            mask = mask[batch_indices]
        assert mask.shape[1] == x_in.shape[2]
        assert mask.shape[2] == x_in.shape[3]
        mask = mask[:, area[2] : area[0] + area[2], area[3] : area[1] + area[3]]
        if mask.shape[0] == 1 and input_x.shape[0] > 1:
            mask = mask.expand(input_x.shape[0], -1, -1)
        elif mask.shape[0] != input_x.shape[0]:
            mask = repeat_to_batch_size(mask, input_x.shape[0])
        if not isinstance(mask_strength, (int, float)) or mask_strength != 1.0:
            mask = mask * mask_strength
        if not isinstance(strength, (int, float)) or strength != 1.0:
            mask = mask * strength
        # Every channel uses the same mask. Keep it as a read-only broadcast
        # instead of materializing a second copy for each latent channel.
        mult = mask.unsqueeze(1).expand(-1, input_x.shape[1], -1, -1)
    else:
        mult = torch.full_like(input_x, strength)

    if not direct_full_area and "mask" not in conds:
        rr = 8
        if area[2] != 0:
            for t in range(rr):
                mult[:, :, t : 1 + t, :] *= (1.0 / rr) * (t + 1)
        if (area[0] + area[2]) < x_in.shape[2]:
            for t in range(rr):
                mult[:, :, area[0] - 1 - t : area[0] - t, :] *= (1.0 / rr) * (t + 1)
        if area[3] != 0:
            for t in range(rr):
                mult[:, :, :, t : 1 + t] *= (1.0 / rr) * (t + 1)
        if (area[1] + area[3]) < x_in.shape[3]:
            for t in range(rr):
                mult[:, :, :, area[1] - 1 - t : area[1] - t] *= (1.0 / rr) * (t + 1)

    conditioning = {}
    model_conds = conds["model_conds"]
    condition_cache_key = tuple((name, getattr(value, "cache_key", None)) for name, value in model_conds.items())
    if not model_conds or any(key is None for _, key in condition_cache_key):
        condition_cache_key = None
    elif condition_cache_key is not None:
        try:
            hash(condition_cache_key)
        except TypeError:
            condition_cache_key = None
    for c in model_conds:
        value = model_conds[c]
        if batch_indices is not None and isinstance(value.cond, torch.Tensor) and value.cond.ndim > 0 and value.cond.shape[0] == batch_size:
            value = value._copy_with(value.cond[batch_indices])
        conditioning[c] = value.process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area, cache=condition_cache)

    control = conds.get("control", None)

    patches = None
    cond_obj = collections.namedtuple(
        "cond_obj", ["input_x", "mult", "conditioning", "area", "control", "patches", "batch_indices", "condition_cache_key"]
    )
    return cond_obj(input_x, mult, conditioning, area, control, patches, batch_indices, condition_cache_key)


def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True


def can_concat_cond(c1, c2):
    if not shared.batch_cond_uncond:
        return False
    if c1.batch_indices != c2.batch_indices:
        return False
    if any(ref.shape[0] > 1 for ref in dynamic_args.ref_latents):
        return False
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)


def cond_cat(c_list):
    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out


def compute_cond_mark(cond_or_uncond, sigmas):
    cond_or_uncond_size = int(sigmas.shape[0])
    return sigmas.new_tensor(cond_or_uncond).repeat_interleave(cond_or_uncond_size)


def should_skip_unconditional(step, sigma, s_min_uncond, every_step=False):
    """Check NGMS only when enabled, avoiding a disabled CUDA scalar read."""
    if not (step % 2 or every_step) or not (s_min_uncond > 0):
        return False
    return 0 < sigma[0] < s_min_uncond


def compute_cond_indices(cond_or_uncond, sigmas):
    cond_indices, uncond_indices = _cached_cond_indices(tuple(cond_or_uncond), int(sigmas.shape[0]))
    # Return fresh lists because model extensions receive these through
    # transformer_options and may mutate them.
    return list(cond_indices), list(uncond_indices)


@lru_cache(maxsize=64)
def _cached_cond_indices(cond_or_uncond, sigma_batch_size):
    cond_indices = []
    uncond_indices = []
    for i, cx in enumerate(cond_or_uncond):
        target = cond_indices if cx == 0 else uncond_indices
        target.extend(range(i * sigma_batch_size, (i + 1) * sigma_batch_size))
    return tuple(cond_indices), tuple(uncond_indices)


def _native_full_area_cfg(model, x, model_options, condition_cache):
    if (not getattr(args, "direct_full_area_cfg", False) or condition_cache is None
            or x.ndim != 4 or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or set(model_options) - {"transformer_options"}
            or model_options.get("transformer_options")
            or dynamic_args.ref_latents or dynamic_args.context_handler is not None
            or any(getattr(args, key, None) is not None for key in
                   ("multi_gpu_anima", "parallel_cfg", "parallel_batch"))):
        return False
    from backend.modules.k_model import KModel
    from backend.nn.anima import Anima
    from backend.nn.flux import IntegratedFluxTransformer2DModel
    from backend.nn.lumina import NextDiT
    from modules import script_callbacks

    diffusion = getattr(model, "diffusion_model", None)
    return (type(model) is KModel and type(diffusion) in (Anima, IntegratedFluxTransformer2DModel, NextDiT)
            and not getattr(diffusion, "_anima_shard_devices", None)
            and not any(script_callbacks.callback_map.get(key) for key in
                        ("callbacks_cfg_denoiser", "callbacks_cfg_denoised", "callbacks_cfg_after_cfg")))


def _unit_full_area(item, x):
    strength = item.get("strength", 1.0)
    if (not isinstance(item.get("model_conds"), dict)
            or any(key in item for key in ("mask", "mask_strength", "batch_indices", "timestep_start",
                                           "timestep_end", "control", "patches"))
            or not isinstance(strength, (int, float)) or strength != 1.0):
        return False
    area = item.get("area")
    return area is None or tuple(area) == (x.shape[2], x.shape[3], 0, 0)


def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options, condition_cache=None, materialize_uncond=True):
    direct = (uncond is not None and len(cond) == len(uncond) == 1
              and _native_full_area_cfg(model, x_in, model_options, condition_cache)
              and _unit_full_area(cond[0], x_in) and _unit_full_area(uncond[0], x_in))
    out_cond = None if direct else torch.zeros_like(x_in)
    out_count = None if direct else torch.full_like(x_in, 1e-37)

    out_uncond = None if direct else (torch.zeros_like(x_in) if uncond is not None or materialize_uncond else None)
    out_uncond_count = torch.full_like(x_in, 1e-37) if uncond is not None and not direct else None

    COND = 0
    UNCOND = 1

    to_run = []
    area_kwargs = {"direct_full_area": True} if direct else {}
    for x in cond:
        p = get_area_and_mult(x, x_in, timestep, condition_cache, **area_kwargs)
        if p is None:
            continue

        to_run += [(p, COND)]
    if uncond is not None:
        for x in uncond:
            p = get_area_and_mult(x, x_in, timestep, condition_cache, **area_kwargs)
            if p is None:
                continue

            to_run += [(p, UNCOND)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = [0]
        for x in range(1, len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        if memory_management.signal_empty_cache:
            memory_management.soft_empty_cache(force=False)

        free_memory = memory_management.get_free_memory(x_in.device)

        if (not args.disable_gpu_warning) and x_in.device.type == "cuda":
            free_memory_mb = free_memory / (1024.0 * 1024.0)
            safe_memory_mb = 1536.0
            if free_memory_mb < safe_memory_mb:
                logger = memory_management.logger

                logger.warning("The current free memory for GPU is {:.2f} MB".format(free_memory_mb))
                logger.warning("This number is lower than the safe threshold ; This may cause extreme slow performance")
                logger.warning('You can add "--reserve-vram 2" to keep a larger headroom')
                logger.warning('You can also (not recommended) add "--disable-gpu-warning" to remove this warning')

        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[: len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        condition_parts = []
        condition_parts_valid = condition_cache is not None and condition_cache_allowed(model_options)
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches
            indices = p.batch_indices
            if indices is not None:
                if isinstance(indices, (list, tuple)) and all(isinstance(value, int) for value in indices):
                    indices = tuple(indices)
                else:
                    condition_parts_valid = False
            if p.condition_cache_key is None or p.control is not None or p.patches is not None:
                condition_parts_valid = False
            else:
                condition_parts.append((p.condition_cache_key, tuple(p.area), indices, o[1]))

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        batch_indices = first[0].batch_indices
        batch_timestep = timestep if batch_indices is None else repeat_to_batch_size(timestep, x_in.shape[0])[batch_indices]
        timestep_ = torch.cat([batch_timestep] * batch_chunks)

        transformer_options = {}
        if "transformer_options" in model_options:
            transformer_options = model_options["transformer_options"].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = batch_timestep

        transformer_options["cond_mark"] = compute_cond_mark(cond_or_uncond=cond_or_uncond, sigmas=batch_timestep)
        transformer_options["cond_indices"], transformer_options["uncond_indices"] = compute_cond_indices(cond_or_uncond=cond_or_uncond, sigmas=batch_timestep)

        c["transformer_options"] = transformer_options

        krea_condition_key = None
        if condition_parts_valid and condition_parts:
            candidate = (id(model), getattr(model, "patches_uuid", None), tuple(condition_parts))
            try:
                hash(candidate)
                krea_condition_key = candidate
            except TypeError:
                pass

        batch_context = contextlib.nullcontext() if batch_indices is None else conditioning_batch(control, batch_indices, x_in.shape[0])
        with batch_context:
            if control is not None:
                p = control
                while p is not None:
                    p.transformer_options = transformer_options
                    p = p.previous_controlnet
                control_cond = c.copy()  # get_control may change items in this dict, so we need to copy it
                c["control"] = control.get_control(input_x, timestep_, control_cond, len(cond_or_uncond))
                c["control_model"] = control

            if getattr(getattr(model, "diffusion_model", None), "_anima_shard_devices", None):
                sources = [model_options.get("transformer_options", {})]
                if patches is not None:
                    sources.append({"patches": patches})
                transformer_options["_multi_gpu_patch_sources"] = sources

            if "model_function_wrapper" in model_options:
                output = model_options["model_function_wrapper"](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
            else:
                with krea_text_fusion_condition(krea_condition_key), invariant_condition(krea_condition_key):
                    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
        del input_x

        if direct:
            for index, kind in enumerate(cond_or_uncond):
                prediction = output[index].to(dtype=x_in.dtype)
                if kind == COND:
                    out_cond = prediction
                else:
                    out_uncond = prediction
            continue

        rows = slice(None) if batch_indices is None else batch_indices
        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                out_cond[rows, :, area[o][2] : area[o][0] + area[o][2], area[o][3] : area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[rows, :, area[o][2] : area[o][0] + area[o][2], area[o][3] : area[o][1] + area[o][3]] += mult[o]
            else:
                out_uncond[rows, :, area[o][2] : area[o][0] + area[o][2], area[o][3] : area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[rows, :, area[o][2] : area[o][0] + area[o][2], area[o][3] : area[o][1] + area[o][3]] += mult[o]
        del mult

    if direct:
        return out_cond, out_uncond
    out_cond /= out_count
    del out_count
    if out_uncond_count is not None:
        out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond


def single_condition_fast_path_eligible(model, cond, x, model_options, condition_cache):
    if condition_cache is None or x.ndim != 4 or len(cond) != 1 or not condition_cache_allowed(model_options):
        return False
    if model_options.get("model_function_wrapper"):
        return False
    if getattr(getattr(model, "diffusion_model", None), "_anima_shard_devices", None):
        return False

    item = cond[0]
    if not isinstance(item.get("model_conds"), dict):
        return False
    if any(key in item for key in ("mask", "mask_strength", "batch_indices", "timestep_start", "timestep_end", "control", "patches")):
        return False

    area = item.get("area")
    if area is not None and tuple(area) != (x.shape[2], x.shape[3], 0, 0):
        return False
    strength = item.get("strength", 1.0)
    return isinstance(strength, (int, float)) and math.isclose(strength, 1.0)


def calc_single_condition(model, cond, x, timestep, model_options, condition_cache=None):
    """Run a single full-image condition without CFG accumulation buffers."""
    item = cond[0]
    area = (x.shape[2], x.shape[3], 0, 0)
    conditioning = {
        name: value.process_cond(batch_size=x.shape[0], device=x.device, area=area, cache=condition_cache)
        for name, value in item["model_conds"].items()
    }
    if _native_full_area_cfg(model, x, model_options, condition_cache):
        # The audited native KModel forwards these fields read-only; keep
        # custom condition subclasses and other inputs on their concat path.
        c = {name: value.cond if name in ("c_crossattn", "y", "guidance")
             and type(value) in (Condition, ConditionCrossAttn) else value.concat([])
             for name, value in conditioning.items()}
    else:
        c = cond_cat([conditioning])
    transformer_options = model_options.get("transformer_options", {}).copy()
    transformer_options["cond_or_uncond"] = [0]
    transformer_options["sigmas"] = timestep
    transformer_options["cond_mark"] = torch.zeros_like(timestep)
    transformer_options["cond_indices"] = list(range(timestep.shape[0]))
    transformer_options["uncond_indices"] = []
    c["transformer_options"] = transformer_options
    keys = tuple((name, getattr(value, "cache_key", None)) for name, value in item["model_conds"].items())
    invariant_key = None
    if (condition_cache is not None and condition_cache_allowed(model_options)
            and keys and all(key is not None for _, key in keys)):
        candidate = (id(model), getattr(model, "patches_uuid", None), ((keys, area, None, 0),))
        try:
            hash(candidate)
            invariant_key = candidate
        except TypeError:
            pass
    krea_key = invariant_key if getattr(args, "invariant_cache_mib", 0) > 0 else None
    with record_region("forge::single_condition_cfg1"), invariant_condition(invariant_key), krea_text_fusion_condition(krea_key):
        return model.apply_model(x, timestep, **c)


@contextlib.contextmanager
def conditioning_batch(control, indices, batch_size):
    saved = []
    out_of_memory = False

    def select(obj, key):
        value = getattr(obj, key)
        saved.append((obj, key, value))
        setattr(obj, key, repeat_to_batch_size(value, batch_size)[indices])

    try:
        refs = dynamic_args.ref_latents
        if any(ref.shape[0] > 1 for ref in refs):
            saved.append((dynamic_args, "ref_latents", refs))
            dynamic_args.ref_latents = [repeat_to_batch_size(ref, batch_size)[indices] for ref in refs]
            saved.append((dynamic_args, "moss_ref_kv", dynamic_args.get("moss_ref_kv")))
            dynamic_args.moss_ref_kv = None

        while control is not None:
            if isinstance(control.cond_hint_original, torch.Tensor) and control.cond_hint_original.shape[0] > 1:
                select(control, "cond_hint_original")
                for key in ("cond_hint", "control_input"):
                    if hasattr(control, key):
                        saved.append((control, key, getattr(control, key)))
                        setattr(control, key, None)
            mask = getattr(control, "advanced_mask_weighting", None)
            if isinstance(mask, torch.Tensor) and mask.shape[0] > 1:
                select(control, "advanced_mask_weighting")
            frames = getattr(control, "advanced_frame_weighting", None)
            if frames is not None and len(frames) == batch_size:
                saved.append((control, "advanced_frame_weighting", frames))
                control.advanced_frame_weighting = [frames[i] for i in indices]
            control = control.previous_controlnet

        yield
    except Exception as error:
        out_of_memory = memory_management.is_oom(error)
        raise
    finally:
        while saved:
            obj, key, value = saved.pop()
            # A batch-specific forward may have discarded the cache while
            # recovering from OOM. Do not revive the larger previous cache.
            setattr(obj, key, None if out_of_memory and obj is dynamic_args and key == "moss_ref_kv" else value)
            value = None


def sampling_function_inner(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None, return_full=False, condition_cache=None, parallel_cfg=None):
    if any("batch_indices" in item for item in cond):
        edit_strength = x.new_zeros(x.shape[0])
        for item in cond:
            edit_strength[item.get("batch_indices", slice(None))] += item.get("strength", 1)
        edit_strength = edit_strength.reshape((-1,) + (1,) * (x.ndim - 1))
    else:
        edit_strength = sum((item["strength"] if "strength" in item else 1) for item in cond)

    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    for fn in model_options.get("sampler_pre_cfg_function", []):
        model, cond, uncond_, x, timestep, model_options = fn(model, cond, uncond_, x, timestep, model_options)

    use_context = getattr(dynamic_args.context_handler, "should_use_context", lambda *args: False)(x)
    direct_cond = (
        not return_full and cond_scale == 1.0 and not isinstance(edit_strength, torch.Tensor) and math.isclose(edit_strength, 1.0) and uncond_ is None
        and not use_context
        and not any(model_options.get(key) for key in (
            "disable_cfg1_optimization", "sampler_pre_cfg_function", "sampler_cfg_function", "sampler_post_cfg_function",
        ))
    )
    if use_context:
        cond_pred, uncond_pred = dynamic_args.context_handler.execute(calc_cond_uncond_batch, model, [cond, uncond_], x, timestep, model_options)
    elif parallel_cfg is not None and parallel_cfg.primary is model and parallel_cfg.eligible(cond, uncond_, model_options, x):
        from backend.multi_gpu.parallel import ParallelBatch
        if direct_cond and type(parallel_cfg) is ParallelBatch:
            cond_pred = parallel_cfg.predict_cond(cond, uncond_, x, timestep, model_options)
            uncond_pred = None
        else:
            cond_pred, uncond_pred = parallel_cfg.predict(cond, uncond_, x, timestep, model_options)
    elif direct_cond and single_condition_fast_path_eligible(model, cond, x, model_options, condition_cache):
        cond_pred = calc_single_condition(model, cond, x, timestep, model_options, condition_cache)
        uncond_pred = None
    else:
        cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options, condition_cache, materialize_uncond=return_full or not direct_cond)

    if direct_cond:
        cfg_result = cond_pred
    elif "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep, "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    elif isinstance(edit_strength, torch.Tensor) or not math.isclose(edit_strength, 1.0):
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "cond_scale": cond_scale, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred, "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    if return_full:
        return cfg_result, cond_pred, uncond_pred

    return cfg_result


def sampling_function(self, denoiser_params, cond_scale, cond_composition, extra_model_options=None, condition_cache=None, cond_cache_key=None, uncond_cache_key=None, return_full=True):
    unet_patcher = self.inner_model.inner_model.forge_objects.unet
    model = unet_patcher.model
    control = unet_patcher.controlnet_linked_list
    extra_concat_condition = unet_patcher.extra_concat_condition
    x = denoiser_params.x
    timestep = denoiser_params.sigma
    uncond = compile_conditions(denoiser_params.text_uncond, cache_namespace=uncond_cache_key)
    cond = compile_weighted_conditions(denoiser_params.text_cond, cond_composition, cache_namespace=cond_cache_key)
    model_options = utils.join_dicts(unet_patcher.model_options, extra_model_options)
    if (not condition_cache_allowed(model_options) or unet_patcher.object_patches
            or unet_patcher.extra_model_patchers_during_sampling):
        condition_cache = None
    seed = self.p.seeds[0]

    if extra_concat_condition is not None:
        image_cond_in = extra_concat_condition
    else:
        image_cond_in = denoiser_params.image_cond

    if isinstance(image_cond_in, torch.Tensor) and self.inner_model.inner_model.is_inpaint:
        if image_cond_in.shape[0] == x.shape[0] and image_cond_in.shape[2] == x.shape[2] and image_cond_in.shape[3] == x.shape[3]:
            if uncond is not None:
                for i in range(len(uncond)):
                    uncond[i]["model_conds"]["c_concat"] = Condition(image_cond_in, cache_key=("image", id(image_cond_in)))
            for i in range(len(cond)):
                cond[i]["model_conds"]["c_concat"] = Condition(image_cond_in, cache_key=("image", id(image_cond_in)))

    if control is not None:
        for h in cond:
            h["control"] = control
        if uncond is not None:
            for h in uncond:
                h["control"] = control

    for modifier in model_options.get("conditioning_modifiers", []):
        model, x, timestep, uncond, cond, cond_scale, model_options, seed = modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

    parallel_cfg = getattr(unet_patcher, "_experimental_parallel_cfg", None)
    from backend.multi_gpu.parallel import ParallelBatch

    batch_mode = bool(getattr(unet_patcher, "_experimental_cfg_batch_mode", False)) and cond_scale == 1.0 and _parallel_batch_callbacks_safe()
    if parallel_cfg is not None:
        replica = getattr(unet_patcher, "_experimental_cfg_replica", None)
        if (replica is None or replica.patches_uuid != unet_patcher.patches_uuid
                or parallel_cfg.primary is not model or isinstance(parallel_cfg, ParallelBatch) != batch_mode
                or (not batch_mode and math.isclose(cond_scale, 1.0))
                or not _parallel_cfg_patcher_eligible(unet_patcher)
                or not parallel_cfg.eligible(cond, uncond, model_options, x)):
            _release_parallel_cfg(unet_patcher)
            parallel_cfg = None
    if parallel_cfg is None and (batch_mode or (args.parallel_cfg is not None and not math.isclose(cond_scale, 1.0))):
        parallel_cfg = _load_parallel_cfg(unet_patcher, cond, uncond, model_options, x, batch_mode=batch_mode)
    try:
        revision = (id(unet_patcher), unet_patcher.patches_uuid)
        invariant_revision_key = None
        if (getattr(args, "invariant_cache_mib", 0) > 0 and condition_cache is not None
                and cond_cache_key is not None and (uncond is None or math.isclose(cond_scale, 1.0) or uncond_cache_key is not None)
                and parallel_cfg is None and control is None and extra_concat_condition is None
                and not unet_patcher.object_patches and not unet_patcher.extra_model_patchers_during_sampling
                and not getattr(unet_patcher, "weight_wrapper_patches", None)
                and set(model_options) <= {"transformer_options"}
                and not getattr(getattr(model, "diffusion_model", None), "_anima_shard_devices", None)
                and not dynamic_args.ref_latents and dynamic_args.context_handler is None):
            from modules import script_callbacks
            if not any(script_callbacks.callback_map.get(key) for key in ("callbacks_cfg_denoiser", "callbacks_cfg_denoised", "callbacks_cfg_after_cfg")):
                invariant_revision_key = revision
        with krea_text_fusion_revision(revision), invariant_revision(invariant_revision_key):
            return sampling_function_inner(model, x, timestep, uncond, cond, cond_scale, model_options, seed, return_full=return_full, condition_cache=condition_cache, parallel_cfg=parallel_cfg)
    except BaseException as error:
        clear_current_krea_cache()
        clear_invariants()
        _release_parallel_cfg(unet_patcher)
        if parallel_cfg is not None and isinstance(error, Exception) and memory_management.is_oom(error):
            try:
                memory_management.free_memory(float("inf"), unet_patcher.load_device)
            except Exception:
                memory_management.logger.exception("Failed to release the primary model after parallel CFG OOM")
        raise


def _release_parallel_cfg(unet, preserve_cpu=False):
    global _parallel_cfg_cache
    adapter = getattr(unet, "_experimental_parallel_cfg", None)
    if adapter is not None:
        adapter.close()
        unet._experimental_parallel_cfg = None
    replica = getattr(unet, "_experimental_cfg_replica", None)
    if replica is None:
        return
    if (preserve_cpu and adapter is not None and args.parallel_cfg_keep_resident
            and replica.patches_uuid == unet.patches_uuid
            and any(loaded.model is replica for loaded in memory_management.current_loaded_models)
            and _discard_parallel_cfg_cache()):
        unet._experimental_cfg_replica = None
        unet._experimental_cfg_needs_rollback = False
        _parallel_cfg_cache = (weakref.ref(unet.model, _clear_parallel_cfg_cache), replica)
        return
    tracked = False
    for loaded in list(memory_management.current_loaded_models):
        if loaded.model is replica:
            tracked = True
            try:
                loaded.model_unload()
            except Exception:
                memory_management.logger.exception("Failed to unload the parallel CFG replica")
                return
            memory_management.current_loaded_models.remove(loaded)
    if not tracked and (getattr(unet, "_experimental_cfg_loading", False) or getattr(unet, "_experimental_cfg_needs_rollback", False)):
        try:
            replica.detach()
        except Exception:
            unet._experimental_cfg_needs_rollback = True
            memory_management.logger.exception("Failed to roll back the parallel CFG replica")
            return
    unet._experimental_cfg_needs_rollback = False
    unet._experimental_cfg_replica = None
    if preserve_cpu and adapter is not None and replica.patches_uuid == unet.patches_uuid and _discard_parallel_cfg_cache():
        _parallel_cfg_cache = (weakref.ref(unet.model, _clear_parallel_cfg_cache), replica)


def _auto_sharded_fallback(unet):
    """Create or reuse a sharded patcher for an automatic replica placement."""
    from backend.patcher.unet import UnetPatcher
    from backend.multi_gpu.patcher import ShardedModelPatcher

    model = getattr(unet, "model", None)
    diffusion = getattr(model, "diffusion_model", None)
    if (type(unet) is not UnetPatcher or getattr(diffusion, "_forge_auto_multi_gpu_mode", None) != "replicas"
            or args.multi_gpu_anima is None or not _parallel_cfg_patcher_eligible(unet)
            or getattr(unet, "_experimental_cfg_needs_rollback", False)):
        return None
    previous = getattr(model, "_forge_auto_sharded_fallback", None)
    wrappers_match = (previous is not None and previous.weight_wrapper_patches.keys() == unet.weight_wrapper_patches.keys()
                      and all(len(previous.weight_wrapper_patches[key]) == len(values)
                              and all(old is new for old, new in zip(previous.weight_wrapper_patches[key], values))
                              for key, values in unet.weight_wrapper_patches.items()))
    if (previous is not None and previous.patches_uuid == unet.patches_uuid
            and previous.force_cast_weights == unet.force_cast_weights and wrappers_match):
        previous.model_options = utils.deepcopy_(unet.model_options)
        previous.extra_preserved_memory_during_sampling = unet.extra_preserved_memory_during_sampling
        return previous
    # A source may be tracked by a different clone of this patcher. Restore its
    # original weights before handing ownership to the sharded patcher.
    for loaded in list(memory_management.current_loaded_models):
        owner = loaded.model
        if owner is not None and owner.model is model:
            if not loaded.model_unload():
                return None
            memory_management.current_loaded_models.remove(loaded)
    try:
        fallback = ShardedModelPatcher(model, unet.load_device, unet.offload_device, unet.model_size(),
                                       current_device=unet.current_device, weight_inplace_update=unet.weight_inplace_update)
    except (ValueError, RuntimeError):
        memory_management.logger.exception("Could not prepare the automatic sharded fallback")
        return None
    fallback.patches = {key: value[:] for key, value in unet.patches.items()}
    fallback.weight_wrapper_patches = {key: value[:] for key, value in unet.weight_wrapper_patches.items()}
    fallback.patches_uuid = unet.patches_uuid
    fallback.model_options = utils.deepcopy_(unet.model_options)
    fallback.force_cast_weights = unet.force_cast_weights
    fallback.controlnet_linked_list = unet.controlnet_linked_list
    fallback.extra_preserved_memory_during_sampling = unet.extra_preserved_memory_during_sampling
    fallback.extra_model_patchers_during_sampling = unet.extra_model_patchers_during_sampling.copy()
    fallback.extra_concat_condition = unet.extra_concat_condition
    model._forge_auto_sharded_fallback = fallback
    memory_management.logger.info("Using automatic model sharding after replica setup failed")
    return fallback


def _load_auto_sharded_fallback(unet, fallback, memory_required, minimum_memory_required=None):
    """Load shards or restore the ordinary patcher if sharding cannot fit."""
    try:
        memory_management.load_models_gpu([fallback], memory_required=memory_required,
                                          minimum_memory_required=minimum_memory_required)
        if not getattr(unet.model, "_anima_shard", None):
            raise RuntimeError("The sharded model was not loaded")
        return True
    except Exception:
        memory_management.logger.exception("Automatic sharding failed; restoring the normal model")
        tracked = False
        for loaded in list(memory_management.current_loaded_models):
            if loaded.model is fallback:
                tracked = True
                if not loaded.model_unload():
                    raise RuntimeError("Could not unload the failed sharded fallback")
                memory_management.current_loaded_models.remove(loaded)
        if not tracked:
            fallback.detach()
        unet.model._forge_auto_sharded_fallback = None
        memory_management.load_models_gpu([unet], memory_required=memory_required,
                                          minimum_memory_required=minimum_memory_required)
        return False


def _auto_single_device_fallback(unet):
    """Use the normal offloading patcher when automatic sharding cannot fit."""
    from backend.multi_gpu.patcher import ShardedModelPatcher
    from backend.patcher.unet import UnetPatcher

    model = getattr(unet, "model", None)
    diffusion = getattr(model, "diffusion_model", None)
    if not isinstance(unet, ShardedModelPatcher) or getattr(diffusion, "_forge_auto_multi_gpu_mode", None) != "sharding":
        return None
    previous = getattr(model, "_forge_auto_single_fallback", None)
    wrappers_match = (previous is not None and previous.weight_wrapper_patches.keys() == unet.weight_wrapper_patches.keys()
                      and all(len(previous.weight_wrapper_patches[key]) == len(values)
                              and all(old is new for old, new in zip(previous.weight_wrapper_patches[key], values))
                              for key, values in unet.weight_wrapper_patches.items()))
    objects_match = (previous is not None and previous.object_patches.keys() == unet.object_patches.keys()
                     and all(previous.object_patches[key] is value for key, value in unet.object_patches.items()))
    if (previous is not None and previous.patches_uuid == unet.patches_uuid
            and previous.force_cast_weights == unet.force_cast_weights and wrappers_match
            and objects_match):
        previous.model_options = utils.deepcopy_(unet.model_options)
        return previous

    tracked = False
    for loaded in list(memory_management.current_loaded_models):
        owner = loaded.model
        if owner is not None and owner.model is model:
            tracked = True
            if not loaded.model_unload():
                raise RuntimeError("Could not unload the model before single-device fallback")
            memory_management.current_loaded_models.remove(loaded)
    if not tracked and (getattr(model, "_anima_shard", None) or unet.loaded_size()):
        unet.detach()

    fallback = UnetPatcher(model, unet.load_device, unet.offload_device, unet.model_size(),
                           current_device=unet.offload_device, weight_inplace_update=unet.weight_inplace_update)
    fallback.patches = {key: value[:] for key, value in unet.patches.items()}
    fallback.weight_wrapper_patches = {key: value[:] for key, value in unet.weight_wrapper_patches.items()}
    fallback.patches_uuid = unet.patches_uuid
    fallback.model_options = utils.deepcopy_(unet.model_options)
    fallback.object_patches = unet.object_patches.copy()
    fallback.force_cast_weights = unet.force_cast_weights
    fallback.controlnet_linked_list = unet.controlnet_linked_list
    fallback.extra_preserved_memory_during_sampling = unet.extra_preserved_memory_during_sampling
    fallback.extra_model_patchers_during_sampling = unet.extra_model_patchers_during_sampling.copy()
    fallback.extra_concat_condition = unet.extra_concat_condition
    model._forge_auto_single_fallback = fallback
    return fallback


def _fallback_after_replica_load_failure(unet, x):
    _release_parallel_cfg(unet)
    if getattr(unet, "_experimental_cfg_replica", None) is not None:
        return
    fallback = _auto_sharded_fallback(unet)
    if fallback is not None:
        required = unet.memory_required(list(x.shape)) + unet.extra_preserved_memory_during_sampling
        _load_auto_sharded_fallback(unet, fallback, required)


def _load_parallel_cfg(unet, cond, uncond, model_options, x, batch_mode=False):
    from backend.multi_gpu.parallel import ParallelBatch, ParallelCFG

    replica = getattr(unet, "_experimental_cfg_replica", None)
    adapter_type = ParallelBatch if batch_mode else ParallelCFG
    if replica is None or getattr(unet, "_experimental_cfg_needs_rollback", False):
        return None
    if not adapter_type.eligible(cond, uncond, model_options, x):
        if batch_mode and not getattr(unet, "_experimental_batch_fallback_reported", False):
            memory_management.logger.info("Parallel batch unavailable for this conditioning; using the normal sampler")
            unet._experimental_batch_fallback_reported = True
        return None
    if (replica.patches_uuid != unet.patches_uuid or not _parallel_cfg_patcher_eligible(unet)
            or unet.model.model_lowvram or unet.loaded_size() < unet.model_size()):
        _fallback_after_replica_load_failure(unet, x)
        return None
    try:
        resident = replica.loaded_size() if getattr(replica, "current_device", None) == replica.load_device else 0
        required = max(0, replica.model_size() - resident) + memory_management.minimum_inference_memory() + unet.memory_required(list(x.shape))
        try:
            available = memory_management.get_free_memory(replica.load_device)
        except Exception:
            memory_management.logger.exception("Could not check replica memory")
            _fallback_after_replica_load_failure(unet, x)
            return None
        if available < required:
            _fallback_after_replica_load_failure(unet, x)
            return None
        unet._experimental_cfg_loading = True
        memory_management.load_models_gpu([unet, replica], memory_required=unet.memory_required(list(x.shape)), force_full_load=True)
        unet._experimental_cfg_loading = False
        if replica.model.model_lowvram or replica.loaded_size() < replica.model_size():
            _fallback_after_replica_load_failure(unet, x)
            return None
        if batch_mode:
            adapter = adapter_type(unet.model, replica.model, (unet.load_device, replica.load_device), adaptive=args.parallel_batch_adaptive)
        else:
            adapter = adapter_type(unet.model, replica.model, (unet.load_device, replica.load_device))
        unet._experimental_parallel_cfg = adapter
        memory_management.logger.info("Parallel %s replicas ready on %s and %s", "batch" if batch_mode else "CFG", unet.load_device, replica.load_device)
        return adapter
    except BaseException as error:
        if isinstance(error, Exception) and memory_management.is_oom(error):
            memory_management.logger.warning("Parallel CFG replica could not fit; trying automatic sharding")
            _fallback_after_replica_load_failure(unet, x)
            return None
        _release_parallel_cfg(unet)
        raise
    finally:
        unet._experimental_cfg_loading = False


def sampling_prepare(unet: "UnetPatcher", x: torch.Tensor, cfg_scale=None, cfg1_single_condition=False):
    global _parallel_cfg_cache
    _release_parallel_cfg(unet)
    batch_mode = (cfg1_single_condition and cfg_scale == 1.0 and x.shape[0] > 1 and args.parallel_batch is not None
                  and (args.parallel_cfg is None or args.parallel_cfg == args.parallel_batch)
                  and _parallel_batch_model_supported(unet)
                  and _parallel_batch_callbacks_safe())
    unet._experimental_cfg_batch_mode = batch_mode
    unet._experimental_batch_fallback_reported = False
    fallback = None
    if getattr(unet.model, "_forge_auto_sharded_fallback", None) is not None:
        fallback = _auto_sharded_fallback(unet)
    single_fallback = None
    if getattr(unet.model, "_forge_auto_single_fallback", None) is not None:
        single_fallback = _auto_single_device_fallback(unet)
    if args.parallel_batch is not None and cfg_scale == 1.0 and x.shape[0] > 1 and not batch_mode:
        memory_management.logger.info("Parallel batch unavailable for this model or conditioning; using the normal sampler")
    secondary_index = args.parallel_batch if batch_mode else args.parallel_cfg
    if fallback is not None:
        secondary_index = None
    if getattr(getattr(unet.model, "diffusion_model", None), "_forge_auto_multi_gpu_mode", None) in ("sharding", "single"):
        secondary_index = None
    if getattr(unet, "_experimental_cfg_replica", None) is not None:
        memory_management.logger.warning("Parallel CFG is disabled until the previous replica can be released")
        secondary_index = None
    if secondary_index is None or (cfg_scale is not None and math.isclose(cfg_scale, 1.0) and not batch_mode):
        cached = _parallel_cfg_cache
        retained_index = args.parallel_batch if cfg_scale is not None and math.isclose(cfg_scale, 1.0) and args.parallel_batch is not None else secondary_index
        keep_cached = (args.parallel_cfg_keep_resident and retained_index is not None and retained_index >= 0 and cached
                       and cached[0]() is unet.model and cached[1].patches_uuid == unet.patches_uuid
                       and cached[1].load_device == torch.device("cuda", retained_index)
                       and _parallel_cfg_patcher_eligible(unet))
        if not keep_cached:
            _discard_parallel_cfg_cache()
    if secondary_index is not None and (batch_mode or cfg_scale is None or not math.isclose(cfg_scale, 1.0)):
        from backend.multi_gpu.replicas import UnsupportedReplica, clone_unet_patcher
        try:
            if secondary_index < 0:
                raise UnsupportedReplica("A non-negative CUDA device index is required")
            secondary = torch.device("cuda", secondary_index)
            if unet.load_device.type != "cuda" or secondary == unet.load_device or secondary.index >= torch.cuda.device_count():
                raise UnsupportedReplica("A distinct available secondary CUDA device is required")
            if not _parallel_cfg_patcher_eligible(unet):
                raise UnsupportedReplica("Control networks and object patches are unsupported")
            cached = _parallel_cfg_cache
            reusable = cached and cached[0]() is unet.model and cached[1].patches_uuid == unet.patches_uuid and cached[1].load_device == secondary
            if cached and not reusable:
                if not _discard_parallel_cfg_cache():
                    raise UnsupportedReplica("The previous replica could not be released")
                cached = None
            resident = cached[1].loaded_size() if reusable and cached[1].current_device == secondary else 0
            try:
                secondary_free = memory_management.get_free_memory(secondary)
            except Exception as error:
                _discard_parallel_cfg_cache()
                raise UnsupportedReplica("Unable to check secondary-device memory") from error
            if secondary_free < max(0, unet.model_size() - resident) + memory_management.minimum_inference_memory():
                _discard_parallel_cfg_cache()
                raise UnsupportedReplica("Not enough free memory for a secondary replica")
            if reusable:
                _parallel_cfg_cache = None
                unet._experimental_cfg_replica = cached[1]
                memory_management.logger.info("Reusing the cached parallel CFG replica")
            else:
                # A previous non-parallel request can leave this model resident,
                # possibly through another patcher sharing the same model. Undo
                # its patches before making the independent CPU replica.
                for loaded in list(memory_management.current_loaded_models):
                    owner = loaded.model
                    if owner is not None and owner.model is unet.model:
                        if not loaded.model_unload():
                            raise UnsupportedReplica("The source model could not be fully offloaded")
                        memory_management.current_loaded_models.remove(loaded)
                unet._experimental_cfg_replica = clone_unet_patcher(unet, secondary)
        except UnsupportedReplica as error:
            memory_management.logger.info("Parallel %s unavailable: %s", "batch" if batch_mode else "CFG", error)
            fallback = _auto_sharded_fallback(unet)
    try:
        shape = list(x.shape)
        single_condition = cfg1_single_condition and cfg_scale == 1.0
        mem_shape = [(2 if shared.batch_cond_uncond and not single_condition else 1) * shape[0]] + shape[1:]

        unet_inference_memory = unet.memory_required(mem_shape)
        memory_management.logger.debug("Sampling memory plan: CFG=%s, latent batch=%s, estimated batch=%s, inference bytes=%s", cfg_scale, shape[0], mem_shape[0], unet_inference_memory)
        if single_condition and shared.batch_cond_uncond:
            memory_management.logger.info("CFG 1 memory plan uses batch %s (single conditioning branch)", shape[0])
        additional_inference_memory = unet.extra_preserved_memory_during_sampling
        additional_model_patchers = unet.extra_model_patchers_during_sampling

        if unet.controlnet_linked_list is not None:
            additional_inference_memory += unet.controlnet_linked_list.inference_memory_requirements(unet.model_dtype())
            additional_model_patchers += unet.controlnet_linked_list.get_models()

        if unet.has_online_lora():
            lora_memory = utils.nested_compute_size(unet.weight_wrapper_patches, element_size=utils.dtype_to_element_size(unet.model.computation_dtype))
            additional_inference_memory += lora_memory

        if fallback is not None:
            _load_auto_sharded_fallback(unet, fallback, unet_inference_memory + additional_inference_memory,
                                        unet_inference_memory // 2 + additional_inference_memory)
        else:
            try:
                memory_management.load_models_gpu(models=[single_fallback or unet] + additional_model_patchers, memory_required=unet_inference_memory + additional_inference_memory, minimum_memory_required=unet_inference_memory // 2 + additional_inference_memory)
            except Exception as error:
                if single_fallback is not None or not memory_management.is_oom(error):
                    raise
                retry_offload = getattr(unet, "enable_cpu_offload_retry", None)
                loaded_with_offload = False
                if callable(retry_offload) and retry_offload():
                    memory_management.logger.warning("Multi-GPU placement ran out of memory; retrying with CPU-offloaded stages")
                    try:
                        memory_management.soft_empty_cache(force=True)
                        memory_management.load_models_gpu(models=[unet] + additional_model_patchers, memory_required=unet_inference_memory + additional_inference_memory, minimum_memory_required=unet_inference_memory // 2 + additional_inference_memory)
                        loaded_with_offload = True
                    except Exception as retry_error:
                        if not memory_management.is_oom(retry_error):
                            raise
                if loaded_with_offload:
                    single_fallback = None
                else:
                    single_fallback = _auto_single_device_fallback(unet)
                    if single_fallback is None:
                        raise
                    memory_management.logger.warning("Automatic sharding could not fit; using single-GPU CPU offloading")
                    memory_management.soft_empty_cache(force=True)
                    memory_management.load_models_gpu(models=[single_fallback] + additional_model_patchers, memory_required=unet_inference_memory + additional_inference_memory, minimum_memory_required=unet_inference_memory // 2 + additional_inference_memory)

        real_model = unet.model
        percent_to_timestep_function = lambda p: real_model.predictor.percent_to_sigma(p)

        for cnet in unet.list_controlnets():
            cnet.pre_run(real_model, percent_to_timestep_function)
    except BaseException:
        _release_parallel_cfg(unet)
        raise


def sampling_cleanup(unet: "UnetPatcher"):
    _release_parallel_cfg(unet, preserve_cpu=True)
    for cnet in unet.list_controlnets():
        cnet.cleanup()

    memory_management.soft_empty_cache(force=False)
