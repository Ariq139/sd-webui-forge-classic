import gc
import json
import logging
import os
import threading
import uuid

import gradio as gr
import torch

from backend import memory_management
from modules import paths, shared
from modules_forge import main_thread, native_validation, native_recipes, pruna_decoder
from modules_forge.native_models import resolve_model_path, select_model_path
from modules_forge.native_pipeline_speed import (
    ATTENTION_BACKEND_CHOICES,
    OFFLOAD_CHOICES,
    configure_pipeline,
    configure_prompt_cache,
    configure_vae_tiling,
    load_native_pipeline,
    native_quantization_kwargs,
    resolve_vae_tiling_mode,
)


logger = logging.getLogger("ltx2_video")
_pipeline = None
_pipeline_key = None
_pipeline_lock = threading.RLock()
_ltx_components = []
_preset_bound = False
_DECODER_ROOTS = (
    os.path.join(paths.models_path, "VAE"),
    os.path.join(paths.models_path, "diffusers"),
)
_DECODER_EXTENSIONS = {".safetensors", ".sft", ".ckpt"}


class _LTXInterrupted(Exception):
    pass


def _option(name: str, default):
    return getattr(shared.opts, name, default)


def _auto_vae_tiling_value() -> bool:
    data = getattr(shared.opts, "data", {})
    if "ltx2_auto_vae_tiling" not in data and "ltx2_tile_vae" in data:
        return False
    return bool(_option("ltx2_auto_vae_tiling", True))


def _model_path():
    return select_model_path("LTX2", _option("forge_checkpoint_ltx2", ""), _option("ltx2_model_path", ""))


def _resolve_prunavaed_path(value: str) -> str:
    value = str(value or "").strip()
    if not value or os.path.exists(value):
        return os.path.abspath(value) if value and os.path.exists(value) else value
    for root in (*_DECODER_ROOTS, paths.models_path):
        candidate = os.path.join(root, value)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return value


def _prunavaed_choices() -> list[str]:
    values = []
    for root in _DECODER_ROOTS:
        if not os.path.isdir(root):
            continue
        for entry in os.scandir(root):
            entry_name = entry.name.lower()
            if "pruna" not in entry_name and "ltx" not in entry_name:
                continue
            has_config = any(
                os.path.isfile(os.path.join(entry.path, config_name))
                for config_name in ("config.json", "model_index.json")
            )
            if entry.is_dir() and has_config:
                values.append(entry.name)
            elif entry.is_file() and os.path.splitext(entry.name)[1].lower() in _DECODER_EXTENSIONS:
                values.append(entry.name)

    configured = str(_option("ltx2_prunavaed_path", "") or "").strip()
    if configured and os.path.exists(configured):
        configured_path = os.path.abspath(configured)
        for root in _DECODER_ROOTS:
            if os.path.dirname(configured_path).lower() == os.path.abspath(root).lower():
                configured = os.path.basename(configured_path)
                break
        if configured not in values:
            values.append(configured)
    return sorted(set(values), key=shared.natural_sort_key)


def _prunavaed_value() -> str | None:
    configured = str(_option("ltx2_prunavaed_path", "") or "").strip()
    choices = _prunavaed_choices()
    if configured in choices:
        return configured
    configured_path = _resolve_prunavaed_path(configured)
    if os.path.exists(configured_path):
        configured_path = os.path.abspath(configured_path)
        for choice in choices:
            if os.path.abspath(_resolve_prunavaed_path(choice)) == configured_path:
                return choice
    return None


def _save_option(name: str, value):
    if value is None:
        value = ""
    shared.opts.set(name, value)
    if name == "ltx2_model_path":
        shared.opts.set("forge_checkpoint_ltx2", value)
    shared.opts.save(shared.config_filename)


def _sync_preset_controls(preset: str):
    if preset != "ltx2":
        return [gr.skip() for _ in _ltx_components]

    auto_vae_tiling = _auto_vae_tiling_value()
    return [
        _model_path(),
        _prunavaed_value(),
        _option("ltx2_offload", "model"),
        _option("ltx2_attention_backend", "automatic"),
        _option("ltx2_precision", "automatic"),
        gr.update(value=auto_vae_tiling),
        gr.update(value=_option("ltx2_tile_vae", True), interactive=not auto_vae_tiling),
        _option("ltx2_prompt_cache", True),
        _option("ltx2_compile", False),
        _option("ltx2_t2i_width", 768),
        _option("ltx2_t2i_height", 512),
        _option("ltx2_t2i_batch_size", 121),
        _option("ltx2_fps", 24),
        _option("ltx2_t2i_step", 30),
        _option("ltx2_t2i_cfg", 3.0),
        _option("ltx2_stg", 1.0),
        _option("ltx2_modality", 3.0),
        _option("ltx2_audio_guidance", 7.0),
        _option("ltx2_audio_stg", 1.0),
        _option("ltx2_audio_modality", 3.0),
        _option("ltx2_guidance_rescale", 0.7),
        _option("ltx2_guidance_blocks", "28"),
        _option("ltx2_seed", -1),
        _option("ltx2_include_audio", True),
        _option("ltx2_cross_timestep_mode", "Automatic"),
        _option("ltx2_workflow", "Full"),
        _option("ltx2_quantization", "Checkpoint"),
        _option("ltx2_upsampler_path", ""),
        _option("ltx2_refine_lora", ""),
    ]


def _sync_model_to_page(preset: str, model_path: str):
    return (model_path or "") if preset == "ltx2" else gr.skip()


def _sync_model_to_quicksettings(preset: str, model_path: str):
    return model_path if preset == "ltx2" else gr.skip()


def _sync_decoder_to_page(preset: str, decoder: str):
    return decoder if preset == "ltx2" else gr.skip()


def bind_preset(preset_component, checkpoint_component=None, decoder_component=None):
    global _preset_bound

    if _preset_bound or not _ltx_components:
        return

    preset_component.change(
        fn=_sync_preset_controls,
        inputs=[preset_component],
        outputs=_ltx_components,
        queue=False,
        show_progress=False,
    )
    if checkpoint_component is not None:
        checkpoint_component.change(
            fn=_sync_model_to_page,
            inputs=[preset_component, checkpoint_component],
            outputs=[_ltx_components[0]],
            queue=False,
            show_progress=False,
        )
    if decoder_component is not None:
        decoder_component.change(
            fn=_sync_decoder_to_page,
            inputs=[preset_component, decoder_component],
            outputs=[_ltx_components[1]],
            queue=False,
            show_progress=False,
        )
    _preset_bound = True


def _default_dtype(precision: str) -> torch.dtype:
    if precision == "float16":
        return torch.float16
    if precision == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _device() -> torch.device:
    device = memory_management.get_torch_device()
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _load_prunavaed(path: str, dtype: torch.dtype):
    import diffusers
    from diffusers import AutoencoderKLLTX2Video
    from diffusers.models.autoencoders import autoencoder_kl_ltx2 as ltx_vae

    path = _resolve_prunavaed_path(path)
    if os.path.isfile(path):
        if "pruna" in os.path.basename(path).lower():
            raise ValueError("Pruned decoders require a local Diffusers directory with config.json and weights.")
        if hasattr(AutoencoderKLLTX2Video, "from_single_file"):
            return AutoencoderKLLTX2Video.from_single_file(path, torch_dtype=dtype, local_files_only=True)
        raise ValueError("This VAE class does not support single-file loading.")
    for folder in (os.path.join(path, "vae"), path):
        config_path = os.path.join(folder, "config.json")
        if not os.path.isfile(config_path):
            continue
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
        with pruna_decoder.pruna_construction(ltx_vae, diffusers.__version__, config):
            vae, loading = AutoencoderKLLTX2Video.from_pretrained(
                folder, torch_dtype=dtype, local_files_only=True, output_loading_info=True)
        problems = {key: loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
        if problems:
            raise RuntimeError(f"VAE checkpoint does not match its architecture: {problems}")
        return vae
    raise ValueError("Select a local Diffusers VAE directory containing config.json and weights.")


def _get_pipeline(model_path: str, prunavaed_path: str, precision: str, offload: str, attention_backend: str, compile_enabled: bool, prompt_cache: bool, image_mode: bool, quantization: str = "Checkpoint"):
    global _pipeline, _pipeline_key

    from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline

    model_path = resolve_model_path(model_path)
    prunavaed_path = _resolve_prunavaed_path(prunavaed_path)
    device = _device()
    dtype = _default_dtype(precision)
    key = (model_path, prunavaed_path, str(dtype), str(device), offload, attention_backend, bool(compile_enabled), bool(prompt_cache), bool(image_mode), quantization)

    with _pipeline_lock:
        if _pipeline is not None and _pipeline_key == key:
            return _pipeline

        unload()
        unload_forge_model()

        vae = None
        if prunavaed_path.strip():
            logger.info("Loading PrunaVAED decoder from %s", prunavaed_path)
            vae = _load_prunavaed(prunavaed_path, dtype)

        pipeline_kwargs = {"torch_dtype": dtype}
        pipeline_kwargs.update(native_quantization_kwargs(model_path, quantization, dtype, device, offload, ("transformer",)))
        if vae is not None:
            pipeline_kwargs["vae"] = vae

        pipeline_class = LTX2ImageToVideoPipeline if image_mode else LTX2Pipeline
        _pipeline = load_native_pipeline(pipeline_class, model_path, pipeline_kwargs, dtype, logger, "LTX-2")
        configure_pipeline(_pipeline, device, offload, attention_backend, compile_enabled, logger)
        configure_prompt_cache(_pipeline, prompt_cache, logger)
        _pipeline_key = key
        return _pipeline


def unload():
    global _pipeline, _pipeline_key

    with _pipeline_lock:
        if _pipeline is None:
            _pipeline_key = None
            return

        if _pipeline is not None:
            try:
                _pipeline.to("cpu")
            except Exception:
                pass
        _pipeline = None
        _pipeline_key = None
        gc.collect()
        memory_management.soft_empty_cache(force=True)


def unload_forge_model():
    try:
        from modules import sd_models

        sd_models.unload_model_weights()
    except Exception:
        logger.debug("No Forge image model needed unloading before LTX-2.3", exc_info=True)


def interrupt_generation():
    if shared.state is not None:
        shared.state.interrupt()


def _parse_blocks(value: str) -> list[int] | None:
    blocks = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return [int(item) for item in blocks] if blocks else None


def _output_path() -> str:
    output_dir = os.path.abspath(shared.opts.outdir_videos)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"ltx2-{uuid.uuid4().hex}.mp4")


def generate(
    model_path: str,
    prunavaed_path: str,
    offload: str,
    attention_backend: str,
    precision: str,
    auto_vae_tiling: bool,
    tile_vae: bool,
    prompt_cache: bool,
    compile_enabled: bool,
    prompt: str,
    negative_prompt: str,
    image,
    width: int,
    height: int,
    frames: int,
    fps: float,
    steps: int,
    guidance: float,
    stg: float,
    modality: float,
    audio_guidance: float,
    audio_stg: float,
    audio_modality: float,
    guidance_rescale: float,
    guidance_blocks: str,
    seed: int,
    include_audio: bool,
    cross_timestep_mode: str = "Automatic",
    workflow: str = "Full",
    quantization: str = "Checkpoint",
    upsampler_path: str = "",
    refine_lora: str = "",
    progress=gr.Progress(track_tqdm=True),
):
    if not model_path or not model_path.strip():
        return None, "Select a compatible LTX-2.3 Diffusers model from the global checkpoint dropdown."
    if not prompt or not prompt.strip():
        return None, "Enter a prompt."

    pipeline_attempted = False
    try:
        import diffusers
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

        if workflow not in native_recipes.LTX_MODES:
            raise ValueError("Unknown LTX workflow.")
        distilled = workflow.startswith("Distilled")
        two_stage = workflow.endswith("2x")
        if two_stage:
            if offload != "model" or compile_enabled:
                raise ValueError("Two-stage LTX currently requires model CPU offload and compile disabled.")
            upsampler_path = os.path.abspath(upsampler_path or os.path.join(resolve_model_path(model_path), "latent_upsampler"))
            if not os.path.isfile(os.path.join(upsampler_path, "config.json")):
                raise ValueError("Choose a compatible local latent upsampler directory.")
            if not distilled:
                if quantization != "Checkpoint":
                    raise ValueError("Full + refine requires an unquantized checkpoint and a compatible distilled refinement LoRA.")
                refine_lora = os.path.abspath(refine_lora)
                if not os.path.isfile(refine_lora) or not refine_lora.lower().endswith(".safetensors"):
                    raise ValueError("Full + refine requires a local distilled refinement LoRA (.safetensors).")
        if distilled:
            guidance, stg, modality = 1.0, 0.0, 1.0
            audio_guidance, audio_stg, audio_modality = 1.0, 0.0, 1.0
            guidance_rescale, guidance_blocks = 0.0, ""
            steps = len(DISTILLED_SIGMA_VALUES)
        width, height, frames, fps, steps = native_validation.ltx_dimensions(width, height, frames, fps, steps)
        guidance = native_validation.finite_number(guidance, "Video CFG", 0, 24)
        stg = native_validation.finite_number(stg, "Video STG", 0)
        modality = native_validation.finite_number(modality, "Video modality", 0)
        audio_guidance = native_validation.audio_override(audio_guidance, guidance, "Audio CFG", version=diffusers.__version__)
        audio_stg = native_validation.audio_override(audio_stg, stg, "Audio STG", version=diffusers.__version__)
        audio_modality = native_validation.audio_override(audio_modality, modality, "Audio modality", version=diffusers.__version__)
        guidance_rescale = native_validation.finite_number(guidance_rescale, "Guidance rescale", 0, 1)
        blocks = _parse_blocks(guidance_blocks)
        if (stg > 0 or audio_stg > 0) and not blocks:
            raise ValueError("Supply STG block indices when using STG.")
        if blocks and (min(blocks) < 0 or len(set(blocks)) != len(blocks)):
            raise ValueError("STG block indices must be unique and nonnegative.")
        seed = native_validation.actual_seed(seed)
        if shared.state.interrupted or shared.state.stopping_generation:
            raise _LTXInterrupted()

        pipeline_attempted = True
        pipe = _get_pipeline(model_path, prunavaed_path, precision, offload, attention_backend, compile_enabled, prompt_cache, image is not None, quantization)
        temporal_ratio = getattr(pipe, "vae_temporal_compression_ratio", 8)
        native_validation.ltx_dimensions(width, height, frames, fps, steps, temporal_ratio)
        cross_timestep = native_validation.ltx_cross_timestep(pipe.transformer.config, cross_timestep_mode)
        num_layers = getattr(pipe.transformer.config, "num_layers", None)
        if blocks and num_layers is not None and max(blocks) >= num_layers:
            raise ValueError(f"STG block indices must be below {num_layers}.")
        configure_vae_tiling(
            pipe.vae,
            resolve_vae_tiling_mode(auto_vae_tiling, tile_vae),
            _device(),
            width,
            height,
            frames,
        )

        total_steps = steps + (len(STAGE_2_DISTILLED_SIGMA_VALUES) if two_stage else 0)
        stage_offset = 0
        shared.state.sampling_steps = total_steps
        shared.state.sampling_step = 0
        shared.state.job_count = 1
        shared.state.job_no = 0
        shared.state.current_image = None
        shared.state.id_live_preview += 1
        shared.state.textinfo = "LTX-2.3: loading and preparing"

        generator = torch.Generator(device=_device()).manual_seed(seed)

        def on_step_end(_pipe, step, timestep, callback_kwargs):
            if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
                raise _LTXInterrupted()

            completed = stage_offset + step + 1
            shared.state.sampling_step = completed
            shared.state.textinfo = f"LTX-2.3: step {completed}/{total_steps}"
            progress(completed, total=total_steps, desc=f"LTX-2.3 step {completed}/{total_steps}")
            return callback_kwargs

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_frames=frames,
            frame_rate=float(fps),
            num_inference_steps=steps,
            guidance_scale=float(guidance),
            stg_scale=float(stg),
            modality_scale=float(modality),
            audio_guidance_scale=audio_guidance,
            audio_stg_scale=audio_stg,
            audio_modality_scale=audio_modality,
            guidance_rescale=float(guidance_rescale),
            spatio_temporal_guidance_blocks=blocks,
            use_cross_timestep=cross_timestep,
            generator=generator,
            output_type="np",
            return_dict=False,
            callback_on_step_end=on_step_end,
        )

        if image is not None:
            kwargs["image"] = image

        first, refine = native_recipes.ltx_recipe(workflow, kwargs, DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES)
        original_scheduler = pipe.scheduler
        loaded_adapter = False
        with _pipeline_lock:
            try:
                if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
                    raise _LTXInterrupted()
                if distilled:
                    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(original_scheduler.config, use_dynamic_shifting=False, shift_terminal=None)
                video, audio = pipe(**first)
                if refine is not None:
                    if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
                        raise _LTXInterrupted()
                    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
                    shared.state.textinfo = "LTX-2.3: latent upsampling"
                    upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                        upsampler_path, torch_dtype=_default_dtype(precision), local_files_only=True)
                    try:
                        upsampler.to(_device())
                        with torch.inference_mode():
                            video = upsampler(video.to(device=_device(), dtype=_default_dtype(precision)))
                    finally:
                        del upsampler
                        memory_management.soft_empty_cache(force=True)
                    if not distilled:
                        if getattr(pipe.transformer, "is_quantized", False) or getattr(pipe.transformer, "hf_quantizer", None) is not None:
                            raise ValueError("Refinement LoRA is not supported on a quantized transformer in this workflow.")
                        pipe.load_lora_weights(os.path.dirname(refine_lora), weight_name=os.path.basename(refine_lora), adapter_name="forge_stage2")
                        loaded_adapter = True
                        pipe.set_adapters("forge_stage2", adapter_weights=1.0)
                    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(original_scheduler.config, use_dynamic_shifting=False, shift_terminal=None)
                    stage_offset = steps
                    refine.update(latents=video, audio_latents=audio)
                    configure_vae_tiling(pipe.vae, resolve_vae_tiling_mode(auto_vae_tiling, tile_vae), _device(), width * 2, height * 2, frames)
                    if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
                        raise _LTXInterrupted()
                    video, audio = pipe(**refine)
            finally:
                pipe.scheduler = original_scheduler
                if loaded_adapter:
                    pipe.unload_lora_weights()
        if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
            raise _LTXInterrupted()
        output_path = _output_path()

        from diffusers.utils import encode_video

        audio_tensor = None
        audio_rate = None
        if include_audio and audio is not None:
            audio_tensor = audio[0].float().cpu()
            audio_rate = getattr(pipe.vocoder.config, "output_sampling_rate", 24000)

        temporary = os.path.splitext(output_path)[0] + ".partial.mp4"
        try:
            encode_video(video[0], fps=fps, audio=audio_tensor, audio_sample_rate=audio_rate, output_path=temporary)
            if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
                raise _LTXInterrupted()
            os.replace(temporary, output_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        sampling = {key: value for key, value in first.items() if key not in ("generator", "callback_on_step_end", "image")}
        metadata = dict(native_model="ltx2", prompt=prompt, negative_prompt=negative_prompt, seed=seed, request_seed=seed,
                        sd_model_name=model_path, sampler_name="LTX-2.3", workflow=workflow,
                        width=width * (2 if two_stage else 1), height=height * (2 if two_stage else 1),
                        base_width=width, base_height=height, frames=len(video[0]), fps=fps, steps=total_steps,
                        cfg_scale=first["guidance_scale"], sampling=sampling, include_audio=audio_tensor is not None,
                        quantization=quantization, precision=precision, decoder=prunavaed_path,
                        upsampler=upsampler_path if two_stage else "", refinement_lora=refine_lora if two_stage and not distilled else "",
                        cross_timestep=cross_timestep, image_conditioned=image is not None,
                        runtime=getattr(pipe, "_forge_runtime", {}))
        if refine is not None:
            metadata["refinement_sampling"] = {key: value for key, value in refine.items() if key not in ("generator", "callback_on_step_end", "image", "latents", "audio_latents")}
        warning = ""
        try:
            native_recipes.atomic_json(os.path.splitext(output_path)[0] + ".native.json", metadata)
        except Exception:
            logger.exception("Failed to save native video metadata")
            warning = " | Warning: metadata sidecar could not be saved."
        shared.state.sampling_step = total_steps
        shared.state.textinfo = "LTX-2.3: finished"
        return output_path, f"Saved: {output_path} | Seed: {seed} | Frames: {len(video[0])} | Cross timestep: {cross_timestep}{warning}"
    except Exception as error:
        # Callback exceptions bypass Diffusers' normal final offload hooks.
        # Invalidate the failed pipeline rather than reusing partial hook state.
        if pipeline_attempted:
            try:
                unload()
            except Exception:
                logger.exception("Native pipeline cleanup failed")
        if isinstance(error, _LTXInterrupted):
            shared.state.textinfo = "LTX-2.3: interrupted"
            return None, "LTX-2.3 generation interrupted."
        if memory_management.is_oom(error):
            logger.exception("LTX-2.3 generation ran out of memory")
            if getattr(shared.opts, "forge_oom_retry_enabled", True):
                raise
            shared.state.textinfo = "LTX-2.3: out of memory"
            return None, "LTX-2.3 ran out of memory. Reduce resolution, frames, or enable sequential CPU offload."
        logger.exception("LTX-2.3 generation failed")
        shared.state.textinfo = "LTX-2.3: failed"
        return None, f"LTX-2.3 error: {error}"


def generate_from_preset(prompt: str, negative_prompt: str, image=None, *, width=None, height=None, frames=None, steps=None, guidance=None, seed=None):
    """Run LTX with the values stored by the shared ``ltx2`` preset."""
    return generate(
        _model_path(),
        _prunavaed_value() or "",
        _option("ltx2_offload", "model"),
        _option("ltx2_attention_backend", "automatic"),
        _option("ltx2_precision", "automatic"),
        _auto_vae_tiling_value(),
        _option("ltx2_tile_vae", True),
        _option("ltx2_prompt_cache", True),
        _option("ltx2_compile", False),
        prompt,
        negative_prompt,
        image,
        width if width is not None else _option("ltx2_t2i_width", 768),
        height if height is not None else _option("ltx2_t2i_height", 512),
        frames if frames is not None else _option("ltx2_t2i_batch_size", 121),
        _option("ltx2_fps", 24),
        steps if steps is not None else _option("ltx2_t2i_step", 30),
        guidance if guidance is not None else _option("ltx2_t2i_cfg", 3.0),
        _option("ltx2_stg", 1.0),
        _option("ltx2_modality", 3.0),
        _option("ltx2_audio_guidance", 7.0),
        _option("ltx2_audio_stg", 1.0),
        _option("ltx2_audio_modality", 3.0),
        _option("ltx2_guidance_rescale", 0.7),
        _option("ltx2_guidance_blocks", "28"),
        seed if seed is not None else _option("ltx2_seed", -1),
        _option("ltx2_include_audio", True),
        _option("ltx2_cross_timestep_mode", "Automatic"),
        _option("ltx2_workflow", "Full"),
        _option("ltx2_quantization", "Checkpoint"),
        _option("ltx2_upsampler_path", ""),
        _option("ltx2_refine_lora", ""),
    )


def create_ui():
    global _ltx_components, _preset_bound
    _preset_bound = False

    from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call

    with gr.Blocks(analytics_enabled=False) as interface:
        gr.Markdown(
            "## LTX-2.3 Video\n"
            "Use a Diffusers-format LTX-2.3 model for text-to-video or image-to-video. "
            "The optional VAE / Decoder selector is in the global quicksettings row. "
            "LTX's video VAE, audio stack, and text encoder come from the selected model."
        )

        # The global quicksettings row owns the model and decoder selections.
        model_path = gr.Textbox(value=_model_path(), visible=False)
        prunavaed_path = gr.Textbox(value=_prunavaed_value() or "", visible=False)

        with gr.Row():
            offload = gr.Radio(OFFLOAD_CHOICES, value=_option("ltx2_offload", "model"), label="CPU offload")
            attention_backend = gr.Dropdown(ATTENTION_BACKEND_CHOICES, value=_option("ltx2_attention_backend", "automatic"), label="Attention backend")
            precision = gr.Radio(["automatic", "bfloat16", "float16"], value=_option("ltx2_precision", "automatic"), label="Precision")
            auto_vae_tiling = gr.Checkbox(value=_auto_vae_tiling_value(), label="Automatic VAE tiling")
            tile_vae = gr.Checkbox(value=_option("ltx2_tile_vae", True), label="Force VAE tiling", interactive=not _auto_vae_tiling_value())
            prompt_cache = gr.Checkbox(value=_option("ltx2_prompt_cache", True), label="Prompt cache")
            compile_enabled = gr.Checkbox(value=_option("ltx2_compile", False), label="Compile denoiser")

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", lines=5)
                negative_prompt = gr.Textbox(label="Negative prompt", lines=3)
                image = gr.Image(label="Image condition (optional)", type="pil")
            with gr.Column(scale=1, min_width=180):
                generate_button = gr.Button("Generate", variant="primary")
                with gr.Row():
                    interrupt_button = gr.Button("Interrupt", variant="stop")
                    unload_button = gr.Button("Unload", variant="secondary")
                status = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    width = gr.Number(value=_option("ltx2_t2i_width", 768), minimum=64, maximum=2048, step=32, label="Width")
                    height = gr.Number(value=_option("ltx2_t2i_height", 512), minimum=64, maximum=2048, step=32, label="Height")
                    frames = gr.Number(value=_option("ltx2_t2i_batch_size", 121), minimum=9, maximum=241, step=8, label="Frames")
                    fps = gr.Number(value=_option("ltx2_fps", 24), minimum=1, maximum=60, step=1, label="FPS")
                with gr.Row():
                    steps = gr.Number(value=_option("ltx2_t2i_step", 30), minimum=1, maximum=100, step=1, label="Sampling steps")
                    guidance = gr.Number(value=_option("ltx2_t2i_cfg", 3.0), minimum=0, maximum=20, step=0.1, label="CFG scale")
                    seed = gr.Number(value=_option("ltx2_seed", -1), minimum=-1, maximum=2**31 - 1, step=1, label="Seed")

                with gr.Accordion("LTX-2.3 options", open=False):
                    with gr.Row():
                        stg = gr.Number(value=_option("ltx2_stg", 1.0), minimum=0, maximum=10, step=0.1, label="Video STG")
                        modality = gr.Number(value=_option("ltx2_modality", 3.0), minimum=0, maximum=10, step=0.1, label="Video modality")
                        guidance_rescale = gr.Number(value=_option("ltx2_guidance_rescale", 0.7), minimum=0, maximum=1, step=0.05, label="Guidance rescale")
                    with gr.Row():
                        audio_guidance = gr.Number(value=_option("ltx2_audio_guidance", 7.0), minimum=0, maximum=20, step=0.1, label="Audio CFG")
                        audio_stg = gr.Number(value=_option("ltx2_audio_stg", 1.0), minimum=0, maximum=10, step=0.1, label="Audio STG")
                        audio_modality = gr.Number(value=_option("ltx2_audio_modality", 3.0), minimum=0, maximum=10, step=0.1, label="Audio modality")
                    with gr.Row():
                        guidance_blocks = gr.Textbox(value=_option("ltx2_guidance_blocks", "28"), label="Spatio-temporal guidance blocks")
                        include_audio = gr.Checkbox(value=_option("ltx2_include_audio", True), label="Include generated audio", info="Controls exported audio, not joint denoising.")
                    cross_timestep_mode = gr.Dropdown(choices=["Automatic", "LTX-2.3", "LTX-2.0"], value=_option("ltx2_cross_timestep_mode", "Automatic"), label="Cross timestep mode", info="Automatic detects prompt modulation in the transformer config.")
                    workflow = gr.Dropdown(native_recipes.LTX_MODES, value=_option("ltx2_workflow", "Full"), label="Workflow", info="Distilled modes require a matching distilled checkpoint; two-stage output doubles the base resolution.")
                    quantization = gr.Dropdown(native_recipes.QUANTIZATION_MODES, value=_option("ltx2_quantization", "Checkpoint"), label="Native quantization (experimental)")
                    upsampler_path = gr.Textbox(value=_option("ltx2_upsampler_path", ""), label="Local latent upsampler", info="Leave blank to use the checkpoint's latent_upsampler directory.")
                    refine_lora = gr.Textbox(value=_option("ltx2_refine_lora", ""), label="Local distilled refinement LoRA", info="Required for Full + refine 2x.")
            with gr.Column(scale=2):
                video_output = gr.Video(label="Output", autoplay=True, include_audio=True)

        _ltx_components = [
            model_path,
            prunavaed_path,
            offload,
            attention_backend,
            precision,
            auto_vae_tiling,
            tile_vae,
            prompt_cache,
            compile_enabled,
            width,
            height,
            frames,
            fps,
            steps,
            guidance,
            stg,
            modality,
            audio_guidance,
            audio_stg,
            audio_modality,
            guidance_rescale,
            guidance_blocks,
            seed,
            include_audio,
            cross_timestep_mode,
            workflow,
            quantization,
            upsampler_path,
            refine_lora,
        ]

        settings_to_save = {
            model_path: "ltx2_model_path",
            offload: "ltx2_offload",
            attention_backend: "ltx2_attention_backend",
            precision: "ltx2_precision",
            auto_vae_tiling: "ltx2_auto_vae_tiling",
            tile_vae: "ltx2_tile_vae",
            prompt_cache: "ltx2_prompt_cache",
            compile_enabled: "ltx2_compile",
            width: "ltx2_t2i_width",
            height: "ltx2_t2i_height",
            frames: "ltx2_t2i_batch_size",
            fps: "ltx2_fps",
            steps: "ltx2_t2i_step",
            guidance: "ltx2_t2i_cfg",
            stg: "ltx2_stg",
            modality: "ltx2_modality",
            audio_guidance: "ltx2_audio_guidance",
            audio_stg: "ltx2_audio_stg",
            audio_modality: "ltx2_audio_modality",
            guidance_rescale: "ltx2_guidance_rescale",
            guidance_blocks: "ltx2_guidance_blocks",
            seed: "ltx2_seed",
            include_audio: "ltx2_include_audio",
            cross_timestep_mode: "ltx2_cross_timestep_mode",
            workflow: "ltx2_workflow",
            quantization: "ltx2_quantization",
            upsampler_path: "ltx2_upsampler_path",
            refine_lora: "ltx2_refine_lora",
        }
        for component, key in settings_to_save.items():
            if component is model_path:
                continue  # The global checkpoint selector owns model persistence.
            component.input(fn=lambda value, key=key: _save_option(key, value), inputs=[component], outputs=[], queue=False, show_progress=False)

        inputs = [
            model_path,
            prunavaed_path,
            offload,
            attention_backend,
            precision,
            auto_vae_tiling,
            tile_vae,
            prompt_cache,
            compile_enabled,
            prompt,
            negative_prompt,
            image,
            width,
            height,
            frames,
            fps,
            steps,
            guidance,
            stg,
            modality,
            audio_guidance,
            audio_stg,
            audio_modality,
            guidance_rescale,
            guidance_blocks,
            seed,
            include_audio,
            cross_timestep_mode,
            workflow,
            quantization,
            upsampler_path,
            refine_lora,
        ]
        from modules_forge.native_requests import PageJobOwner
        from uuid import uuid4

        owner = PageJobOwner()
        session = gr.State(value=lambda: uuid4().hex)

        def run_page(session_id, *args):
            # Resolve random seeds once, before the worker's optional OOM retry.
            import inspect
            bound = inspect.signature(generate).bind(*args)
            bound.arguments["seed"] = native_validation.actual_seed(bound.arguments["seed"])
            with owner.active(session_id):
                result = main_thread.run_and_wait_result(generate, *bound.args, **bound.kwargs)
                if result is None:
                    raise RuntimeError("Native generation failed")
                output_value, message = result
                return (output_value if output_value else gr.skip()), message

        generate_ltx = wrap_gradio_gpu_call(run_page, extra_outputs=[gr.skip()])
        generate_button.click(generate_ltx, inputs=[session, *inputs], outputs=[video_output, status])
        prompt.submit(generate_ltx, inputs=[session, *inputs], outputs=[video_output, status])
        auto_vae_tiling.change(
            fn=lambda enabled: gr.update(interactive=not enabled),
            inputs=[auto_vae_tiling],
            outputs=[tile_vae],
            queue=False,
            show_progress=False,
        )
        interrupt_button.click(lambda session_id: owner.interrupt(session_id, interrupt_generation), inputs=[session], outputs=[], queue=False, show_progress=False)
        unload_button.click(wrap_queued_call(unload), outputs=[])

    return interface
