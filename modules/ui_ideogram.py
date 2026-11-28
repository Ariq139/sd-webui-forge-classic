import gc
import sys
import logging
import threading

import gradio as gr
import torch

from backend import memory_management
from modules import shared
from modules_forge import main_thread, native_requests, native_validation, native_recipes, native_outputs, native_runtime
from modules_forge.native_models import resolve_model_path, select_model_path
from modules_forge.native_pipeline_speed import (
    ATTENTION_BACKEND_CHOICES,
    OFFLOAD_CHOICES,
    configure_pipeline,
    configure_prompt_cache,
    configure_vae_tiling,
    load_native_pipeline,
    native_quantization_kwargs,
    require_dependency,
    resolve_vae_tiling_mode,
)


logger = logging.getLogger("ideogram")
_pipeline = None
_pipeline_key = None
_pipeline_lock = threading.RLock()
_preset_components = []
_preset_bound = False


class _IdeogramInterrupted(Exception):
    pass


def _option(name: str, default):
    return getattr(shared.opts, name, default)


def _auto_vae_tiling_value() -> bool:
    data = getattr(shared.opts, "data", {})
    if "ideogram_auto_vae_tiling" not in data and "ideogram_tile_vae" in data:
        return False
    return bool(_option("ideogram_auto_vae_tiling", True))


def _model_path() -> str:
    return select_model_path("Ideogram4", _option("forge_checkpoint_ideogram", ""), _option("ideogram_model_path", ""))


def _save_option(name: str, value):
    shared.opts.set(name, value)
    if name == "ideogram_model_path":
        shared.opts.set("forge_checkpoint_ideogram", value)
    shared.opts.save(shared.config_filename)


def _sync_preset_controls(preset: str):
    if preset != "ideogram":
        return [gr.skip() for _ in _preset_components]

    auto_vae_tiling = _auto_vae_tiling_value()
    return [
        _model_path(),
        _option("ideogram_offload", "model"),
        _option("ideogram_attention_backend", "automatic"),
        _option("ideogram_precision", "automatic"),
        gr.update(value=auto_vae_tiling),
        gr.update(value=_option("ideogram_tile_vae", True), interactive=not auto_vae_tiling),
        _option("ideogram_prompt_cache", True),
        _option("ideogram_compile", False),
        _option("ideogram_t2i_width", 1024),
        _option("ideogram_t2i_height", 1024),
        _option("ideogram_t2i_batch_size", 1),
        _option("ideogram_t2i_step", 48),
        _option("ideogram_t2i_cfg", 7.0),
        _option("ideogram_mu", 0.0),
        _option("ideogram_std", 1.5),
        _option("ideogram_prompt_upsampling", False),
        _option("ideogram_prompt_temperature", 1.0),
        _option("ideogram_seed", -1),
        _option("ideogram_polish_guidance", True),
        _option("ideogram_recipe", "Custom"),
        _option("ideogram_custom_schedule", ""),
        _option("ideogram_quantization", "Checkpoint"),
        _option("ideogram_enhancer_path", ""),
    ]


def _sync_model_to_page(preset: str, model_path: str):
    return (model_path or "") if preset == "ideogram" else gr.skip()


def _sync_model_to_quicksettings(preset: str, model_path: str):
    return model_path if preset == "ideogram" else gr.skip()


def bind_preset(preset_component, checkpoint_component=None):
    global _preset_bound

    if _preset_bound or not _preset_components:
        return

    preset_component.change(
        fn=_sync_preset_controls,
        inputs=[preset_component],
        outputs=_preset_components,
        queue=False,
        show_progress=False,
    )
    if checkpoint_component is not None:
        checkpoint_component.change(
            fn=_sync_model_to_page,
            inputs=[preset_component, checkpoint_component],
            outputs=[_preset_components[0]],
            queue=False,
            show_progress=False,
        )
    _preset_bound = True


def _default_dtype(precision: str) -> torch.dtype:
    if precision == "automatic" and _device().type == "cpu":
        return torch.float32
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


def _get_pipeline(model_path: str, precision: str, offload: str, attention_backend: str, compile_enabled: bool, prompt_cache: bool, quantization="Checkpoint", enhancer_path=""):
    global _pipeline, _pipeline_key

    from diffusers import Ideogram4Pipeline

    model_path = native_runtime.require_pipeline_directory(resolve_model_path(model_path), "Ideogram4")
    device = _device()
    dtype = _default_dtype(precision)
    key = (model_path, str(dtype), str(device), offload, attention_backend, bool(compile_enabled), bool(prompt_cache), quantization, enhancer_path)

    with _pipeline_lock:
        if _pipeline is not None and _pipeline_key == key:
            return _pipeline

        unload()
        # The caller owns the generation queue. Release the other native
        # pipeline before loading this one, including dedicated-page calls.
        peer = sys.modules.get("modules.ui_ltx2_video")
        if peer is not None:
            peer.unload()
        unload_forge_model()
        pipeline_kwargs = {"torch_dtype": dtype}
        pipeline_kwargs.update(native_quantization_kwargs(model_path, quantization, dtype, device, offload, ("transformer", "unconditional_transformer")))
        if enhancer_path:
            import os
            from diffusers import Ideogram4PromptEnhancerHead

            enhancer_path = resolve_model_path(enhancer_path)
            if not os.path.isdir(enhancer_path):
                raise ValueError("Select a local Ideogram prompt enhancer head directory.")
            pipeline_kwargs["prompt_enhancer_head"] = Ideogram4PromptEnhancerHead.from_pretrained(enhancer_path, torch_dtype=dtype, local_files_only=True)
        _pipeline = load_native_pipeline(Ideogram4Pipeline, model_path, pipeline_kwargs, dtype, logger, "Ideogram 4")
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

        pipe, _pipeline = _pipeline, None
        _pipeline_key = None
        try:
            native_runtime.release_runtime(pipe)
        finally:
            del pipe
            gc.collect()
            memory_management.soft_empty_cache(force=True)


def unload_forge_model():
    try:
        from modules import sd_models

        sd_models.unload_model_weights(include_native=False)
    except Exception:
        logger.debug("No Forge image model needed unloading before Ideogram 4", exc_info=True)


def interrupt_generation():
    if shared.state is not None:
        shared.state.interrupt()


def generate(
    model_path: str,
    offload: str,
    attention_backend: str,
    precision: str,
    auto_vae_tiling: bool,
    tile_vae: bool,
    prompt_cache: bool,
    compile_enabled: bool,
    prompt: str,
    width: int,
    height: int,
    num_images: int,
    steps: int,
    guidance: float,
    mu: float,
    std: float,
    prompt_upsampling: bool,
    prompt_temperature: float,
    seed: int,
    polish_guidance: bool = True,
    recipe: str = "Custom",
    custom_schedule: str = "",
    quantization: str = "Checkpoint",
    enhancer_path: str = "",
    progress=gr.Progress(track_tqdm=True),
):
    if not model_path or not model_path.strip():
        return [], "Select a compatible Ideogram 4 Diffusers model from the global checkpoint dropdown."
    if not prompt or not prompt.strip():
        return [], "Enter a prompt."

    pipeline_attempted = False
    try:
        width = native_requests.positive_integer(width, "Width")
        height = native_requests.positive_integer(height, "Height")
        num_images = native_requests.positive_integer(num_images, "Batch size")
        steps = native_requests.positive_integer(steps, "Sampling steps")
        if width < 64 or height < 64:
            return [], "Ideogram 4 width and height must be at least 64."
        if not 1 <= num_images <= 8:
            return [], "Ideogram 4 supports 1 to 8 images per prompt in this UI."
        if not 1 <= steps <= 150:
            return [], "Ideogram 4 steps must be between 1 and 150."

        guidance = native_validation.finite_number(guidance, "CFG scale", 0, 24)
        mu = native_validation.finite_number(mu, "Schedule mu", -10, 10)
        std = native_validation.finite_number(std, "Schedule std", 0.01, 10)
        recipe_kwargs = native_recipes.ideogram_recipe(recipe, steps, guidance, mu, std, polish_guidance, custom_schedule)
        steps = recipe_kwargs["num_inference_steps"]
        if prompt_upsampling:
            require_dependency("outlines", "Structured Ideogram prompt upsampling")
        prompt_temperature = native_validation.finite_number(prompt_temperature, "Upsampling temperature", 0.1, 2)
        seed = native_validation.actual_seed(seed)
        if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
            raise _IdeogramInterrupted()

        shared.state.job_count = 1
        shared.state.job_no = 0
        shared.state.sampling_step = 0
        shared.state.sampling_steps = steps
        pipeline_attempted = True
        shared.state.textinfo = "Ideogram 4: loading model"
        pipe = _get_pipeline(model_path, precision, offload, attention_backend, compile_enabled, prompt_cache, quantization, enhancer_path)
        if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
            raise _IdeogramInterrupted()
        divisor = pipe.vae_scale_factor * pipe.patch_size
        if width % divisor or height % divisor:
            return [], f"Ideogram 4 width and height must be divisible by {divisor} for this model."
        configure_vae_tiling(
            pipe.vae,
            resolve_vae_tiling_mode(auto_vae_tiling, tile_vae),
            _device(),
            width,
            height,
            num_images,
        )

        if prompt_upsampling and getattr(pipe, "prompt_enhancer_head", None) is None:
            return [], "Ideogram 4 prompt upsampling is unavailable for this model; use a model with prompt_enhancer_head."

        shared.state.sampling_steps = steps
        shared.state.sampling_step = 0
        shared.state.job_count = 1
        shared.state.job_no = 0
        shared.state.current_image = None
        shared.state.id_live_preview += 1
        shared.state.textinfo = "Ideogram 4: loading and preparing"

        seeds = native_recipes.seed_sequence(seed, num_images)
        generator = [torch.Generator(device=_device()).manual_seed(s) for s in seeds]
        effective_prompt = prompt
        if prompt_upsampling:
            shared.state.textinfo = "Ideogram 4: enhancing prompt"
            from accelerate.hooks import remove_hook_from_module
            with native_runtime.ideogram_enhancement_device(pipe, remove_hook_from_module) as enhancer_device, torch.inference_mode():
                enhanced = pipe.upsample_prompt(prompt, height=height, width=width, temperature=prompt_temperature,
                                                generator=torch.Generator(device=enhancer_device).manual_seed(seed), device=enhancer_device)
            effective_prompt = enhanced[0] if isinstance(enhanced, list) else enhanced
            import json
            caption = json.loads(effective_prompt)
            if not isinstance(caption, dict) or not isinstance(caption.get("high_level_description"), str) or not isinstance(caption.get("compositional_deconstruction"), dict):
                raise ValueError("Prompt enhancer did not produce the expected structured caption.")
            if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
                raise _IdeogramInterrupted()

        def on_step_end(_pipe, step, _timestep, callback_kwargs):
            if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
                raise _IdeogramInterrupted()

            shared.state.sampling_step = step + 1
            shared.state.textinfo = f"Ideogram 4: step {step + 1}/{steps}"
            progress(step + 1, total=steps, desc=f"Ideogram 4 step {step + 1}/{steps}")
            return callback_kwargs

        with _pipeline_lock:
            if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
                raise _IdeogramInterrupted()
            result = pipe(
                prompt=effective_prompt,
                height=height,
                width=width,
                **recipe_kwargs,
                prompt_upsampling=False,
                num_images_per_prompt=num_images,
                generator=generator,
                output_type="pil",
                return_dict=True,
                callback_on_step_end=on_step_end,
            )

        if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
            raise _IdeogramInterrupted()
        images = list(result.images if hasattr(result, "images") else result[0])
        if len(images) != len(seeds):
            raise RuntimeError("Ideogram returned an unexpected image count.")
        shared.state.textinfo = "Ideogram 4: saving results"
        schedule = recipe_kwargs["guidance_schedule"]
        metadata = dict(native_model="ideogram", prompt=prompt, effective_prompt=effective_prompt, negative_prompt="",
                        request_seed=seed, width=width, height=height, steps=steps,
                        cfg_scale=schedule[0] if schedule else recipe_kwargs["guidance_scale"],
                        sampler_name="Ideogram 4", sd_model_name=model_path, recipe=recipe,
                        quantization=quantization, precision=precision, enhancer_path=enhancer_path,
                        prompt_upsampling=bool(prompt_upsampling), prompt_temperature=prompt_temperature,
                        sampling=recipe_kwargs, runtime=getattr(pipe, "_forge_runtime", {}))
        saved, warnings = native_outputs.finalize_images(images, metadata, seeds)
        shared.state.current_image = images[0]
        shared.state.sampling_step = steps
        shared.state.textinfo = "Ideogram 4: finished"

        message = f"Ideogram 4 generated {len(images)} image(s). Request seed: {seed}. Saved: {len(saved)}."
        if warnings:
            message += " " + "; ".join(warnings)
        return images, message
    except Exception as error:
        # Callback exceptions bypass Diffusers' normal final offload hooks.
        # Invalidate the failed pipeline rather than reusing partial hook state.
        if pipeline_attempted:
            try:
                unload()
            except Exception:
                logger.exception("Native pipeline cleanup failed")
        if isinstance(error, _IdeogramInterrupted):
            shared.state.textinfo = "Ideogram 4: interrupted"
            return [], "Ideogram 4 generation interrupted."
        if memory_management.is_oom(error):
            logger.exception("Ideogram 4 generation ran out of memory")
            if getattr(shared.opts, "forge_oom_retry_enabled", True):
                raise
            shared.state.textinfo = "Ideogram 4: out of memory"
            return [], "Ideogram 4 ran out of memory. Reduce resolution, image count, or enable sequential CPU offload."
        logger.exception("Ideogram 4 generation failed")
        shared.state.textinfo = "Ideogram 4: failed"
        return [], f"Ideogram 4 error: {error}"


def generate_from_preset(prompt: str, *, width=None, height=None, num_images=None, steps=None, guidance=None, seed=None):
    return generate(
        _model_path(),
        _option("ideogram_offload", "model"),
        _option("ideogram_attention_backend", "automatic"),
        _option("ideogram_precision", "automatic"),
        _auto_vae_tiling_value(),
        _option("ideogram_tile_vae", True),
        _option("ideogram_prompt_cache", True),
        _option("ideogram_compile", False),
        prompt,
        width if width is not None else _option("ideogram_t2i_width", 1024),
        height if height is not None else _option("ideogram_t2i_height", 1024),
        num_images if num_images is not None else _option("ideogram_t2i_batch_size", 1),
        steps if steps is not None else _option("ideogram_t2i_step", 48),
        guidance if guidance is not None else _option("ideogram_t2i_cfg", 7.0),
        _option("ideogram_mu", 0.0),
        _option("ideogram_std", 1.5),
        _option("ideogram_prompt_upsampling", False),
        _option("ideogram_prompt_temperature", 1.0),
        seed if seed is not None else _option("ideogram_seed", -1),
        _option("ideogram_polish_guidance", True),
        _option("ideogram_recipe", "Custom"),
        _option("ideogram_custom_schedule", ""),
        _option("ideogram_quantization", "Checkpoint"),
        _option("ideogram_enhancer_path", ""),
    )


def create_ui():
    global _preset_components, _preset_bound
    _preset_bound = False

    from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call

    with gr.Blocks(analytics_enabled=False) as interface:
        gr.Markdown(
            "## Ideogram 4\n"
            "Native Diffusers text-to-image generation. Select the model from the global Checkpoint quicksetting; its VAE and text encoder are bundled with the model."
        )

        # The visible model selector is Forge's global checkpoint quicksetting.
        # This hidden mirror keeps the standalone page on the same selection.
        model_path = gr.Textbox(value=_model_path(), visible=False)

        with gr.Row():
            offload = gr.Radio(OFFLOAD_CHOICES, value=_option("ideogram_offload", "model"), label="CPU offload")
            attention_backend = gr.Dropdown(ATTENTION_BACKEND_CHOICES, value=_option("ideogram_attention_backend", "automatic"), label="Attention backend")
            precision = gr.Radio(["automatic", "bfloat16", "float16"], value=_option("ideogram_precision", "automatic"), label="Precision")
            auto_vae_tiling = gr.Checkbox(value=_auto_vae_tiling_value(), label="Automatic VAE tiling")
            tile_vae = gr.Checkbox(value=_option("ideogram_tile_vae", True), label="Force VAE tiling", interactive=not _auto_vae_tiling_value())
            prompt_cache = gr.Checkbox(value=_option("ideogram_prompt_cache", True), label="Prompt cache")
            compile_enabled = gr.Checkbox(value=_option("ideogram_compile", False), label="Compile denoiser")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=5)
            with gr.Column(scale=1, min_width=180):
                generate_button = gr.Button("Generate", variant="primary")
                with gr.Row():
                    interrupt_button = gr.Button("Interrupt", variant="stop")
                    unload_button = gr.Button("Unload", variant="secondary")
                status = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    width = gr.Number(value=_option("ideogram_t2i_width", 1024), minimum=64, maximum=2048, step=16, label="Width")
                    height = gr.Number(value=_option("ideogram_t2i_height", 1024), minimum=64, maximum=2048, step=16, label="Height")
                with gr.Row():
                    num_images = gr.Number(value=_option("ideogram_t2i_batch_size", 1), minimum=1, maximum=8, step=1, label="Batch size")
                    steps = gr.Number(value=_option("ideogram_t2i_step", 48), minimum=1, maximum=150, step=1, label="Sampling steps")
                with gr.Row():
                    guidance = gr.Number(value=_option("ideogram_t2i_cfg", 7.0), minimum=0, maximum=20, step=0.1, label="CFG scale")
                    seed = gr.Number(value=_option("ideogram_seed", -1), minimum=-1, maximum=2**31 - 1, step=1, label="Seed")

                with gr.Accordion("Ideogram options", open=False):
                    with gr.Row():
                        mu = gr.Number(value=_option("ideogram_mu", 0.0), minimum=-10, maximum=10, step=0.1, label="Schedule mu")
                        std = gr.Number(value=_option("ideogram_std", 1.5), minimum=0.01, maximum=10, step=0.1, label="Schedule std")
                    with gr.Row():
                        prompt_upsampling = gr.Checkbox(value=_option("ideogram_prompt_upsampling", False), label="Prompt upsampling")
                        prompt_temperature = gr.Number(value=_option("ideogram_prompt_temperature", 1.0), minimum=0.1, maximum=2, step=0.1, label="Upsampling temperature")
                    polish_guidance = gr.Checkbox(value=_option("ideogram_polish_guidance", True), label="Final CFG polishing", info="Use up to three final steps capped at CFG 3; disable for constant CFG.")
                    recipe = gr.Dropdown(choices=list(native_recipes.IDEOGRAM_MODES), value=_option("ideogram_recipe", "Custom"), label="Sampler recipe", info="Named recipes override steps, CFG, mu and std.")
                    custom_schedule = gr.Textbox(value=_option("ideogram_custom_schedule", ""), label="Custom per-step CFG", info="Comma-separated in sampling order; leave empty for scalar/polishing.")
                    quantization = gr.Dropdown(choices=list(native_recipes.QUANTIZATION_MODES), value=_option("ideogram_quantization", "Checkpoint"), label="Native quantization", info="NF4 on-load is experimental; prefer a compatible prequantized checkpoint on low VRAM.")
                    enhancer_path = gr.Textbox(value=_option("ideogram_enhancer_path", ""), label="Local prompt enhancer head directory", info="Optional; requires outlines for structured captions.")
            with gr.Column(scale=2):
                output = gr.Gallery(label="Output", columns=2, height=512, object_fit="contain", type="pil")

        _preset_components = [
            model_path,
            offload,
            attention_backend,
            precision,
            auto_vae_tiling,
            tile_vae,
            prompt_cache,
            compile_enabled,
            width,
            height,
            num_images,
            steps,
            guidance,
            mu,
            std,
            prompt_upsampling,
            prompt_temperature,
            seed,
            polish_guidance,
            recipe,
            custom_schedule,
            quantization,
            enhancer_path,
        ]

        settings_to_save = {
            model_path: "ideogram_model_path",
            offload: "ideogram_offload",
            attention_backend: "ideogram_attention_backend",
            precision: "ideogram_precision",
            auto_vae_tiling: "ideogram_auto_vae_tiling",
            tile_vae: "ideogram_tile_vae",
            prompt_cache: "ideogram_prompt_cache",
            compile_enabled: "ideogram_compile",
            width: "ideogram_t2i_width",
            height: "ideogram_t2i_height",
            num_images: "ideogram_t2i_batch_size",
            steps: "ideogram_t2i_step",
            guidance: "ideogram_t2i_cfg",
            mu: "ideogram_mu",
            std: "ideogram_std",
            prompt_upsampling: "ideogram_prompt_upsampling",
            prompt_temperature: "ideogram_prompt_temperature",
            seed: "ideogram_seed",
            polish_guidance: "ideogram_polish_guidance",
            recipe: "ideogram_recipe",
            custom_schedule: "ideogram_custom_schedule",
            quantization: "ideogram_quantization",
            enhancer_path: "ideogram_enhancer_path",
        }
        for component, key in settings_to_save.items():
            if component is model_path:
                continue  # The global checkpoint selector owns model persistence.
            component.input(fn=lambda value, key=key: _save_option(key, value), inputs=[component], outputs=[], queue=False, show_progress=False)

        inputs = [
            model_path,
            offload,
            attention_backend,
            precision,
            auto_vae_tiling,
            tile_vae,
            prompt_cache,
            compile_enabled,
            prompt,
            width,
            height,
            num_images,
            steps,
            guidance,
            mu,
            std,
            prompt_upsampling,
            prompt_temperature,
            seed,
            polish_guidance,
            recipe,
            custom_schedule,
            quantization,
            enhancer_path,
        ]
        from modules_forge.native_requests import PageJobOwner
        from uuid import uuid4

        owner = PageJobOwner()
        session = gr.State(value=lambda: uuid4().hex)

        def run_page(session_id, *args):
            # Resolve random seeds once, before the worker's optional OOM retry.
            import inspect
            bound = inspect.signature(generate).bind(*args)
            if shared.opts.forge_preset != "ideogram":
                return gr.skip(), "The active preset changed; select ideogram before generating."
            # Do not trust an asynchronously updated hidden checkpoint mirror.
            bound.arguments["model_path"] = _model_path()
            bound.arguments["seed"] = native_validation.actual_seed(bound.arguments["seed"])
            with owner.active(session_id):
                result = main_thread.run_and_wait_result(generate, *bound.args, **bound.kwargs)
                if result is None:
                    raise RuntimeError("Native generation failed")
                output_value, message = result
                return (output_value if output_value else gr.skip()), message

        generate_ideogram = wrap_gradio_gpu_call(run_page, extra_outputs=[gr.skip()])
        generate_button.click(generate_ideogram, inputs=[session, *inputs], outputs=[output, status])
        prompt.submit(generate_ideogram, inputs=[session, *inputs], outputs=[output, status])
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
