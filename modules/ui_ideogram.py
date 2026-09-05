import gc
import logging
import threading

import gradio as gr
import torch

from backend import memory_management
from modules import shared
from modules_forge import main_thread
from modules_forge.native_models import is_compatible_model, resolve_model_path
from modules_forge.native_pipeline_speed import (
    ATTENTION_BACKEND_CHOICES,
    OFFLOAD_CHOICES,
    configure_pipeline,
    configure_prompt_cache,
    configure_vae_tiling,
    load_native_pipeline,
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
    for value in (_option("forge_checkpoint_ideogram", ""), _option("ideogram_model_path", "")):
        if is_compatible_model(value, "Ideogram4"):
            return value
    return ""


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
        _preset_components[0].change(
            fn=_sync_model_to_quicksettings,
            inputs=[preset_component, _preset_components[0]],
            outputs=[checkpoint_component],
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


def _get_pipeline(model_path: str, precision: str, offload: str, attention_backend: str, compile_enabled: bool, prompt_cache: bool):
    global _pipeline, _pipeline_key

    from diffusers import Ideogram4Pipeline

    model_path = resolve_model_path(model_path)
    device = _device()
    dtype = _default_dtype(precision)
    key = (model_path, str(dtype), str(device), offload, attention_backend, bool(compile_enabled), bool(prompt_cache))

    with _pipeline_lock:
        if _pipeline is not None and _pipeline_key == key:
            return _pipeline

        unload()
        unload_forge_model()
        pipeline_kwargs = {"torch_dtype": dtype}
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
    progress=gr.Progress(track_tqdm=True),
):
    if not model_path or not model_path.strip():
        return [], "Select a compatible Ideogram 4 Diffusers model from the global checkpoint dropdown."
    if not prompt or not prompt.strip():
        return [], "Enter a prompt."

    try:
        width, height = int(width), int(height)
        num_images, steps = int(num_images), int(steps)
        if width < 64 or height < 64 or width % 16 or height % 16:
            return [], "Ideogram 4 width and height must be at least 64 and divisible by 16."
        if not 1 <= num_images <= 8:
            return [], "Ideogram 4 supports 1 to 8 images per prompt in this UI."
        if not 1 <= steps <= 150:
            return [], "Ideogram 4 steps must be between 1 and 150."

        pipe = _get_pipeline(model_path, precision, offload, attention_backend, compile_enabled, prompt_cache)
        configure_vae_tiling(
            pipe.vae,
            resolve_vae_tiling_mode(auto_vae_tiling, tile_vae),
            _device(),
            width,
            height,
            1,
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

        generator = None
        if int(seed) >= 0:
            generator = torch.Generator(device=_device()).manual_seed(int(seed))

        def on_step_end(_pipe, step, _timestep, callback_kwargs):
            if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
                raise _IdeogramInterrupted()

            shared.state.sampling_step = step + 1
            shared.state.textinfo = f"Ideogram 4: step {step + 1}/{steps}"
            progress(step + 1, total=steps, desc=f"Ideogram 4 step {step + 1}/{steps}")
            return callback_kwargs

        with _pipeline_lock:
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=float(guidance),
                guidance_schedule=None,
                mu=float(mu),
                std=float(std),
                prompt_upsampling=bool(prompt_upsampling),
                prompt_upsampling_temperature=float(prompt_temperature),
                num_images_per_prompt=num_images,
                generator=generator,
                output_type="pil",
                return_dict=True,
                callback_on_step_end=on_step_end,
            )

        images = list(result.images if hasattr(result, "images") else result[0])
        shared.state.current_image = images[0] if images else None
        shared.state.sampling_step = steps
        shared.state.textinfo = "Ideogram 4: finished"

        return images, f"Ideogram 4 generated {len(images)} image(s)."
    except Exception as error:
        if isinstance(error, _IdeogramInterrupted):
            shared.state.textinfo = "Ideogram 4: interrupted"
            return [], "Ideogram 4 generation interrupted."
        if memory_management.is_oom(error):
            logger.exception("Ideogram 4 generation ran out of memory")
            unload()
            if getattr(shared.opts, "forge_oom_retry_enabled", True):
                raise
            shared.state.textinfo = "Ideogram 4: out of memory"
            return [], "Ideogram 4 ran out of memory. Reduce resolution, image count, or enable sequential CPU offload."
        logger.exception("Ideogram 4 generation failed")
        shared.state.textinfo = "Ideogram 4: failed"
        return [], f"Ideogram 4 error: {error}"


def generate_from_preset(prompt: str, *, width=None, height=None, num_images=None, steps=None, guidance=None):
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
        _option("ideogram_seed", -1),
    )


def create_ui():
    global _preset_components

    from modules.call_queue import wrap_gradio_gpu_call

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
        }
        for component, key in settings_to_save.items():
            component.change(fn=lambda value, key=key: _save_option(key, value), inputs=[component], outputs=[], queue=False, show_progress=False)

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
        ]
        generate_ideogram = wrap_gradio_gpu_call(
            lambda *args, **kwargs: main_thread.run_and_wait_result(generate, *args, **kwargs),
            extra_outputs=[None],
        )
        generate_button.click(generate_ideogram, inputs=inputs, outputs=[output, status])
        prompt.submit(generate_ideogram, inputs=inputs, outputs=[output, status])
        auto_vae_tiling.change(
            fn=lambda enabled: gr.update(interactive=not enabled),
            inputs=[auto_vae_tiling],
            outputs=[tile_vae],
            queue=False,
            show_progress=False,
        )
        interrupt_button.click(interrupt_generation, outputs=[])
        unload_button.click(unload, outputs=[])

    return interface
