import gc
import logging
import os
import threading
import uuid

import gradio as gr
import torch

from backend import memory_management
from modules import shared


logger = logging.getLogger("ltx2_video")
_pipeline = None
_pipeline_key = None
_pipeline_lock = threading.RLock()
_ltx_components = []
_preset_bound = False
LTX2_MODEL_DEFAULT = "diffusers/LTX-2.3-Diffusers"


class _LTXInterrupted(Exception):
    pass


def _option(name: str, default):
    return getattr(shared.opts, name, default)


def _model_path():
    configured = _option("ltx2_model_path", LTX2_MODEL_DEFAULT)
    selected = _option("forge_checkpoint_ltx2", None)
    if selected and selected != LTX2_MODEL_DEFAULT:
        return selected
    return configured or selected or LTX2_MODEL_DEFAULT


def _save_option(name: str, value):
    shared.opts.set(name, value)
    if name == "ltx2_model_path":
        shared.opts.set("forge_checkpoint_ltx2", value)
    shared.opts.save(shared.config_filename)


def _sync_preset_controls(preset: str):
    if preset != "ltx2":
        return [gr.skip() for _ in _ltx_components]

    return [
        _model_path(),
        _option("ltx2_prunavaed_path", ""),
        _option("ltx2_offload", "model"),
        _option("ltx2_precision", "automatic"),
        _option("ltx2_tile_vae", True),
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
    ]


def bind_preset(preset_component):
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
    from diffusers import AutoencoderKLLTX2Video

    last_error = None
    for subfolder in ("vae", None):
        try:
            kwargs = {"torch_dtype": dtype}
            if subfolder is not None:
                kwargs["subfolder"] = subfolder
            return AutoencoderKLLTX2Video.from_pretrained(path, **kwargs)
        except (OSError, ValueError, RuntimeError) as error:
            last_error = error

    raise RuntimeError(f"Could not load PrunaVAED from {path}: {last_error}")


def _configure_pipeline(pipe, device: torch.device, offload: str, tile_vae: bool):
    if tile_vae and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    if device.type != "cuda":
        pipe.to(device)
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    elif offload == "model":
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)


def _get_pipeline(model_path: str, prunavaed_path: str, precision: str, offload: str, tile_vae: bool, image_mode: bool):
    global _pipeline, _pipeline_key

    from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline

    model_path = str(model_path or "").strip()
    prunavaed_path = str(prunavaed_path or "").strip()
    device = _device()
    dtype = _default_dtype(precision)
    key = (model_path, prunavaed_path, str(dtype), str(device), offload, bool(tile_vae), bool(image_mode))

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
        if vae is not None:
            pipeline_kwargs["vae"] = vae

        pipeline_class = LTX2ImageToVideoPipeline if image_mode else LTX2Pipeline
        _pipeline = pipeline_class.from_pretrained(model_path, **pipeline_kwargs)
        _configure_pipeline(_pipeline, device, offload, tile_vae)
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
        memory_management.soft_empty_cache()


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
    precision: str,
    tile_vae: bool,
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
    progress=gr.Progress(track_tqdm=True),
):
    if not model_path or not model_path.strip():
        return None, "Enter an LTX-2.3 Diffusers model path or Hugging Face repository ID."
    if not prompt or not prompt.strip():
        return None, "Enter a prompt."

    try:
        width, height, frames, steps = int(width), int(height), int(frames), int(steps)
        if width < 64 or height < 64 or width % 32 or height % 32:
            return None, "LTX-2.3 width and height must be at least 64 and divisible by 32."
        if frames < 9:
            return None, "LTX-2.3 requires at least 9 frames."

        pipe = _get_pipeline(model_path, prunavaed_path, precision, offload, tile_vae, image is not None)

        shared.state.sampling_steps = steps
        shared.state.sampling_step = 0
        shared.state.job_count = 1
        shared.state.job_no = 0
        shared.state.current_image = None
        shared.state.id_live_preview += 1
        shared.state.textinfo = "LTX-2.3: loading and preparing"

        generator = None
        if int(seed) >= 0:
            generator = torch.Generator(device=_device()).manual_seed(int(seed))

        def on_step_end(_pipe, step, timestep, callback_kwargs):
            if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
                raise _LTXInterrupted()

            shared.state.sampling_step = step + 1
            shared.state.textinfo = f"LTX-2.3: step {step + 1}/{int(steps)}"
            progress(step + 1, total=steps, desc=f"LTX-2.3 step {step + 1}/{steps}")
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
            audio_guidance_scale=float(audio_guidance),
            audio_stg_scale=float(audio_stg),
            audio_modality_scale=float(audio_modality),
            guidance_rescale=float(guidance_rescale),
            spatio_temporal_guidance_blocks=_parse_blocks(guidance_blocks),
            generator=generator,
            output_type="np",
            return_dict=False,
            callback_on_step_end=on_step_end,
        )

        if image is not None:
            kwargs["image"] = image

        with _pipeline_lock:
            video, audio = pipe(**kwargs)
        output_path = _output_path()

        from diffusers.utils import encode_video

        audio_tensor = None
        audio_rate = None
        if include_audio and audio is not None:
            audio_tensor = audio[0].float().cpu()
            audio_rate = getattr(pipe.vocoder.config, "output_sampling_rate", 24000)

        encode_video(video[0], fps=int(fps), audio=audio_tensor, audio_sample_rate=audio_rate, output_path=output_path)
        shared.state.sampling_step = steps
        shared.state.textinfo = "LTX-2.3: finished"
        return output_path, f"Saved: {output_path}"
    except Exception as error:
        if isinstance(error, _LTXInterrupted):
            shared.state.textinfo = "LTX-2.3: interrupted"
            return None, "LTX-2.3 generation interrupted."
        if memory_management.is_oom(error):
            logger.exception("LTX-2.3 generation ran out of memory")
            unload()
            shared.state.textinfo = "LTX-2.3: out of memory"
            return None, "LTX-2.3 ran out of memory. Reduce resolution, frames, or enable sequential CPU offload."
        logger.exception("LTX-2.3 generation failed")
        shared.state.textinfo = "LTX-2.3: failed"
        return None, f"LTX-2.3 error: {error}"


def generate_from_preset(prompt: str, negative_prompt: str, image=None, *, width=None, height=None, frames=None, steps=None, guidance=None):
    """Run LTX with the values stored by the shared ``ltx2`` preset."""
    return generate(
        _model_path(),
        _option("ltx2_prunavaed_path", ""),
        _option("ltx2_offload", "model"),
        _option("ltx2_precision", "automatic"),
        _option("ltx2_tile_vae", True),
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
        _option("ltx2_seed", -1),
        _option("ltx2_include_audio", True),
    )


def create_ui():
    global _ltx_components

    from modules.call_queue import wrap_gradio_gpu_call

    with gr.Blocks(analytics_enabled=False) as interface:
        gr.Markdown(
            "## LTX-2.3 Video\n"
            "Use a Diffusers-format LTX-2.3 model for text-to-video or image-to-video. "
            "PrunaVAED is an optional LTX-2.3 decoder replacement."
        )

        with gr.Row():
            model_path = gr.Textbox(value=_model_path(), label="LTX-2.3 model path or Hugging Face ID", scale=3)
            prunavaed_path = gr.Textbox(value=_option("ltx2_prunavaed_path", ""), label="PrunaVAED path or Hugging Face ID (optional)", scale=2)

        with gr.Row():
            offload = gr.Radio(["none", "model", "sequential"], value=_option("ltx2_offload", "model"), label="CPU offload")
            precision = gr.Radio(["automatic", "bfloat16", "float16"], value=_option("ltx2_precision", "automatic"), label="Precision")
            tile_vae = gr.Checkbox(value=_option("ltx2_tile_vae", True), label="Tile VAE")
            interrupt_button = gr.Button("Interrupt", variant="stop")
            unload_button = gr.Button("Unload LTX-2.3", variant="secondary")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=5)
                negative_prompt = gr.Textbox(label="Negative prompt", lines=3)
                image = gr.Image(label="Image condition (optional)", type="pil")
            with gr.Column():
                video_output = gr.Video(label="Output", autoplay=True, include_audio=True)
                status = gr.Markdown()

        with gr.Row():
            width = gr.Number(value=_option("ltx2_t2i_width", 768), minimum=64, maximum=2048, step=32, label="Width")
            height = gr.Number(value=_option("ltx2_t2i_height", 512), minimum=64, maximum=2048, step=32, label="Height")
            frames = gr.Number(value=_option("ltx2_t2i_batch_size", 121), minimum=9, maximum=241, step=8, label="Frames")
            fps = gr.Number(value=_option("ltx2_fps", 24), minimum=1, maximum=60, step=1, label="FPS")

        with gr.Row():
            steps = gr.Number(value=_option("ltx2_t2i_step", 30), minimum=1, maximum=100, step=1, label="Steps")
            guidance = gr.Number(value=_option("ltx2_t2i_cfg", 3.0), minimum=0, maximum=20, step=0.1, label="Video CFG")
            stg = gr.Number(value=_option("ltx2_stg", 1.0), minimum=0, maximum=10, step=0.1, label="Video STG")
            modality = gr.Number(value=_option("ltx2_modality", 3.0), minimum=0, maximum=10, step=0.1, label="Video modality")

        with gr.Row():
            audio_guidance = gr.Number(value=_option("ltx2_audio_guidance", 7.0), minimum=0, maximum=20, step=0.1, label="Audio CFG")
            audio_stg = gr.Number(value=_option("ltx2_audio_stg", 1.0), minimum=0, maximum=10, step=0.1, label="Audio STG")
            audio_modality = gr.Number(value=_option("ltx2_audio_modality", 3.0), minimum=0, maximum=10, step=0.1, label="Audio modality")
            guidance_rescale = gr.Number(value=_option("ltx2_guidance_rescale", 0.7), minimum=0, maximum=1, step=0.05, label="Guidance rescale")

        with gr.Row():
            guidance_blocks = gr.Textbox(value=_option("ltx2_guidance_blocks", "28"), label="Spatio-temporal guidance blocks")
            seed = gr.Number(value=_option("ltx2_seed", -1), minimum=-1, maximum=2**31 - 1, step=1, label="Seed")
            include_audio = gr.Checkbox(value=_option("ltx2_include_audio", True), label="Include generated audio")
            generate_button = gr.Button("Generate video", variant="primary")

        _ltx_components = [
            model_path,
            prunavaed_path,
            offload,
            precision,
            tile_vae,
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
        ]

        settings_to_save = {
            model_path: "ltx2_model_path",
            prunavaed_path: "ltx2_prunavaed_path",
            offload: "ltx2_offload",
            precision: "ltx2_precision",
            tile_vae: "ltx2_tile_vae",
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
        }
        for component, key in settings_to_save.items():
            component.change(fn=lambda value, key=key: _save_option(key, value), inputs=[component], outputs=[], queue=False, show_progress=False)

        inputs = [
            model_path,
            prunavaed_path,
            offload,
            precision,
            tile_vae,
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
        ]
        generate_ltx = wrap_gradio_gpu_call(generate, extra_outputs=[None])
        generate_button.click(generate_ltx, inputs=inputs, outputs=[video_output, status])
        prompt.submit(generate_ltx, inputs=inputs, outputs=[video_output, status])
        interrupt_button.click(interrupt_generation, outputs=[])
        unload_button.click(unload, outputs=[])

    return interface
