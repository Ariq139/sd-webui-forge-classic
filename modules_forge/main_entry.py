import logging
import os.path

import gradio as gr
import torch
from gradio.context import Context
from rich import print_json

from backend import memory_management
from backend.args import dynamic_args
from backend.logging import setup_logger
from modules import (
    infotext_utils,
    paths,
    processing,
    sd_models,
    shared,
    shared_items,
    ui_common,
)
from modules_forge.presets import PresetArch, is_video, use_distill, use_shift

logger = logging.getLogger("ui_models")
setup_logger(logger)

ui_forge_preset: gr.Radio
ui_checkpoint: gr.Dropdown
ui_vae: gr.Dropdown
ui_forge_unet_dtype: gr.Radio

forge_unet_storage_dtype_options: dict[str, tuple[torch.dtype, bool]] = {
    "Automatic": (None, False),
    "Automatic (fp16 LoRA)": (None, True),
    "float8-e4m3fn": (torch.float8_e4m3fn, False),
    "float8-e4m3fn (fp16 LoRA)": (torch.float8_e4m3fn, True),
    "float8-e5m2": (torch.float8_e5m2, False),
    "float8-e5m2 (fp16 LoRA)": (torch.float8_e5m2, True),
}


module_list: dict[str, os.PathLike] = {}
module_categories: dict[str, str] = {}
LTX2_MODEL_DEFAULT = "diffusers/LTX-2.3-Diffusers"


def module_choices() -> list[tuple[str, str]]:
    """Return display/value pairs for the VAE and text-encoder selector.

    Gradio dropdowns do not provide optgroup support, so the category is kept
    in the display label while the value remains the original filename. This
    preserves the existing multi-select payload consumed by ``modules_change``.
    """
    category_order = {"VAE": 0, "Text Encoder": 1}

    return [
        (f"{category} / {name}", name)
        for name, category in sorted(
            ((name, module_categories.get(name, "VAE")) for name in module_list),
            key=lambda item: (category_order.get(item[1], 99), shared.natural_sort_key(item[0])),
        )
    ]


def make_checkpoint_manager_ui():
    global ui_forge_preset, ui_checkpoint, ui_vae, ui_forge_unet_dtype

    if shared.opts.sd_model_checkpoint in [None, "None", "none", ""]:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set("sd_model_checkpoint", next(iter(sd_models.checkpoints_list.values())).name)

    ckpt_list, vae_list = refresh_models()
    preset = getattr(shared.opts, "forge_preset", "sd")
    ckpt_list = checkpoint_choices(preset, ckpt_list)
    checkpoint_value = checkpoint_value_for_preset(preset, ckpt_list)
    module_value = [os.path.basename(x) for x in shared.opts.forge_additional_modules if os.path.basename(x) in vae_list]

    ui_forge_preset = gr.Dropdown(label="UI Preset", value=shared.opts.forge_preset, choices=PresetArch.choices(), elem_id="forge_ui_preset")

    ui_checkpoint = gr.Dropdown(label=checkpoint_label(preset), value=checkpoint_value, choices=ckpt_list, elem_id="setting_sd_model_checkpoint", elem_classes=["model_selection"])

    ui_vae = gr.Dropdown(label="VAE / Text Encoder", value=module_value, choices=module_choices(), multiselect=True, visible=preset != PresetArch.ltx2.name, elem_id="setting_sd_modules", elem_classes=["model_selection"])

    def refresh_model_list():
        ckpt_list, _ = refresh_models()
        current_preset = getattr(shared.opts, "forge_preset", "sd")
        choices = checkpoint_choices(current_preset, ckpt_list)
        return [gr.update(value=checkpoint_value_for_preset(current_preset, choices), choices=choices, label=checkpoint_label(current_preset)), gr.update(choices=module_choices())]

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id="forge_refresh_checkpoint", tooltip="Refresh")
    refresh_button.click(fn=refresh_model_list, outputs=[ui_checkpoint, ui_vae], queue=False)
    Context.root_block.load(fn=refresh_model_list, outputs=[ui_checkpoint, ui_vae], queue=False)

    ui_forge_unet_dtype = gr.Dropdown(label="Diffusion in Low Bits", value=None, choices=list(forge_unet_storage_dtype_options.keys()), visible=preset != PresetArch.ltx2.name, elem_id="forge_ui_dtype")

    ui_checkpoint.input(checkpoint_change, inputs=[ui_checkpoint, ui_forge_preset], queue=False, show_progress=False)
    ui_vae.input(modules_change, inputs=[ui_vae, ui_forge_preset], queue=False, show_progress=False)
    ui_forge_unet_dtype.input(dtype_change, inputs=[ui_forge_unet_dtype, ui_forge_preset], queue=False, show_progress=False)


def checkpoint_label(preset: str) -> str:
    return "LTX-2.3 Model" if preset == PresetArch.ltx2.name else "Checkpoint"


def ltx2_model_value() -> str:
    configured = getattr(shared.opts, "ltx2_model_path", None)
    selected = getattr(shared.opts, "forge_checkpoint_ltx2", None)
    if selected and selected != LTX2_MODEL_DEFAULT:
        return selected
    return configured or selected or LTX2_MODEL_DEFAULT


def checkpoint_choices(preset: str, choices: list[str]) -> list[str]:
    if preset != PresetArch.ltx2.name:
        return choices

    values = []
    for value in (ltx2_model_value(), LTX2_MODEL_DEFAULT):
        if value and value not in values:
            values.append(value)
    return values


def checkpoint_value_for_preset(preset: str, choices: list[str]) -> str | None:
    if preset == PresetArch.ltx2.name:
        value = ltx2_model_value()
        return value if value in choices else (choices[-1] if choices else LTX2_MODEL_DEFAULT)
    value = getattr(shared.opts, f"forge_checkpoint_{preset}", None) or shared.opts.sd_model_checkpoint
    return value if value in choices else (choices[0] if choices else None)


def find_files_with_extensions(base_path: os.PathLike, extensions: list[str]) -> dict[str, os.PathLike]:
    found_files = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                found_files[file] = full_path
    return found_files


def refresh_models() -> tuple[list[os.PathLike], list[os.PathLike]]:
    shared_items.refresh_checkpoints()
    ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)

    file_extensions = ("ckpt", "pt", "pth", "bin", "safetensors", "sft", "gguf")

    module_list.clear()
    module_categories.clear()

    module_paths = [
        ("VAE", [os.path.join(paths.models_path, "VAE"), *shared.cmd_opts.vae_dirs]),
        ("Text Encoder", [os.path.join(paths.models_path, "text_encoder"), *shared.cmd_opts.text_encoder_dirs]),
    ]
    visited_paths = set()

    for category, roots in module_paths:
        for root in roots:
            root = os.path.abspath(os.fspath(root))
            if root in visited_paths:
                continue
            visited_paths.add(root)

            module_files = find_files_with_extensions(root, file_extensions)
            module_list.update(module_files)
            module_categories.update({name: category for name in module_files})

    return sorted(ckpt_list, key=shared.natural_sort_key), sorted(module_list.keys(), key=shared.natural_sort_key)


def refresh_model_loading_parameters(*, refresh: bool = True):
    if not refresh:
        return

    from modules.sd_models import model_data, select_checkpoint

    checkpoint_info = select_checkpoint()
    if checkpoint_info is None:
        logger.critical('You do not have any model... Please download models to "models/Stable-diffusion"')
        return

    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))

    model_data.forge_loading_parameters = dict(checkpoint_info=checkpoint_info, additional_modules=shared.opts.forge_additional_modules, unet_storage_dtype=unet_storage_dtype)

    ckpt: str = checkpoint_info.filename
    modules: list[str] = [os.path.basename(x) for x in shared.opts.forge_additional_modules]
    dtype = str(unet_storage_dtype or [torch.float16, torch.bfloat16])

    logger.info("Model Selected:")
    print_json(data=dict(checkpoint=os.path.basename(ckpt), modules=modules, dtype=dtype))

    if ckpt.endswith(("gguf", "GGUF")) and not lora_fp16:
        logger.warning("GGUF requires fp16 LoRA ; overriding option")
        lora_fp16 = True

    dynamic_args.online_lora = lora_fp16
    logger.info(f"Patch LoRAs on-the-fly: {lora_fp16}")
    if not ckpt.endswith(("gguf", "GGUF")) and lora_fp16:
        logger.warning("on-the-fly WILL be slower ; enable only if you know what you are doing")

    processing.need_global_unload = True


def checkpoint_change(ckpt_name: str, preset: str, save=True, refresh=True) -> bool:
    """`ckpt_name` accepts valid aliases; returns `True` if checkpoint changed"""
    if preset == PresetArch.ltx2.name:
        ckpt_name = (ckpt_name or LTX2_MODEL_DEFAULT).strip()
        current = ltx2_model_value()
        if ckpt_name == current:
            return False

        shared.opts.set("forge_checkpoint_ltx2", ckpt_name)
        shared.opts.set("ltx2_model_path", ckpt_name)
        if save:
            shared.opts.save(shared.config_filename)
        return True

    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(getattr(shared.opts, "sd_model_checkpoint", ""))
    if new_ckpt_info == current_ckpt_info:
        return False

    shared.opts.set("sd_model_checkpoint", ckpt_name)
    if preset is not None:
        shared.opts.set(f"forge_checkpoint_{preset}", ckpt_name)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def modules_change(module_values: list, preset: str, save=True, refresh=True) -> bool:
    """`module_values` accepts file paths or just the module names; returns `True` if modules changed"""
    if preset == PresetArch.ltx2.name:
        return False

    modules = []
    for v in module_values:
        module_name = os.path.basename(v)  # If the input is a filepath, extract the filename
        if module_name in module_list:
            modules.append(module_list[module_name])
    modules.sort()

    # skip further processing if value unchanged
    if modules == getattr(shared.opts, "forge_additional_modules", []):
        return False

    shared.opts.set("forge_additional_modules", modules)
    if preset is not None:
        shared.opts.set(f"forge_additional_modules_{preset}", modules)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def dtype_change(dtype: str, preset: str, save=True, refresh=True) -> bool:
    if preset == PresetArch.ltx2.name:
        return False

    shared.opts.set("forge_unet_storage_dtype", dtype)
    if preset is not None:
        shared.opts.set(f"forge_unet_storage_dtype_{preset}", dtype)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def get_a1111_ui_component(tab: str, label: str) -> gr.components.Component:
    fields = infotext_utils.paste_fields[tab]["fields"]
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_steps = get_a1111_ui_component("txt2img", "Steps")
    ui_txt2img_hr_steps = get_a1111_ui_component("txt2img", "Hires steps")
    ui_img2img_steps = get_a1111_ui_component("img2img", "Steps")

    ui_txt2img_sampler = get_a1111_ui_component("txt2img", "sampler_name")
    ui_img2img_sampler = get_a1111_ui_component("img2img", "sampler_name")
    ui_txt2img_scheduler = get_a1111_ui_component("txt2img", "scheduler")
    ui_img2img_scheduler = get_a1111_ui_component("img2img", "scheduler")

    ui_txt2img_width = get_a1111_ui_component("txt2img", "Size-1")
    ui_img2img_width = get_a1111_ui_component("img2img", "Size-1")
    ui_txt2img_height = get_a1111_ui_component("txt2img", "Size-2")
    ui_img2img_height = get_a1111_ui_component("img2img", "Size-2")

    ui_txt2img_cfg = get_a1111_ui_component("txt2img", "CFG scale")
    ui_txt2img_hr_cfg = get_a1111_ui_component("txt2img", "Hires CFG Scale")
    ui_img2img_cfg = get_a1111_ui_component("img2img", "CFG scale")

    ui_txt2img_distilled_cfg = get_a1111_ui_component("txt2img", "Distilled CFG Scale")
    ui_txt2img_hr_distilled_cfg = get_a1111_ui_component("txt2img", "Hires Distilled CFG Scale")
    ui_img2img_distilled_cfg = get_a1111_ui_component("img2img", "Distilled CFG Scale")

    ui_txt2img_batch_size = get_a1111_ui_component("txt2img", "Batch size")
    ui_img2img_batch_size = get_a1111_ui_component("img2img", "Batch size")

    output_targets = [
        ui_checkpoint,
        ui_vae,
        ui_forge_unet_dtype,
        ui_txt2img_steps,
        ui_txt2img_hr_steps,
        ui_img2img_steps,
        ui_txt2img_sampler,
        ui_img2img_sampler,
        ui_txt2img_scheduler,
        ui_img2img_scheduler,
        ui_txt2img_width,
        ui_img2img_width,
        ui_txt2img_height,
        ui_img2img_height,
        ui_txt2img_cfg,
        ui_txt2img_hr_cfg,
        ui_img2img_cfg,
        ui_txt2img_distilled_cfg,
        ui_txt2img_hr_distilled_cfg,
        ui_img2img_distilled_cfg,
        ui_txt2img_batch_size,
        ui_img2img_batch_size,
    ]

    ui_forge_preset.change(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False).success(
        fn=_load_presets,
        inputs=[ui_checkpoint, ui_vae, ui_forge_unet_dtype, ui_forge_preset],
        queue=False,
        show_progress=False,
    ).then(js="clickLoraRefresh", fn=None, queue=False, show_progress=False)
    Context.root_block.load(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)

    from modules import ui_ltx2_video

    ui_ltx2_video.bind_preset(ui_forge_preset)

    if getattr(shared.opts, "forge_preset", "sd") != PresetArch.ltx2.name:
        refresh_model_loading_parameters()


def _load_presets(ui_checkpoint: str, ui_vae: list[str], ui_forge_unet_dtype: str, ui_forge_preset: str):
    if ui_forge_preset == PresetArch.ltx2.name:
        try:
            from modules import sd_models

            sd_models.unload_model_weights()
        except Exception:
            logger.debug("No Forge image model needed unloading before LTX-2.3", exc_info=True)
        checkpoint_change(ui_checkpoint, ui_forge_preset, save=True, refresh=False)
        return

    from modules import ui_ltx2_video

    ui_ltx2_video.unload()
    dtype_change(ui_forge_unet_dtype, ui_forge_preset, save=False, refresh=False)
    modules_change(ui_vae, ui_forge_preset, save=False, refresh=False)
    checkpoint_change(ui_checkpoint, ui_forge_preset, save=True, refresh=True)


def on_preset_change(preset: str):
    assert preset is not None
    shared.opts.set("forge_preset", preset)
    shared.opts.save(shared.config_filename)

    if use_shift(preset):
        d_args = {"visible": getattr(shared.opts, f"{preset}_show_shift", True), "label": "Shift"}
    elif use_distill(preset):
        d_args = {"visible": True, "label": "Distilled CFG Scale"}
    else:
        d_args = {"visible": False}

    if (fps := is_video(preset)) > 1:
        max_frames = fps * (30 if preset == PresetArch.ltx2.name else 15) + 1
        min_frames = 9 if preset == PresetArch.ltx2.name else 1
        batch_args_t2i = {"minimum": min_frames, "maximum": max_frames, "step": fps, "label": "Frames", "value": getattr(shared.opts, f"{preset}_t2i_batch_size", 1)}
    else:
        batch_args_t2i = {"minimum": 1, "maximum": 8, "step": 1, "label": "Batch Size", "value": getattr(shared.opts, f"{preset}_t2i_batch_size", 1)}

    batch_args_i2i = batch_args_t2i.copy()
    batch_args_i2i["value"] = getattr(shared.opts, f"{preset}_i2i_batch_size", 1)

    ltx2 = preset == PresetArch.ltx2.name
    checkpoint_list = checkpoint_choices(preset, shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short))

    return [
        # ui_checkpoint, ui_vae, ui_forge_unet_dtype
        gr.update(
            value=checkpoint_value_for_preset(preset, checkpoint_list),
            choices=checkpoint_list,
            label=checkpoint_label(preset),
        ),
        gr.update(
            value=[] if ltx2 else [os.path.basename(m) for m in getattr(shared.opts, f"forge_additional_modules_{preset}", [])],
            visible=not ltx2,
            interactive=not ltx2,
        ),
        gr.update(value=getattr(shared.opts, f"forge_unet_storage_dtype_{preset}", "Automatic"), visible=not ltx2, interactive=not ltx2),
        # ui_txt2img_steps, ui_txt2img_hr_steps, ui_img2img_steps
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_step", 20)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_hr_step", 20)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_step", 20)) > 0 else gr.skip(),
        # ui_txt2img_sampler, ui_img2img_sampler, ui_txt2img_scheduler, ui_img2img_scheduler
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_sampler", "Euler")),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_sampler", "Euler")),
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_scheduler", "Simple")),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_scheduler", "Simple")),
        # ui_txt2img_width, ui_img2img_width, ui_txt2img_height, ui_img2img_height
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_width", 1024)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_width", 1024)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_height", 1024)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_height", 1024)) > 0 else gr.skip(),
        # ui_txt2img_cfg, ui_txt2img_hr_cfg, ui_img2img_cfg
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_cfg", 1.0)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_hr_cfg", 1.0)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_cfg", 1.0)) > 0 else gr.skip(),
        # ui_txt2img_distilled_cfg, ui_img2img_distilled_cfg, ui_txt2img_hr_distilled_cfg
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_dcfg", 3.0), **d_args),
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_hr_dcfg", 3.0), **d_args),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_dcfg", 3.0), **d_args),
        # ui_txt2img_batch_size, ui_img2img_batch_size
        gr.update(**batch_args_t2i),
        gr.update(**batch_args_i2i),
    ]
