import logging
import os.path
import sys

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
from modules_forge.presets import PresetArch, get_capabilities, is_video, use_distill, use_shift
from modules_forge.native_models import compatible_model_directories, is_compatible_model

logger = logging.getLogger("ui_models")
setup_logger(logger)

ui_forge_preset: gr.Radio
ui_checkpoint: gr.Dropdown
ui_vae: gr.Dropdown
ui_forge_unet_dtype: gr.Radio
ui_forge_vram_mode: gr.Dropdown
native_tabs: dict[str, gr.TabItem] = {}

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
IDEOGRAM_MODEL_DEFAULT = "ideogram-ai/ideogram-v4"
NATIVE_PIPELINE_PRESETS = {PresetArch.ltx2.name, PresetArch.ideogram.name}
NATIVE_TAB_PRESETS = {"ltx_video": PresetArch.ltx2.name, "ideogram": PresetArch.ideogram.name}
PRESET_TAB_IDS = {*NATIVE_TAB_PRESETS, "txt2img", "img2img", "img2img_batch"}


def register_native_tab(tab_id: str, tab: gr.TabItem):
    native_tabs[tab_id] = tab


def preset_tab_visible(tab_id: str, preset: str) -> bool:
    if tab_id == "txt2img":
        return get_capabilities(preset)["txt2img"]
    if tab_id == "img2img":
        return get_capabilities(preset)["img2img"]
    if tab_id == "img2img_batch":
        return get_capabilities(preset)["img2img_batch_tab"]
    expected_preset = NATIVE_TAB_PRESETS.get(tab_id)
    return expected_preset is None or expected_preset == preset


def native_model_marker(preset: str) -> str:
    return "LTX2" if preset == PresetArch.ltx2.name else "Ideogram4"


def native_model_directories(preset: str) -> list[str]:
    return compatible_model_directories(native_model_marker(preset))


def native_model_is_compatible(value: str, preset: str) -> bool:
    return is_compatible_model(value, native_model_marker(preset))


def module_choices() -> list[tuple[str, str]]:
    """Return sorted display/value pairs for the module selector."""
    category_order = {"VAE": 0, "Text Encoder": 1}

    return [
        (f"{category} / {name}", name)
        for name, category in sorted(
            ((name, module_categories.get(name, "VAE")) for name in module_list),
            key=lambda item: (category_order.get(item[1], 99), shared.natural_sort_key(item[0])),
        )
    ]


def valid_module_values(values, choices: list[tuple[str, str]]) -> list[str]:
    """Keep saved module selections valid after a refresh or preset switch."""
    valid = {value for _, value in choices}
    return [os.path.basename(value) for value in values or [] if os.path.basename(value) in valid]


def vram_mode_update():
    locked_mode = memory_management.commandline_vram_mode()
    mode = locked_mode or memory_management.normalize_vram_mode(getattr(shared.opts, "forge_vram_mode", "Automatic"))
    if locked_mode is None:
        memory_management.set_vram_mode(mode)
    return gr.update(value=mode, choices=memory_management.vram_mode_choices(), interactive=locked_mode is None)


def unload_for_vram_mode_change():
    """Unload Forge and native pipelines before changing placement policy."""
    from modules import sd_models

    try:
        sd_models.unload_model_weights()
    except Exception:
        logger.debug("Forge model cleanup during VRAM mode change failed", exc_info=True)

    for module_name in ("modules.ui_ltx2_video", "modules.ui_ideogram"):
        try:
            module = sys.modules.get(module_name)
            if module is not None:
                module.unload()
        except Exception:
            logger.debug("Native pipeline cleanup during VRAM mode change failed for %s", module_name, exc_info=True)

    memory_management.unload_all_models()
    memory_management.soft_empty_cache(force=True)
    processing.need_global_unload = True


def vram_mode_change(mode: str, save: bool = True) -> bool:
    locked_mode = memory_management.commandline_vram_mode()
    if locked_mode is not None:
        return False

    mode = memory_management.normalize_vram_mode(mode)
    if save and getattr(shared.opts, "forge_vram_mode", "Automatic") != mode:
        shared.opts.set("forge_vram_mode", mode)
        shared.opts.save(shared.config_filename)

    if memory_management.vram_mode_state(mode) is memory_management.vram_state:
        return True

    logger.info("Unloading models before applying VRAM mode: %s", mode)
    unload_for_vram_mode_change()
    memory_management.set_vram_mode(mode)
    return True


def make_checkpoint_manager_ui():
    global ui_forge_preset, ui_checkpoint, ui_vae, ui_forge_unet_dtype, ui_forge_vram_mode

    if shared.opts.sd_model_checkpoint in [None, "None", "none", ""]:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set("sd_model_checkpoint", next(iter(sd_models.checkpoints_list.values())).name)

    ckpt_list, vae_list = refresh_models()
    preset = getattr(shared.opts, "forge_preset", "sd")
    ckpt_list = checkpoint_choices(preset, ckpt_list)
    checkpoint_value = checkpoint_value_for_preset(preset, ckpt_list)
    module_choices_value = module_choices()
    module_value = valid_module_values(shared.opts.forge_additional_modules, module_choices_value)
    module_label = "VAE / Text Encoder"
    module_multiselect = True
    module_visible = preset not in NATIVE_PIPELINE_PRESETS

    if preset == PresetArch.ltx2.name:
        from modules import ui_ltx2_video

        module_label = "VAE / Decoder"
        module_choices_value = ui_ltx2_video._prunavaed_choices()
        module_value = ui_ltx2_video._prunavaed_value()
        module_multiselect = False

    ui_forge_preset = gr.Dropdown(
        label="UI Preset",
        value=shared.opts.forge_preset,
        choices=PresetArch.choices(),
        elem_id="forge_ui_preset",
    )

    ui_checkpoint = gr.Dropdown(
        label=checkpoint_label(preset),
        value=checkpoint_value,
        choices=ckpt_list,
        elem_id="setting_sd_model_checkpoint",
        elem_classes=["model_selection"],
        interactive=bool(ckpt_list),
    )

    ui_vae = gr.Dropdown(
        label=module_label,
        value=module_value,
        choices=module_choices_value,
        multiselect=module_multiselect,
        visible=module_visible,
        elem_id="setting_sd_modules",
        elem_classes=["model_selection"],
        interactive=bool(module_choices_value),
    )

    def refresh_model_list():
        ckpt_list, _ = refresh_models()
        current_preset = getattr(shared.opts, "forge_preset", "sd")
        choices = checkpoint_choices(current_preset, ckpt_list)
        return [
            gr.update(value=checkpoint_value_for_preset(current_preset, choices), choices=choices, label=checkpoint_label(current_preset), interactive=bool(choices)),
            module_dropdown_update(current_preset),
        ]

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id="forge_refresh_checkpoint", tooltip="Refresh")
    refresh_button.click(fn=refresh_model_list, outputs=[ui_checkpoint, ui_vae], queue=False)
    Context.root_block.load(fn=refresh_model_list, outputs=[ui_checkpoint, ui_vae], queue=False)

    ui_forge_unet_dtype = gr.Dropdown(
        label="Diffusion in Low Bits",
        value=None,
        choices=list(forge_unet_storage_dtype_options.keys()),
        visible=preset not in NATIVE_PIPELINE_PRESETS,
        elem_id="forge_ui_dtype",
    )

    initial_vram_mode = memory_management.commandline_vram_mode() or memory_management.normalize_vram_mode(getattr(shared.opts, "forge_vram_mode", "Automatic"))
    ui_forge_vram_mode = gr.Dropdown(
        label="VRAM Mode",
        value=initial_vram_mode,
        choices=memory_management.vram_mode_choices(),
        interactive=memory_management.commandline_vram_mode() is None,
        elem_id="forge_vram_mode",
    )
    if memory_management.commandline_vram_mode() is None:
        memory_management.set_vram_mode(initial_vram_mode)

    ui_checkpoint.input(checkpoint_change, inputs=[ui_checkpoint, ui_forge_preset], queue=False, show_progress=False)
    ui_vae.input(modules_change, inputs=[ui_vae, ui_forge_preset], queue=False, show_progress=False)
    ui_forge_unet_dtype.input(dtype_change, inputs=[ui_forge_unet_dtype, ui_forge_preset], queue=False, show_progress=False)
    ui_forge_vram_mode.change(vram_mode_change, inputs=[ui_forge_vram_mode], queue=False, show_progress=False)


def checkpoint_label(preset: str) -> str:
    return "Checkpoint"


def module_dropdown_update(preset: str):
    if preset == PresetArch.ltx2.name:
        from modules import ui_ltx2_video

        return gr.update(
            value=ui_ltx2_video._prunavaed_value(),
            choices=ui_ltx2_video._prunavaed_choices(),
            label="VAE / Decoder",
            multiselect=False,
            visible=True,
            interactive=bool(ui_ltx2_video._prunavaed_choices()),
        )

    choices = module_choices()
    value = [] if preset in NATIVE_PIPELINE_PRESETS else valid_module_values(
        getattr(shared.opts, f"forge_additional_modules_{preset}", []), choices
    )
    return gr.update(
        value=value,
        choices=choices,
        label="VAE / Text Encoder",
        multiselect=True,
        visible=preset not in NATIVE_PIPELINE_PRESETS,
        interactive=bool(choices) and preset not in NATIVE_PIPELINE_PRESETS,
    )


def ltx2_model_value() -> str:
    return native_model_value(PresetArch.ltx2.name)


def ideogram_model_value() -> str:
    return native_model_value(PresetArch.ideogram.name)


def native_model_value(preset: str) -> str:
    default = LTX2_MODEL_DEFAULT if preset == PresetArch.ltx2.name else IDEOGRAM_MODEL_DEFAULT
    for key in (f"forge_checkpoint_{preset}", f"{preset}_model_path"):
        value = str(getattr(shared.opts, key, "") or "").strip()
        if value and value != default and native_model_is_compatible(value, preset):
            return value
    return ""


def checkpoint_choices(preset: str, choices: list[str]) -> list[str]:
    if preset not in NATIVE_PIPELINE_PRESETS:
        return choices

    model_value = ltx2_model_value() if preset == PresetArch.ltx2.name else ideogram_model_value()
    values = []
    for value in (model_value, *native_model_directories(preset)):
        if value and value not in values:
            values.append(value)
    return values


def checkpoint_value_for_preset(preset: str, choices: list[str]) -> str | None:
    choices = [str(choice) for choice in choices if choice not in (None, "")]
    if preset in NATIVE_PIPELINE_PRESETS:
        value = ltx2_model_value() if preset == PresetArch.ltx2.name else ideogram_model_value()
        return value if value in choices else (choices[0] if choices else None)
    value = getattr(shared.opts, f"forge_checkpoint_{preset}", None) or shared.opts.sd_model_checkpoint
    value = str(value or "").strip()
    if value.lower() in {"none", "null"}:
        value = ""
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

    loading_parameters = dict(checkpoint_info=checkpoint_info, additional_modules=shared.opts.forge_additional_modules, unet_storage_dtype=unet_storage_dtype)

    ckpt: str = checkpoint_info.filename
    modules: list[str] = [os.path.basename(x) for x in shared.opts.forge_additional_modules]
    dtype = str(unet_storage_dtype or [torch.float16, torch.bfloat16])

    logger.info("Model Selected:")
    print_json(data=dict(checkpoint=os.path.basename(ckpt), modules=modules, dtype=dtype))

    if ckpt.endswith(("gguf", "GGUF")) and not lora_fp16:
        logger.warning("GGUF requires fp16 LoRA ; overriding option")
        lora_fp16 = True

    online_lora = lora_fp16
    parameters_changed = model_data.forge_loading_parameters != loading_parameters
    lora_mode_changed = dynamic_args.online_lora != online_lora
    model_data.forge_loading_parameters = loading_parameters
    dynamic_args.online_lora = online_lora
    if not parameters_changed and not lora_mode_changed:
        return

    logger.info(f"Patch LoRAs on-the-fly: {online_lora} (fp16={lora_fp16})")
    if not ckpt.endswith(("gguf", "GGUF")) and lora_fp16:
        logger.warning("on-the-fly WILL be slower ; enable only if you know what you are doing")

    processing.need_global_unload = True


def checkpoint_change(ckpt_name: str, preset: str, save=True, refresh=True) -> bool:
    """Save a checkpoint selection and return whether it changed."""
    if preset in NATIVE_PIPELINE_PRESETS:
        value_getter = ltx2_model_value if preset == PresetArch.ltx2.name else ideogram_model_value
        ckpt_name = str(ckpt_name or "").strip()
        current = value_getter()
        if ckpt_name == current:
            return False

        shared.opts.set(f"forge_checkpoint_{preset}", ckpt_name)
        shared.opts.set(f"{preset}_model_path", ckpt_name)
        if save:
            shared.opts.save(shared.config_filename)
        return True

    ckpt_name = str(ckpt_name or "").strip()
    if ckpt_name.lower() in {"none", "null"}:
        ckpt_name = ""
    if not ckpt_name:
        shared_items.refresh_checkpoints()
        choices = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)
        ckpt_name = checkpoint_value_for_preset(preset, choices) or ""
    if not ckpt_name:
        logger.error("No checkpoint is available for preset %s", preset)
        return False

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
    """Save module selections and return whether they changed."""
    if preset == PresetArch.ltx2.name:
        from modules import ui_ltx2_video

        value = module_values[0] if isinstance(module_values, list) and module_values else module_values
        value = value if value in ui_ltx2_video._prunavaed_choices() else ""
        ui_ltx2_video._save_option("ltx2_prunavaed_path", value)
        return True

    if preset in NATIVE_PIPELINE_PRESETS:
        return False

    modules = []
    if isinstance(module_values, str):
        module_values = [module_values]
    for v in module_values or []:
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
    if preset in NATIVE_PIPELINE_PRESETS:
        return False

    shared.opts.set("forge_unet_storage_dtype", dtype)
    if preset is not None:
        shared.opts.set(f"forge_unet_storage_dtype_{preset}", dtype)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def restore_standard_preset(preset: str):
    """Restore saved normal-model state instead of stale native-page inputs."""
    shared_items.refresh_checkpoints()
    choices = checkpoint_choices(
        preset,
        shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short),
    )
    checkpoint = str(getattr(shared.opts, f"forge_checkpoint_{preset}", "") or "").strip()
    if checkpoint not in choices:
        checkpoint = str(getattr(shared.opts, "sd_model_checkpoint", "") or "").strip()
    if checkpoint not in choices:
        checkpoint = choices[0] if choices else ""

    modules = getattr(shared.opts, f"forge_additional_modules_{preset}", [])
    modules = list(modules) if isinstance(modules, (list, tuple)) else []
    modules = [module_list[os.path.basename(module)] for module in modules if os.path.basename(module) in module_list]
    dtype = getattr(shared.opts, f"forge_unet_storage_dtype_{preset}", "Automatic")

    shared.opts.set("forge_additional_modules", modules)
    shared.opts.set(f"forge_additional_modules_{preset}", modules)
    shared.opts.set("forge_unet_storage_dtype", dtype)
    shared.opts.set(f"forge_unet_storage_dtype_{preset}", dtype)
    if checkpoint:
        shared.opts.set("sd_model_checkpoint", checkpoint)
        shared.opts.set(f"forge_checkpoint_{preset}", checkpoint)
    shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=bool(checkpoint))


def get_a1111_ui_component(tab: str, label: str) -> gr.components.Component:
    fields = infotext_utils.paste_fields[tab]["fields"]
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_steps = get_a1111_ui_component("txt2img", "Steps")
    ui_txt2img_hr_steps = get_a1111_ui_component("txt2img", "Hires steps")
    ui_img2img_steps = get_a1111_ui_component("img2img", "Steps")
    ui_txt2img_enable_hr = get_a1111_ui_component("txt2img", "enable_hr")
    ui_txt2img_negative_prompt = get_a1111_ui_component("txt2img", "negative_prompt")
    ui_txt2img_hr_negative_prompt = get_a1111_ui_component("txt2img", "Hires negative prompt")
    ui_img2img_negative_prompt = get_a1111_ui_component("img2img", "Negative prompt")
    ui_img2img_denoising_strength = get_a1111_ui_component("img2img", "Denoising strength")
    ui_img2img_image_cfg = get_a1111_ui_component("img2img", "Image CFG scale")

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
    ui_txt2img_batch_count = get_a1111_ui_component("txt2img", "Batch count")
    ui_img2img_batch_count = get_a1111_ui_component("img2img", "Batch count")

    for batch_count in (ui_txt2img_batch_count, ui_img2img_batch_count):
        if batch_count is not None:
            batch_count.minimum = 0
            batch_count.maximum = 128
            batch_count.step = 1
            batch_count.info = "Set to 0 for infinite generation."

    output_targets = [
        ui_checkpoint,
        ui_vae,
        ui_forge_unet_dtype,
        ui_forge_vram_mode,
        ui_txt2img_steps,
        ui_txt2img_hr_steps,
        ui_img2img_steps,
        ui_txt2img_enable_hr,
        ui_txt2img_negative_prompt,
        ui_txt2img_hr_negative_prompt,
        ui_img2img_negative_prompt,
        ui_img2img_denoising_strength,
        ui_img2img_image_cfg,
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
        ui_txt2img_batch_count,
        ui_img2img_batch_count,
    ]

    ui_forge_preset.change(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False).success(
        fn=_load_presets,
        inputs=[ui_checkpoint, ui_vae, ui_forge_unet_dtype, ui_forge_preset],
        queue=False,
        show_progress=False,
    ).then(js="clickLoraRefresh", fn=None, queue=False, show_progress=False)
    # A checkpoint can change capabilities without changing the preset, for
    # example Flux dev versus Flux Schnell.
    ui_checkpoint.change(on_preset_change, inputs=[ui_forge_preset, ui_checkpoint], outputs=output_targets, queue=False, show_progress=False)
    Context.root_block.load(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)

    for tab_id in PRESET_TAB_IDS:
        tab = native_tabs.get(tab_id)
        if tab is None:
            continue
        ui_forge_preset.change(
            fn=lambda preset, tab_id=tab_id: gr.update(visible=preset_tab_visible(tab_id, preset)),
            inputs=[ui_forge_preset],
            outputs=[tab],
            queue=False,
            show_progress=False,
        )

    from modules import ui_ltx2_video
    from modules import ui_ideogram

    ui_ltx2_video.bind_preset(ui_forge_preset, ui_checkpoint, ui_vae)
    ui_ideogram.bind_preset(ui_forge_preset, ui_checkpoint)

    if getattr(shared.opts, "forge_preset", "sd") not in NATIVE_PIPELINE_PRESETS:
        refresh_model_loading_parameters()


def _load_presets(ui_checkpoint: str, ui_vae: list[str], ui_forge_unet_dtype: str, ui_forge_preset: str):
    from modules import ui_ideogram, ui_ltx2_video

    ui_ltx2_video.unload()
    ui_ideogram.unload()

    if ui_forge_preset in NATIVE_PIPELINE_PRESETS:
        try:
            from modules import sd_models

            sd_models.unload_model_weights()
        except Exception:
            logger.debug("No Forge image model needed unloading before native pipeline", exc_info=True)
        native_checkpoint = native_model_value(ui_forge_preset)
        if native_checkpoint:
            checkpoint_change(native_checkpoint, ui_forge_preset, save=True, refresh=False)
        else:
            shared.opts.set(f"forge_checkpoint_{ui_forge_preset}", "")
            shared.opts.set(f"{ui_forge_preset}_model_path", "")
            shared.opts.save(shared.config_filename)
        return

    restore_standard_preset(ui_forge_preset)


def on_preset_change(preset: str, checkpoint_override: str | None = None):
    assert preset is not None
    preset_changed = shared.opts.set("forge_preset", preset)
    if preset_changed:
        shared.opts.save(shared.config_filename)

    checkpoint_list = checkpoint_choices(preset, shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short))
    checkpoint = checkpoint_override or checkpoint_value_for_preset(preset, checkpoint_list)
    capabilities = get_capabilities(preset, checkpoint)
    flux_schnell = preset == PresetArch.flux.name and "schnell" in str(checkpoint).lower()

    if use_shift(preset):
        d_args = {"visible": getattr(shared.opts, f"{preset}_show_shift", True), "label": "Shift"}
    elif use_distill(preset) and not flux_schnell:
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

    native_pipeline = preset in NATIVE_PIPELINE_PRESETS
    ideogram = preset == PresetArch.ideogram.name
    t2i_cfg = getattr(shared.opts, f"{preset}_t2i_cfg", 1.0)
    i2i_cfg = getattr(shared.opts, f"{preset}_i2i_cfg", 1.0)
    hr_cfg = getattr(shared.opts, f"{preset}_t2i_hr_cfg", 1.0)

    def cfg_enabled(value):
        try:
            return float(value) > 1.0
        except (TypeError, ValueError):
            return False

    t2i_cfg = float(t2i_cfg) if capabilities["cfg"] else 1.0
    i2i_cfg = float(i2i_cfg) if capabilities["cfg"] else 1.0
    hr_cfg = float(hr_cfg) if capabilities["cfg"] else 1.0
    t2i_batch_size_args = {**batch_args_t2i, "visible": capabilities["txt2img_batch_size"], "interactive": capabilities["txt2img_batch_size"]}
    i2i_batch_size_args = {**batch_args_i2i, "visible": capabilities["img2img_batch_size"], "interactive": capabilities["img2img_batch_size"]}
    batch_count_args = {
        "minimum": 0,
        "maximum": 128,
        "step": 1,
        "label": "Batch Count",
        "info": "Set to 0 for infinite generation.",
    }
    t2i_batch_count_args = {**batch_count_args, "visible": capabilities["txt2img_batch_count"], "interactive": capabilities["txt2img_batch_count"]}
    i2i_batch_count_args = {**batch_count_args, "visible": capabilities["img2img_batch_count"], "interactive": capabilities["img2img_batch_count"]}
    if not capabilities["txt2img_batch_size"]:
        t2i_batch_size_args["value"] = 1
    if not capabilities["img2img_batch_size"]:
        i2i_batch_size_args["value"] = 1
    if not capabilities["txt2img_batch_count"]:
        t2i_batch_count_args["value"] = 1
    if not capabilities["img2img_batch_count"]:
        i2i_batch_count_args["value"] = 1

    return [
        # ui_checkpoint, ui_vae, ui_forge_unet_dtype, ui_forge_vram_mode
        gr.update(
            value=checkpoint_value_for_preset(preset, checkpoint_list),
            choices=checkpoint_list,
            label=checkpoint_label(preset),
            interactive=bool(checkpoint_list),
        ),
        module_dropdown_update(preset),
        gr.update(value=getattr(shared.opts, f"forge_unet_storage_dtype_{preset}", "Automatic"), visible=not native_pipeline, interactive=not native_pipeline),
        vram_mode_update(),
        # ui_txt2img_steps, ui_txt2img_hr_steps, ui_img2img_steps
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_step", 20)) > 0 else gr.skip(),
        gr.update(value=v, visible=not native_pipeline, interactive=not native_pipeline) if (v := getattr(shared.opts, f"{preset}_t2i_hr_step", 20)) > 0 else gr.skip(),
        gr.update(value=v, visible=not ideogram, interactive=not ideogram) if (v := getattr(shared.opts, f"{preset}_i2i_step", 20)) > 0 else gr.skip(),
        gr.update(visible=capabilities["hires"] and capabilities["txt2img"], interactive=capabilities["hires"] and capabilities["txt2img"]),
        gr.update(visible=capabilities["txt2img"], interactive=capabilities["negative_prompt"] and cfg_enabled(t2i_cfg)),
        gr.update(visible=capabilities["hires"] and capabilities["negative_prompt"], interactive=capabilities["negative_prompt"] and cfg_enabled(hr_cfg)),
        gr.update(visible=capabilities["img2img"], interactive=capabilities["negative_prompt"] and cfg_enabled(i2i_cfg)),
        gr.update(visible=capabilities["img2img"] and capabilities["img2img_denoising"], interactive=capabilities["img2img"] and capabilities["img2img_denoising"]),
        gr.update(visible=False),
        # ui_txt2img_sampler, ui_img2img_sampler, ui_txt2img_scheduler, ui_img2img_scheduler
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_sampler", "Euler"), visible=capabilities["txt2img"] and capabilities["standard_sampling"], interactive=capabilities["txt2img"] and capabilities["standard_sampling"]),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_sampler", "Euler"), visible=capabilities["img2img"] and not native_pipeline, interactive=capabilities["img2img"] and not native_pipeline),
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_scheduler", "Simple"), visible=capabilities["txt2img"] and capabilities["standard_sampling"], interactive=capabilities["txt2img"] and capabilities["standard_sampling"]),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_scheduler", "Simple"), visible=capabilities["img2img"] and not native_pipeline, interactive=capabilities["img2img"] and not native_pipeline),
        # ui_txt2img_width, ui_img2img_width, ui_txt2img_height, ui_img2img_height
        gr.update(value=v, visible=capabilities["txt2img"], interactive=capabilities["txt2img"]) if (v := getattr(shared.opts, f"{preset}_t2i_width", 1024)) > 0 else gr.skip(),
        gr.update(value=v, visible=capabilities["img2img"], interactive=capabilities["img2img"]) if (v := getattr(shared.opts, f"{preset}_i2i_width", 1024)) > 0 else gr.skip(),
        gr.update(value=v, visible=capabilities["txt2img"], interactive=capabilities["txt2img"]) if (v := getattr(shared.opts, f"{preset}_t2i_height", 1024)) > 0 else gr.skip(),
        gr.update(value=v, visible=capabilities["img2img"], interactive=capabilities["img2img"]) if (v := getattr(shared.opts, f"{preset}_i2i_height", 1024)) > 0 else gr.skip(),
        # ui_txt2img_cfg, ui_txt2img_hr_cfg, ui_img2img_cfg
        gr.update(value=t2i_cfg, visible=capabilities["txt2img"], interactive=capabilities["cfg"]),
        gr.update(value=hr_cfg, visible=capabilities["hires"] and capabilities["txt2img"], interactive=capabilities["cfg"]),
        gr.update(value=i2i_cfg, visible=capabilities["img2img"], interactive=capabilities["cfg"]),
        # ui_txt2img_distilled_cfg, ui_img2img_distilled_cfg, ui_txt2img_hr_distilled_cfg
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_dcfg", 3.0), **d_args),
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_hr_dcfg", 3.0), **d_args),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_dcfg", 3.0), **d_args),
        # ui_txt2img_batch_size, ui_img2img_batch_size
        gr.update(**t2i_batch_size_args),
        gr.update(**i2i_batch_size_args),
        gr.update(**t2i_batch_count_args),
        gr.update(**i2i_batch_count_args),
    ]
