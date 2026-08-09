from enum import Enum


class PresetArch(Enum):
    sd = 1  # SD1
    xl = 2  # SDXL
    flux = 3  # Flux.1
    klein = 4  # Flux.2
    qwen = 5  # Qwen-Image
    lumina = 6  # Lumina-Image-2.0
    zit = 7  # Z-Image-Turbo
    wan = 8  # Wan2.2
    anima = 9  # Anima
    ernie = 10  # Ernie-Image
    pid = 11  # PiD
    krea = 12  # Krea2
    ltx2 = 13  # LTX-2.3 video

    @staticmethod
    def choices() -> list[str]:
        return [preset.name for preset in PresetArch]


SAMPLERS = {
    PresetArch.sd: "Euler a",
    PresetArch.xl: "Euler a",
    PresetArch.flux: "Euler",
    PresetArch.klein: "Euler",
    PresetArch.qwen: "LCM",
    PresetArch.lumina: "Res Multistep",
    PresetArch.zit: "Euler",
    PresetArch.wan: "Euler",
    PresetArch.anima: "ER SDE",
    PresetArch.ernie: "Euler",
    PresetArch.pid: "LCM",
    PresetArch.krea: "Euler",
    PresetArch.ltx2: "Euler",
}

SCHEDULERS = {
    PresetArch.sd: "Automatic",
    PresetArch.xl: "Automatic",
    PresetArch.flux: "Beta",
    PresetArch.klein: "Beta",
    PresetArch.qwen: "Normal",
    PresetArch.lumina: "Simple",
    PresetArch.zit: "Beta",
    PresetArch.wan: "Simple",
    PresetArch.anima: "Beta",
    PresetArch.ernie: "Simple",
    PresetArch.pid: "Simple",
    PresetArch.krea: "Simple",
    PresetArch.ltx2: "Simple",
}

STEPS = {
    PresetArch.sd: 32,
    PresetArch.xl: 24,
    PresetArch.flux: 20,
    PresetArch.klein: 4,
    PresetArch.qwen: 8,
    PresetArch.lumina: 32,
    PresetArch.zit: 9,
    PresetArch.wan: 4,
    PresetArch.anima: 32,
    PresetArch.ernie: 8,
    PresetArch.pid: 4,
    PresetArch.krea: 8,
    PresetArch.ltx2: 30,
}

CFG = {
    PresetArch.sd: 6.0,
    PresetArch.xl: 4.5,
    PresetArch.flux: 1.0,
    PresetArch.klein: 1.0,
    PresetArch.qwen: 1.0,
    PresetArch.lumina: 4.0,
    PresetArch.zit: 1.0,
    PresetArch.wan: 1.0,
    PresetArch.anima: 4.0,
    PresetArch.ernie: 1.0,
    PresetArch.pid: 1.0,
    PresetArch.krea: 1.0,
    PresetArch.ltx2: 3.0,
}

DISTILL = {
    PresetArch.flux: 3.0,
}

SHIFT = {
    PresetArch.xl: -9.0,
    PresetArch.lumina: 6.0,
    PresetArch.zit: 9.0,
    PresetArch.wan: 5.0,
    PresetArch.anima: 3.0,
    PresetArch.ernie: 3.0,
    PresetArch.pid: -1.5,
    PresetArch.krea: -1.15,
}

FRAMES = {
    PresetArch.wan.name: 16,
    PresetArch.ltx2.name: 8,
}


def use_distill(arch: str) -> bool:
    return arch in [preset.name for preset in DISTILL.keys()]


def use_shift(arch: str) -> bool:
    return arch in [preset.name for preset in SHIFT.keys()]


def is_video(arch: str) -> int:
    return FRAMES.get(arch, 1)


def register(options_templates: dict):
    from gradio import Dropdown, Slider

    from modules.options import OptionInfo, OptionRow, options_section
    from modules.shared_items import list_samplers, list_schedulers

    for arch in PresetArch:
        name = arch.name

        checkpoint_default = "diffusers/LTX-2.3-Diffusers" if arch is PresetArch.ltx2 else None

        options_templates.update(
            options_section(
                (None, "Forge Hidden Options"),
                {
                    f"forge_checkpoint_{name}": OptionInfo(checkpoint_default),
                    f"forge_additional_modules_{name}": OptionInfo([]),
                    f"forge_unet_storage_dtype_{name}": OptionInfo("Automatic"),
                },
            )
        )

        sampler, scheduler = SAMPLERS[arch], SCHEDULERS[arch]

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_t2i_ss1": OptionRow(),
                    f"{name}_t2i_sampler": OptionInfo(sampler, "txt2img Sampler", Dropdown, lambda: {"choices": [x.name for x in list_samplers()]}),
                    f"{name}_t2i_scheduler": OptionInfo(scheduler, "txt2img Scheduler", Dropdown, lambda: {"choices": list_schedulers()}),
                    f"{name}_t2i_ss0": OptionRow(),
                    f"{name}_i2i_ss1": OptionRow(),
                    f"{name}_i2i_sampler": OptionInfo(sampler, "img2img Sampler", Dropdown, lambda: {"choices": [x.name for x in list_samplers()]}),
                    f"{name}_i2i_scheduler": OptionInfo(scheduler, "img2img Scheduler", Dropdown, lambda: {"choices": list_schedulers()}),
                    f"{name}_i2i_ss0": OptionRow(),
                },
            )
        )

        step = STEPS[arch]

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_steps1": OptionRow(),
                    f"{name}_t2i_step": OptionInfo(step, "txt2img Steps", Slider, {"minimum": 0, "maximum": 150, "step": 1}),
                    f"{name}_t2i_hr_step": OptionInfo(step, "txt2img Hires. Steps", Slider, {"minimum": 0, "maximum": 150, "step": 1}),
                    f"{name}_i2i_step": OptionInfo(step, "img2img Steps", Slider, {"minimum": 0, "maximum": 150, "step": 1}),
                    f"{name}_steps0": OptionRow(),
                },
            )
        )

        cfg = CFG[arch]

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_cfg1": OptionRow(),
                    f"{name}_t2i_cfg": OptionInfo(cfg, "txt2img CFG", Slider, {"minimum": 0, "maximum": 24, "step": 0.5}),
                    f"{name}_t2i_hr_cfg": OptionInfo(cfg, "txt2img Hires. CFG", Slider, {"minimum": 0, "maximum": 24, "step": 0.5}),
                    f"{name}_i2i_cfg": OptionInfo(cfg, "img2img CFG", Slider, {"minimum": 0, "maximum": 24, "step": 0.5}),
                    f"{name}_cfg0": OptionRow(),
                },
            )
        )

        if (distill := DISTILL.get(arch, None)) is not None:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_dcfg1": OptionRow(),
                        f"{name}_t2i_dcfg": OptionInfo(distill, "txt2img Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_t2i_hr_dcfg": OptionInfo(distill, "txt2img Hires. Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_i2i_dcfg": OptionInfo(distill, "img2img Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_dcfg0": OptionRow(),
                    },
                )
            )

        if (shift := SHIFT.get(arch, None)) is not None:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_show_shift": OptionInfo((shift > 0.0), "Display Shift Slider"),
                        f"{name}_dcfg1": OptionRow(),
                        f"{name}_t2i_dcfg": OptionInfo(abs(shift), "txt2img Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_t2i_hr_dcfg": OptionInfo(abs(shift), "txt2img Hires. Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_i2i_dcfg": OptionInfo(abs(shift), "img2img Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_dcfg0": OptionRow(),
                    },
                )
            )

        if (fps := FRAMES.get(arch.name, 1)) > 1:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_batch1": OptionRow(),
                        f"{name}_t2i_batch_size": OptionInfo(121 if arch is PresetArch.ltx2 else 1, "txt2img Frames", Slider, {"minimum": 9 if arch is PresetArch.ltx2 else 1, "maximum": fps * (30 if arch is PresetArch.ltx2 else 15) + 1, "step": fps}),
                        f"{name}_i2i_batch_size": OptionInfo(121 if arch is PresetArch.ltx2 else 1, "img2img Frames", Slider, {"minimum": 9 if arch is PresetArch.ltx2 else 1, "maximum": fps * (30 if arch is PresetArch.ltx2 else 15) + 1, "step": fps}),
                        f"{name}_batch0": OptionRow(),
                    },
                )
            )
        else:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_batch1": OptionRow(),
                        f"{name}_t2i_batch_size": OptionInfo(1, "txt2img Batch Size", Slider, {"minimum": 1, "maximum": 8, "step": 1}),
                        f"{name}_i2i_batch_size": OptionInfo(1, "img2img Batch Size", Slider, {"minimum": 1, "maximum": 8, "step": 1}),
                        f"{name}_batch0": OptionRow(),
                    },
                )
            )

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_t2i_dim1": OptionRow(),
                    f"{name}_t2i_width": OptionInfo(768 if arch is PresetArch.ltx2 else 0, "txt2img Width", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_i2i_width": OptionInfo(768 if arch is PresetArch.ltx2 else 0, "img2img Width", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_t2i_dim0": OptionRow(),
                    f"{name}_i2i_dim1": OptionRow(),
                    f"{name}_t2i_height": OptionInfo(512 if arch is PresetArch.ltx2 else 0, "txt2img Height", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_i2i_height": OptionInfo(512 if arch is PresetArch.ltx2 else 0, "img2img Height", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_i2i_dim0": OptionRow(),
                },
            )
        )

        if arch is PresetArch.ltx2:
            from gradio import Checkbox, Number, Radio, Textbox

            options_templates.update(
                options_section(
                    ("ui_ltx2", "LTX-2.3", "presets"),
                    {
                        "ltx2_model_path": OptionInfo("diffusers/LTX-2.3-Diffusers", "Model path or Hugging Face ID", Textbox),
                        "ltx2_prunavaed_path": OptionInfo("", "PrunaVAED path or Hugging Face ID", Textbox),
                        "ltx2_offload": OptionInfo("model", "CPU offload", Radio, {"choices": ["none", "model", "sequential"]}),
                        "ltx2_precision": OptionInfo("automatic", "Precision", Radio, {"choices": ["automatic", "bfloat16", "float16"]}),
                        "ltx2_tile_vae": OptionInfo(True, "Tile VAE", Checkbox),
                        "ltx2_fps": OptionInfo(24, "FPS", Number, {"minimum": 1, "maximum": 60, "step": 1}),
                        "ltx2_stg": OptionInfo(1.0, "Video STG", Number, {"minimum": 0, "maximum": 10, "step": 0.1}),
                        "ltx2_modality": OptionInfo(3.0, "Video modality", Number, {"minimum": 0, "maximum": 10, "step": 0.1}),
                        "ltx2_audio_guidance": OptionInfo(7.0, "Audio CFG", Number, {"minimum": 0, "maximum": 20, "step": 0.1}),
                        "ltx2_audio_stg": OptionInfo(1.0, "Audio STG", Number, {"minimum": 0, "maximum": 10, "step": 0.1}),
                        "ltx2_audio_modality": OptionInfo(3.0, "Audio modality", Number, {"minimum": 0, "maximum": 10, "step": 0.1}),
                        "ltx2_guidance_rescale": OptionInfo(0.7, "Guidance rescale", Number, {"minimum": 0, "maximum": 1, "step": 0.05}),
                        "ltx2_guidance_blocks": OptionInfo("28", "Spatio-temporal guidance blocks", Textbox),
                        "ltx2_seed": OptionInfo(-1, "Seed", Number, {"minimum": -1, "maximum": 2**31 - 1, "step": 1}),
                        "ltx2_include_audio": OptionInfo(True, "Include generated audio", Checkbox),
                    },
                )
            )
