import inspect
import json
from contextlib import closing

import gradio as gr

import modules.scripts
import modules.shared as shared
from modules import infotext_utils, processing, ui_result_utils
from modules.infotext_utils import (
    create_override_settings_dict,
    parse_generation_parameters,
)
from modules.shared import opts
from modules.ui import plaintext_to_html
from modules_forge import main_thread, native_requests
from modules_forge.presets import get_capabilities


def txt2img_create_processing(id_task: str, request: gr.Request, prompt: str, negative_prompt: str, prompt_styles, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_additional_modules: list, hr_sampler_name: str, hr_scheduler: str, hr_prompt: str, hr_negative_prompt, hr_cfg: float, hr_distilled_cfg: float, override_settings_texts, *args, force_enable_hr=False):
    try:
        n_iter = int(n_iter)
    except (TypeError, ValueError):
        n_iter = 1
    infinite_generation = n_iter == 0
    n_iter = max(1, n_iter)

    capabilities = get_capabilities(
        opts.forge_preset,
        getattr(opts, f"forge_checkpoint_{opts.forge_preset}", getattr(opts, "sd_model_checkpoint", "")),
    )
    if not capabilities["txt2img"]:
        raise RuntimeError(f"The {opts.forge_preset} preset does not support txt2img.")

    if not capabilities["cfg"]:
        cfg_scale = hr_cfg = 1.0
    if not capabilities["negative_prompt"]:
        negative_prompt = ""
        hr_negative_prompt = ""
    if not capabilities["hires"]:
        enable_hr = False
    if not capabilities["txt2img_batch_count"]:
        n_iter = 1
        infinite_generation = False
    if not capabilities["txt2img_batch_size"]:
        batch_size = 1

    override_settings = create_override_settings_dict(override_settings_texts)

    if force_enable_hr and capabilities["hires"]:
        enable_hr = True

    p = processing.StableDiffusionProcessingTxt2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
        n_iter=n_iter,
        infinite_generation=infinite_generation,
        cfg_scale=cfg_scale,
        distilled_cfg_scale=distilled_cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == "Use same checkpoint" else hr_checkpoint_name,
        hr_additional_modules=hr_additional_modules,
        hr_sampler_name=None if hr_sampler_name == "Use same sampler" else hr_sampler_name,
        hr_scheduler=None if hr_scheduler == "Use same scheduler" else hr_scheduler,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        hr_cfg=hr_cfg,
        hr_distilled_cfg=hr_distilled_cfg,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    p.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    return p


def txt2img_upscale_function(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
    capabilities = get_capabilities(
        opts.forge_preset,
        getattr(opts, f"forge_checkpoint_{opts.forge_preset}", getattr(opts, "sd_model_checkpoint", "")),
    )
    if not capabilities["hires"]:
        return gallery, generation_info, "This preset does not support the Forge Hires fix/upscale action.", ""

    assert len(gallery) > 0, "No image to upscale"

    if gallery_index < 0 or gallery_index >= len(gallery):
        return gallery, generation_info, f"Bad image index: {gallery_index}", ""

    geninfo = ui_result_utils.decode_metadata(generation_info)

    # catch situation where user tries to hires-fix the grid: probably a mistake, results can be bad aspect ratio - just don't do it
    first_image_index = ui_result_utils.first_image_index(geninfo)
    if gallery_index < first_image_index or not ui_result_utils.image_infotext(geninfo, gallery_index):
        return gallery, generation_info, "Unable to upscale grid or control images.", ""

    p = txt2img_create_processing(id_task, request, *args, force_enable_hr=True)
    if opts.txt2img_upscale_single_batch:
        p.batch_size = 1
        p.n_iter = 1

    # txt2img_upscale attribute that signifies this is called by txt2img_upscale
    p.txt2img_upscale = True
    p.infinite_generation = False

    image_info = gallery[gallery_index]
    p.firstpass_image = infotext_utils.image_from_url_text(image_info)

    if opts.txt2img_upscale_same_seed:
        parameters = parse_generation_parameters(geninfo.get("infotexts")[gallery_index], [])
        p.seed = parameters.get("Seed", -1)
        p.subseed = parameters.get("Variation seed", -1)

    # update processing width/height based on actual dimensions of source image
    p.width = gallery[gallery_index][0].size[0]
    p.height = gallery[gallery_index][0].size[1]
    p.extra_generation_params["Original Size"] = f"{args[8]}x{args[7]}"

    p.override_settings["save_images_before_highres_fix"] = False

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    insert = getattr(shared.opts, "hires_button_gallery_insert", False)
    first = processed.index_of_first_image
    new_images = processed.images[first:]
    new_texts = processed.infotexts[first:first + len(new_images)]
    if not new_images or len(new_texts) != len(new_images) or any(not text for text in new_texts):
        return gr.skip(), gr.skip(), gr.skip(), "No completed upscale result."

    for image in gallery:
        filename = getattr(image[0], "filename", None)
        if filename:
            image[0].already_saved_as = filename.rsplit("?", 1)[0]

    new_gallery = list(gallery)
    start = gallery_index + int(insert)
    end = start if insert else start + 1
    new_gallery[start:end] = new_images
    geninfo = ui_result_utils.splice_gallery_metadata(
        geninfo, ui_result_utils.decode_metadata(processed.js()), len(gallery),
        gallery_index, insert, new_texts,
    )
    selected = start if not insert or shared.opts.hires_insert_index else gallery_index
    return gr.update(value=new_gallery, selected_index=selected), json.dumps(geninfo), plaintext_to_html(geninfo["infotexts"][selected]), plaintext_to_html(processed.comments, classname="comments")


def txt2img_function(id_task: str, request: gr.Request, *args):
    capabilities = get_capabilities(
        opts.forge_preset,
        getattr(opts, f"forge_checkpoint_{opts.forge_preset}", getattr(opts, "sd_model_checkpoint", "")),
    )
    if not capabilities["txt2img"]:
        raise RuntimeError(f"The {opts.forge_preset} preset does not support txt2img.")

    if opts.forge_preset in ("ltx2", "ideogram"):
        # Bind to the actual processing signature; script controls have a
        # separate dynamic layout.
        native = inspect.signature(txt2img_create_processing).bind(id_task, request, *args).arguments
        runner = modules.scripts.scripts_txt2img
        steps = native_requests.script_control(runner, native["args"], "ScriptSampler", "steps")
        seed = native_requests.script_control(runner, native["args"], "ScriptSeed", "seed")
        count = native_requests.positive_integer(native["n_iter"], "Batch Count")
        batch_size = native_requests.positive_integer(native["batch_size"], "Batch Size / Frames")
        prompt = shared.prompt_styles.apply_styles_to_prompt(native["prompt"], native["prompt_styles"] or [])
        width, height = native["width"], native["height"]
        guidance = native["cfg_scale"]
        if opts.forge_preset == "ltx2":
            from modules.ui_ltx2_video import generate_from_preset

            if count != 1:
                raise ValueError("LTX-2.3 supports one video per request; set Batch Count to 1.")
            negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(native["negative_prompt"], native["prompt_styles"] or [])
            video_path, status = generate_from_preset(
                prompt, negative_prompt, width=width, height=height,
                frames=batch_size, guidance=guidance, steps=steps, seed=seed,
            )
            if not video_path:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip(), plaintext_to_html(status)
            generation_info_js = json.dumps({"infotexts": [status], "prompt": prompt, "negative_prompt": negative_prompt})
            return gr.update(value=None, visible=False), gr.update(value=video_path, visible=True), generation_info_js, plaintext_to_html(status), ""

        from modules.ui_ideogram import generate_from_preset

        images, status = generate_from_preset(
            prompt, width=width, height=height,
            num_images=count * batch_size, steps=steps, guidance=guidance, seed=seed,
        )
        if not images:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), plaintext_to_html(status)
        # One RNG is used for the entire request. Do not invent per-image seeds.
        generation_info_js = json.dumps({
            "infotexts": [None] * len(images), "prompt": prompt,
            "negative_prompt": "", "width": width, "height": height,
            "batch_size": len(images), "steps": steps, "cfg_scale": guidance,
            "sampler_name": "Ideogram 4", "index_of_first_image": 0,
            "sd_model_name": getattr(opts, "forge_checkpoint_ideogram", ""),
            "request_seed": seed, "all_seeds": [], "styles": native["prompt_styles"] or [],
        })
        return gr.update(value=images, visible=True), gr.update(value=None, visible=False), generation_info_js, plaintext_to_html(status), ""

    p = txt2img_create_processing(id_task, request, *args)

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    if processed.video_path is None:
        gallery_arg = gr.update(value=processed.images + processed.extra_images, visible=True)
        video_arg = gr.update(value=None, visible=False)
    else:
        gallery_arg = gr.update(value=None, visible=False)
        video_arg = gr.update(value=processed.video_path, visible=True)

    return gallery_arg, video_arg, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def txt2img_upscale(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
    return main_thread.run_and_wait_result(txt2img_upscale_function, id_task, request, gallery, gallery_index, generation_info, *args)


def txt2img_upscale_ui(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
    # Record the same five-field result contract used by normal txt2img restore.
    result = txt2img_upscale(id_task, request, gallery, gallery_index, generation_info, *args)
    if result is None:
        raise RuntimeError("Upscale failed")
    return result[0], gr.skip(), *result[1:]


def txt2img(id_task: str, request: gr.Request, *args):
    return main_thread.run_and_wait_result(txt2img_function, id_task, request, *args)
