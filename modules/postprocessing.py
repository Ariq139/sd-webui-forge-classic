import os
from itertools import chain

import av
import gradio as gr
import numpy as np
from PIL import Image
from tqdm import tqdm

from modules import devices, images, infotext_utils, scripts, scripts_postprocessing, shared, ui_common
from modules.shared import opts


def run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, _, *args, save_output: bool = True):
    devices.torch_gc()

    shared.state.begin(job="extras")

    outputs = []

    if isinstance(image, dict):
        image = image["composite"]

    def get_images(extras_mode, image, image_folder, input_dir):
        if extras_mode == 1:
            for img in image_folder:
                if isinstance(img, Image.Image):
                    image = images.fix_image(img)
                    fn = ""
                else:
                    image = images.read(os.path.abspath(img.name))
                    fn = os.path.splitext(img.name)[0]
                yield image, fn
        elif extras_mode == 2:
            assert not shared.cmd_opts.hide_ui_dir_config, "--hide-ui-dir-config option must be disabled"
            assert input_dir, "input directory not selected"

            image_list = shared.listfiles(input_dir)
            for filename in image_list:
                yield filename, filename
        else:
            assert image, "image not selected"
            yield image, None

    if extras_mode == 2 and output_dir != "":
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples

    infotext = ""

    data_to_process = list(get_images(extras_mode, image, image_folder, input_dir))
    shared.state.job_count = len(data_to_process)

    for image_placeholder, name in data_to_process:
        image_data: Image.Image

        shared.state.nextjob()
        shared.state.textinfo = name
        shared.state.skipped = False

        if shared.state.interrupted or shared.state.stopping_generation:
            break

        if isinstance(image_placeholder, str):
            try:
                image_data = images.read(image_placeholder)
            except Exception:
                continue
        else:
            image_data = image_placeholder

        image_data = image_data if image_data.mode in ("RGBA", "RGB") else image_data.convert("RGB")

        parameters, existing_pnginfo = images.read_info_from_image(image_data)
        if parameters:
            existing_pnginfo["parameters"] = parameters

        initial_pp = scripts_postprocessing.PostprocessedImage(image_data)

        scripts.scripts_postproc.run(initial_pp, args)

        if shared.state.skipped:
            continue

        used_suffixes = {}
        for pp in [initial_pp, *initial_pp.extra_images]:
            suffix = pp.get_suffix(used_suffixes)

            if opts.use_original_name_batch and name is not None:
                basename = os.path.splitext(os.path.basename(name))[0]
                forced_filename = basename + suffix
            else:
                basename = ""
                forced_filename = None

            infotext = ", ".join([k if k == v else f"{k}: {infotext_utils.quote(v)}" for k, v in pp.info.items() if v is not None])

            if opts.enable_pnginfo:
                pp.image.info = existing_pnginfo

            shared.state.assign_current_image(pp.image)

            if save_output:
                fullfn, _ = images.save_image(pp.image, path=outpath, basename=basename, extension=opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="postprocessing", existing_info=existing_pnginfo, forced_filename=forced_filename, suffix=suffix)

            if extras_mode != 2 or show_extras_results:
                outputs.append(pp.image)

    devices.torch_gc()
    shared.state.end()
    return outputs, ui_common.plaintext_to_html(infotext), ""


class _VideoInterrupted(Exception):
    pass


def run_postprocessing_video(_mode, _img, _folder, _in_dir, _out_dir, _show, video_input, *args, save_output: bool = True):
    devices.torch_gc()
    shared.state.begin(job="extras")
    last_image = None
    infotext = ""

    def check_interrupted():
        if shared.state.interrupted or shared.state.stopping_generation or shared.state.skipped:
            raise _VideoInterrupted()

    try:
        if not video_input:
            raise ValueError("Select an input video.")
        with av.open(video_input) as container:
            video_stream = container.streams.best("video")
            if video_stream is None:
                raise ValueError("Input has no video stream.")
            fps = video_stream.average_rate
            if fps is None or fps <= 0:
                raise ValueError("Input video has no usable frame rate.")
            total = video_stream.frames or None
            shared.state.job_count = total or -1

            def processed_frames():
                nonlocal last_image, infotext
                for index, frame in enumerate(tqdm(container.decode(video_stream), desc="Processing Video", total=total, unit="frame")):
                    check_interrupted()
                    shared.state.textinfo = str(index)
                    pp = scripts_postprocessing.PostprocessedImage(frame.to_image())
                    scripts.scripts_postproc.run(pp, args)
                    # Dropping a frame would desynchronize copied audio. Abort the clip.
                    check_interrupted()
                    if last_image is None:
                        infotext = ", ".join([k if k == v else f"{k}: {infotext_utils.quote(v)}" for k, v in pp.info.items() if v is not None])
                    last_image = pp.image.convert("RGB")
                    shared.state.assign_current_image(last_image)
                    shared.state.nextjob()
                    yield np.asarray(last_image, dtype=np.uint8)
                check_interrupted()

            frames = processed_frames()
            first = next(frames, None)
            if first is None:
                return [], ui_common.plaintext_to_html("No video frames decoded."), ""
            if save_output:
                path = images.save_video(
                    os.path.splitext(os.path.basename(video_input))[0],
                    chain((first,), frames),
                    fps=fps,
                    basename=None,
                    info=infotext,
                    audio_copy=video_input,
                )
                status = f"Saved: {path}"
            else:
                for _ in frames:
                    pass
                status = "Video processed without saving."
            return [last_image], ui_common.plaintext_to_html(infotext), ui_common.plaintext_to_html(status)
    except _VideoInterrupted:
        return gr.skip(), gr.skip(), ui_common.plaintext_to_html("Video processing interrupted; no partial output published.")
    finally:
        devices.torch_gc()
        shared.state.end()


def run_postprocessing_webui(id_task, *args, **kwargs):
    if args[0] == 3:
        return run_postprocessing_video(*args, **kwargs)
    else:
        return run_postprocessing(*args, **kwargs)


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True, max_side_length: int = 0):
    # Handler for API (does not support video)

    args = scripts.scripts_postproc.create_args_for_run(
        {
            "Upscale": {
                "upscale_enabled": True,
                "upscale_mode": resize_mode,
                "upscale_by": upscaling_resize,
                "max_side_length": max_side_length,
                "upscale_to_width": upscaling_resize_w,
                "upscale_to_height": upscaling_resize_h,
                "upscale_crop": upscaling_crop,
                "upscaler_1_name": extras_upscaler_1,
                "upscaler_2_name": extras_upscaler_2,
                "upscaler_2_visibility": extras_upscaler_2_visibility,
            },
            "GFPGAN": {
                "enable": True,
                "gfpgan_visibility": gfpgan_visibility,
            },
            "CodeFormer": {
                "enable": True,
                "codeformer_visibility": codeformer_visibility,
                "codeformer_weight": codeformer_weight,
            },
        }
    )

    return run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, "", *args, save_output=save_output)
