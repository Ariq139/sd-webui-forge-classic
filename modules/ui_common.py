import csv
import dataclasses
import html
import json
import os
from contextlib import nullcontext

import gradio as gr

import modules.images
import modules.infotext_utils as parameters_copypaste
from modules import call_queue, shared, ui_result_utils, ui_tempdir, util
from modules.ui_components import ToolButton

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄


def update_generation_info(generation_info, html_info, img_index):
    try:
        generation_info = json.loads(generation_info)
        img_index = int(img_index)
        infotext = ui_result_utils.image_infotext(generation_info, img_index)
        return plaintext_to_html(infotext) if infotext else html_info
    except Exception:
        pass
    # if the json parse or anything else fails, just return the old html_info
    return html_info


def plaintext_to_html(text, classname=None):
    content = "<br>\n".join(html.escape(x) for x in (text or "").split("\n"))

    return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"


def update_logfile(logfile_path, fields):
    """Update a logfile from old format to new format to maintain CSV integrity."""
    with open(logfile_path, "r", encoding="utf8", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)

    # blank file: leave it as is
    if not rows:
        return

    # file is already synced, do nothing
    if len(rows[0]) == len(fields):
        return

    rows[0] = fields

    # append new fields to each row as empty values
    for row in rows[1:]:
        while len(row) < len(fields):
            row.append("")

    with open(logfile_path, "w", encoding="utf8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def save_files(js_data, images, do_make_zip, index):
    filenames = []
    fullfns = []
    records = []

    class MyObject:
        def __init__(self, data):
            self.__dict__.update(data)

    data = ui_result_utils.decode_metadata(js_data)
    first = ui_result_utils.first_image_index(data)
    if not images:
        return gr.update(value=[], visible=False), plaintext_to_html("No images to save.")

    kinds = data.get("gallery_kinds")
    if not isinstance(kinds, list) or len(kinds) != len(images) or any(kind not in ("image", "grid", "auxiliary") for kind in kinds):
        kinds = ["grid" if i < first else "image" for i in range(len(images))]
    try:
        selected_index = int(index)
    except (TypeError, ValueError, OverflowError):
        selected_index = -1
    if 0 <= selected_index < len(kinds) and kinds[selected_index] == "grid":
        selected_index = -1  # A grid is a batch overview, not a selected sample.
    selected = ui_result_utils.selected_range(len(images), selected_index, first, shared.opts.save_selected_only)
    ordinals = []
    generated = 0
    for kind in kinds:
        ordinals.append(generated)
        generated += kind == "image"
    path = shared.opts.outdir_save
    save_to_dirs = shared.opts.use_save_to_dirs_for_ui
    extension = shared.opts.samples_format
    os.makedirs(path, exist_ok=True)

    fields = [
        "prompt", "seed", "width", "height", "sampler", "cfgs", "steps",
        "filename", "negative_prompt", "sd_model_name", "sd_model_hash",
    ]
    logfile_path = os.path.join(path, "log.csv")
    if shared.opts.save_write_log_csv and os.path.exists(logfile_path):
        update_logfile(logfile_path, fields)

    with open(logfile_path, "a", encoding="utf8", newline="") if shared.opts.save_write_log_csv else nullcontext() as file:
        writer = csv.writer(file) if file else None
        if file and file.tell() == 0:
            writer.writerow(fields)

        for image_index in selected:
            image = images[image_index][0]
            is_grid = kinds[image_index] == "grid"
            raw = ui_result_utils.image_infotext(data, image_index)
            parameters = {}
            if raw:
                try:
                    parameters = parameters_copypaste.parse_generation_parameters(raw, [], style_mode="Ignore")
                except Exception:
                    # Missing or invalid metadata must not prevent saving.
                    raw = None
            native = "request_seed" in data
            prompt = parameters.get("Prompt", data.get("prompt", "") if native else "")
            seed = parameters.get("Seed")
            p = MyObject(data)
            p.batch_size = ui_result_utils.safe_batch_size(data)
            ordinal = ordinals[image_index]
            p.iteration, p.batch_index = divmod(ordinal, p.batch_size)
            p.n_iter = max(1, (generated + p.batch_size - 1) // p.batch_size)
            p.prompt = prompt
            p.negative_prompt = parameters.get("Negative prompt", data.get("negative_prompt", "") if native else "")
            p.sd_model_name = parameters.get("Model", data.get("sd_model_name", ""))
            p.sd_model_hash = parameters.get("Model hash", data.get("sd_model_hash", ""))
            p.steps = parameters.get("Steps", data.get("steps") if native else None)
            p.sampler_name = parameters.get("Sampler", data.get("sampler_name", "") if native else "")
            p.cfg_scale = parameters.get("CFG scale", data.get("cfg_scale") if native else None)
            p.styles = data.get("styles") or []
            p.all_seeds = data.get("all_seeds") or ([seed] if seed is not None else [])
            fullfn, txt_fullfn = modules.images.save_image(
                image, path, "", seed=seed, prompt=prompt, extension=extension,
                info=raw, grid=is_grid, p=p, save_to_dirs=save_to_dirs,
            )
            if not fullfn:
                continue
            filename = os.path.relpath(fullfn, path)
            filenames.append(filename)
            fullfns.append(fullfn)
            records.append((p, seed, prompt, image))
            if txt_fullfn:
                filenames.append(os.path.relpath(txt_fullfn, path))
                fullfns.append(txt_fullfn)
            if writer:
                writer.writerow([prompt, seed, image.width, image.height, p.sampler_name, p.cfg_scale,
                                 p.steps, filename, p.negative_prompt, p.sd_model_name, p.sd_model_hash])

    if not fullfns:
        return gr.update(value=[], visible=False), plaintext_to_html("No images were saved.")

    if do_make_zip:
        p, seed, prompt, image = records[0]
        p.all_seeds = [record[1] for record in records if record[1] is not None]
        namegen = modules.images.FilenameGenerator(p, seed, prompt, image, True)
        pattern = shared.opts.grid_zip_filename_pattern or ("[datetime]_[[model_name]]_[seed]-[seed_last]" if p.all_seeds else "[datetime]")
        zip_filename = namegen.apply(pattern)
        from zipfile import ZipFile

        # Never overwrite an archive made by another save in the same second.
        suffix = 0
        while True:
            name = zip_filename if suffix == 0 else f"{zip_filename}-{suffix}"
            zip_filepath = os.path.join(path, f"{name}.zip")
            try:
                with ZipFile(zip_filepath, "x") as zip_file:
                    for fullfn, filename in zip(fullfns, filenames):
                        zip_file.write(fullfn, filename)
                break
            except FileExistsError:
                suffix += 1
        fullfns.insert(0, zip_filepath)

    return gr.update(value=fullfns, visible=True), plaintext_to_html(f"Saved: {filenames[0]}")

@dataclasses.dataclass
class OutputPanel:
    gallery: gr.Gallery = None
    player: gr.Video = None
    generation_info: gr.HTML = None
    infotext: gr.HTML = None
    html_log: gr.HTML = None
    button_upscale: gr.Button = None


def create_output_panel(tabname, outdir, toprow=None):
    res = OutputPanel()

    def open_folder(f, images=None, index=None):
        if shared.cmd_opts.hide_ui_dir_config:
            return

        try:
            if "Sub" in shared.opts.open_dir_button_choice:
                image_dir = os.path.split(images[index]["name"].rsplit("?", 1)[0])[0]
                if "temp" in shared.opts.open_dir_button_choice or not ui_tempdir.is_gradio_temp_path(image_dir):
                    f = image_dir
        except Exception:
            pass

        util.open_folder(f)

    with gr.Column(elem_id=f"{tabname}_results"):
        if toprow:
            toprow.create_inline_toprow_image()

        with gr.Column(variant="panel", elem_id=f"{tabname}_results_panel"):
            with gr.Group(elem_id=f"{tabname}_gallery_container"):
                res.gallery = gr.Gallery(label="Output", show_label=False, elem_id=f"{tabname}_gallery", columns=4, preview=True, height=shared.opts.gallery_height or None, interactive=False, type="pil", object_fit="contain")
                res.player = gr.Video(label="Output", show_label=False, elem_id=f"{tabname}_player", height=shared.opts.gallery_height or None, interactive=False, autoplay=shared.opts.video_player_auto, loop=shared.opts.video_player_loop, include_audio=tabname in {"txt2img", "img2img"}, show_download_button=tabname in {"txt2img", "img2img"}, show_share_button=False, visible=False)

            with gr.Row(elem_id=f"image_buttons_{tabname}", elem_classes="image-buttons"):
                open_folder_button = ToolButton(folder_symbol, elem_id=f"{tabname}_open_folder", visible=not shared.cmd_opts.hide_ui_dir_config, tooltip="Open images output directory.")

                if tabname != "extras":
                    save = ToolButton("💾", elem_id=f"save_{tabname}", tooltip=f"Save the image to a dedicated directory ({shared.opts.outdir_save}).")
                    save_zip = ToolButton("🗃️", elem_id=f"save_zip_{tabname}", tooltip=f"Save zip archive with images to a dedicated directory ({shared.opts.outdir_save})")

                buttons = {
                    "img2img": ToolButton("🖼️", elem_id=f"{tabname}_send_to_img2img", tooltip="Send image and generation parameters to img2img tab."),
                    "inpaint": ToolButton("🎨️", elem_id=f"{tabname}_send_to_inpaint", tooltip="Send image and generation parameters to img2img inpaint tab."),
                    "extras": ToolButton("📐", elem_id=f"{tabname}_send_to_extras", tooltip="Send image and generation parameters to extras tab."),
                }

                if tabname == "txt2img":
                    res.button_upscale = ToolButton("✨", elem_id=f"{tabname}_upscale", tooltip="Create an upscaled version of the current image using hires fix settings.")

            open_folder_button.click(
                fn=lambda images, index: open_folder(shared.opts.outdir_samples or outdir, images, index),
                _js="(y, w) => [y, selected_gallery_index()]",
                inputs=[
                    res.gallery,
                    open_folder_button,  # placeholder for index
                ],
                outputs=[],
            )

            if tabname != "extras":
                download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False, elem_id=f"download_files_{tabname}")

                with gr.Group():
                    res.infotext = gr.HTML(elem_id=f"html_info_{tabname}", elem_classes="infotext")
                    res.html_log = gr.HTML(elem_id=f"html_log_{tabname}", elem_classes="html-log")

                    res.generation_info = gr.Textbox(visible=False, elem_id=f"generation_info_{tabname}")
                    if tabname == "txt2img" or tabname == "img2img":
                        gallery_index = gr.Number(value=0, visible=False)
                        generation_info_button = gr.Button(visible=False, elem_id=f"{tabname}_generation_info_button")
                        generation_info_button.click(
                            fn=update_generation_info,
                            _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
                            inputs=[res.generation_info, res.infotext, gallery_index],
                            outputs=[res.infotext],
                            show_progress=False,
                        )
                        res.gallery.select(
                            fn=update_generation_info,
                            _js=f"(x, y, z) => [x, y, selected_gallery_index_id('{tabname}_gallery')]",
                            inputs=[res.generation_info, res.infotext, gallery_index],
                            outputs=[res.infotext],
                            show_progress=False,
                        )

                    save.click(
                        fn=call_queue.wrap_gradio_call_no_job(save_files),
                        _js="(x, y, z, w) => [x, y, false, selected_gallery_index()]",
                        inputs=[
                            res.generation_info,
                            res.gallery,
                            res.infotext,
                            res.infotext,
                        ],
                        outputs=[
                            download_files,
                            res.html_log,
                        ],
                        show_progress=False,
                    )

                    save_zip.click(
                        fn=call_queue.wrap_gradio_call_no_job(save_files),
                        _js="(x, y, z, w) => [x, y, true, selected_gallery_index()]",
                        inputs=[
                            res.generation_info,
                            res.gallery,
                            res.infotext,
                            res.infotext,
                        ],
                        outputs=[
                            download_files,
                            res.html_log,
                        ],
                    )

            else:
                res.generation_info = gr.HTML(elem_id=f"html_info_x_{tabname}")
                res.infotext = gr.HTML(elem_id=f"html_info_{tabname}", elem_classes="infotext")
                res.html_log = gr.HTML(elem_id=f"html_log_{tabname}")

            paste_field_names = []
            if tabname == "txt2img":
                paste_field_names = modules.scripts.scripts_txt2img.paste_field_names
            elif tabname == "img2img":
                paste_field_names = modules.scripts.scripts_img2img.paste_field_names

            for paste_tabname, paste_button in buttons.items():
                parameters_copypaste.register_paste_params_button(
                    parameters_copypaste.ParamBinding(
                        paste_button=paste_button,
                        tabname=paste_tabname,
                        source_tabname=tabname if (tabname == "txt2img" or (tabname == "img2img" and shared.opts.allow_i2i_send_info)) else None,
                        source_image_component=res.gallery,
                        paste_field_names=paste_field_names,
                    )
                )

    return res


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    refresh_components = refresh_component if isinstance(refresh_component, list) else [refresh_component]

    label = None
    for comp in refresh_components:
        label = getattr(comp, "label", None)
        if label is not None:
            break

    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            for comp in refresh_components:
                setattr(comp, k, v)

        return [gr.update(**(args or {})) for _ in refresh_components] if len(refresh_components) > 1 else gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id, tooltip=f"{label}: refresh" if label else "Refresh")
    refresh_button.click(fn=refresh, outputs=refresh_components)
    return refresh_button


def setup_dialog(button_show, dialog, *, button_close=None):
    """Sets up the UI so that the dialog (gr.Group) is invisible, and is only shown when buttons_show is clicked, in a fullscreen modal window."""

    dialog.visible = False

    button_show.click(
        fn=lambda: gr.update(visible=True),
        outputs=[dialog],
    ).then(fn=None, _js="function(){ popupId('" + dialog.elem_id + "'); }")

    if button_close:
        button_close.click(fn=None, _js="closePopup")
