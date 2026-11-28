"""Native result publication and gallery compatibility."""
import json
import os
import uuid

from modules_forge.native_recipes import atomic_json, native_gallery, native_infotext, read_native_metadata


def finalize_images(images, metadata, seeds):
    from modules import images as image_io, shared

    if len(images) != len(seeds):
        raise ValueError("Native image and seed counts differ.")
    saved, warnings = [], []
    for index, (image, seed) in enumerate(zip(images, seeds)):
        item = dict(metadata, seed=seed, image_index=index)
        text = native_infotext(item, seed)
        image.info["parameters"] = text
        image.info["forge_native"] = json.dumps(item, ensure_ascii=False, allow_nan=False)
        if not shared.opts.samples_save:
            continue
        try:
            path, _ = image_io.save_image(
                image, shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples, "ideogram",
                seed=seed, prompt=metadata["prompt"], extension=shared.opts.samples_format,
                info=text, existing_info=dict(image.info),
                forced_filename=f"ideogram-{seed}-{uuid.uuid4().hex}", save_to_dirs=False)
            if path:
                saved.append(path)
                atomic_json(os.path.splitext(path)[0] + ".native.json", item)
        except Exception as error:
            warnings.append(f"Image {index + 1} save/metadata failed: {type(error).__name__}")
    return saved, warnings


def gallery_metadata(images):
    metadata = read_native_metadata(images[0]) if images else {}
    seeds, texts = [], []
    for image in images:
        item = read_native_metadata(image)
        if not item or item.get("request_seed") != metadata.get("request_seed"):
            raise ValueError("Native gallery metadata is missing or inconsistent.")
        seeds.append(item["seed"])
        texts.append(image.info.get("parameters", ""))
    return native_gallery(metadata, seeds, texts)


def video_metadata(path):
    try:
        with open(os.path.splitext(path)[0] + ".native.json", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return metadata if isinstance(metadata, dict) else {}
    except (OSError, ValueError):
        return {}
