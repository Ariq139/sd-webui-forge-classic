"""Dependency-free native recipe and result contracts."""
import json
import math
import os
import tempfile

IDEOGRAM_RECIPES = {
    "Quality 48": (48, 3, 0.0, 1.5),
    "Default 20": (20, 2, 0.0, 1.75),
    "Turbo 12": (12, 1, 0.5, 1.75),
}
IDEOGRAM_MODES = ("Custom", *IDEOGRAM_RECIPES)
LTX_MODES = ("Full", "Distilled 8", "Full + refine 2x", "Distilled + refine 2x")
QUANTIZATION_MODES = ("Checkpoint", "NF4 denoisers", "NF4 denoisers + text encoder")


def ideogram_recipe(mode, steps, guidance, mu, std, polish, custom_schedule=""):
    if mode in IDEOGRAM_RECIPES:
        steps, tail, mu, std = IDEOGRAM_RECIPES[mode]
        schedule = [7.0] * (steps - tail) + [3.0] * tail
        return dict(num_inference_steps=steps, guidance_scale=None, guidance_schedule=schedule, mu=mu, std=std)
    if mode != "Custom":
        raise ValueError("Unknown Ideogram recipe.")
    if not 1 <= steps <= 150 or not all(math.isfinite(x) for x in (guidance, mu, std)):
        raise ValueError("Invalid Ideogram sampling parameters.")
    if custom_schedule and str(custom_schedule).strip():
        schedule = [float(x.strip()) for x in str(custom_schedule).split(",")]
        if len(schedule) != steps or any(not math.isfinite(x) or not 0 <= x <= 24 for x in schedule):
            raise ValueError("Custom CFG schedule needs one finite weight (0-24) per step, in sampling order.")
        guidance = None
    elif polish:
        tail = min(3, steps - 1)
        schedule = [guidance] * (steps - tail) + [min(guidance, 3.0)] * tail
        guidance = None
    else:
        schedule = None
    return dict(num_inference_steps=steps, guidance_scale=guidance, guidance_schedule=schedule, mu=mu, std=std)


def ltx_recipe(mode, kwargs, distilled_sigmas, refine_sigmas):
    if mode not in LTX_MODES:
        raise ValueError("Unknown LTX workflow.")
    first = dict(kwargs)
    refine = None
    neutral = dict(guidance_scale=1.0, stg_scale=0.0, modality_scale=1.0, guidance_rescale=0.0,
                   audio_guidance_scale=1.0, audio_stg_scale=0.0, audio_modality_scale=1.0,
                   audio_guidance_rescale=0.0, spatio_temporal_guidance_blocks=None)
    if mode.startswith("Distilled"):
        first.update(neutral, sigmas=list(distilled_sigmas), num_inference_steps=len(distilled_sigmas))
    if mode.endswith("2x"):
        first["output_type"] = "latent"
        refine = dict(first)
        refine.update(neutral, width=first["width"] * 2, height=first["height"] * 2,
                      sigmas=list(refine_sigmas), num_inference_steps=len(refine_sigmas),
                      noise_scale=refine_sigmas[0], output_type="np")
    return first, refine


def seed_sequence(seed, count):
    if not 0 <= seed < 2**63 or not 1 <= count <= 8:
        raise ValueError("Invalid native seed or image count.")
    return [(seed + i) % (2**63) for i in range(count)]


def native_gallery(metadata, seeds, texts):
    """The shared gallery uses actual image RNG seeds, never fabricated ones."""
    if len(seeds) != len(texts):
        raise ValueError("Native result metadata does not match its images.")
    count = len(seeds)
    return dict(metadata, all_seeds=list(seeds), seed=seeds[0] if seeds else metadata.get("request_seed"),
                all_prompts=[metadata["prompt"]] * count, all_negative_prompts=[metadata.get("negative_prompt", "")] * count,
                all_subseeds=[None] * count, infotexts=list(texts), index_of_first_image=0,
                gallery_kinds=["image"] * count, batch_size=count, styles=[])


def atomic_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".forge-native-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_native_metadata(image):
    raw = getattr(image, "info", {}).get("forge_native")
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def native_infotext(metadata, seed):
    """A standard infotext footer; the full structured request lives in JSON."""
    prompt = metadata["prompt"]
    negative = metadata.get("negative_prompt", "")
    model = metadata.get("sd_model_name", "")
    return (f"{prompt}\nNegative prompt: {negative}\nSteps: {metadata['steps']}, "
            f"Sampler: {metadata['sampler_name']}, CFG scale: {metadata['cfg_scale']}, "
            f"Seed: {seed}, Size: {metadata['width']}x{metadata['height']}, Model: {json.dumps(model, ensure_ascii=False)}")
