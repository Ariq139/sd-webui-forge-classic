"""Dependency-free helpers for gallery metadata and selection contracts."""
import json
import os


def decode_metadata(raw):
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def first_image_index(data):
    try:
        return max(0, int(data.get("index_of_first_image", 0)))
    except (TypeError, ValueError, OverflowError):
        return 0


def safe_batch_size(data):
    try:
        return max(1, int(data.get("batch_size", 1)))
    except (TypeError, ValueError, OverflowError):
        return 1


def selected_range(count, index, first, selected_only):
    try:
        index = int(index)
    except (TypeError, ValueError, OverflowError):
        index = -1
    if selected_only and first <= index < count:
        return range(index, index + 1)
    return range(count)


def image_infotext(data, index):
    texts = data.get("infotexts")
    if not isinstance(texts, list) or not 0 <= index < len(texts):
        return None
    text = texts[index]
    return text if isinstance(text, str) and text.strip() else None


def reusable_seed(parameters, variation):
    try:
        use_variation = variation and float(parameters.get("Variation seed strength", 0)) > 0
        value = int(parameters.get("Variation seed" if use_variation else "Seed"))
        return value if value >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def path_is_child(parent, child):
    try:
        parent = os.path.normcase(os.path.realpath(parent))
        child = os.path.normcase(os.path.realpath(child))
        return child != parent and os.path.commonpath([parent, child]) == parent
    except (TypeError, ValueError, OSError):
        return False


def _fit(values, length):
    values = list(values) if isinstance(values, (list, tuple)) else []
    return (values + [None] * length)[:length]


def splice_gallery_metadata(data, new_data, gallery_count, target, insert, new_texts):
    """Splice generated-image metadata separately from grid/auxiliary entries."""
    first = first_image_index(data)
    texts = [image_infotext(data, i) for i in range(gallery_count)]
    generated = [i for i in range(first, gallery_count) if texts[i] is not None]
    slot = generated.index(target)
    count = len(new_texts)
    array_start = slot + int(insert)
    array_end = array_start if insert else array_start + 1
    result = dict(data)
    if slot == 0 and not insert:
        result.update(new_data)

    for key in ("all_prompts", "all_negative_prompts", "all_seeds", "all_subseeds"):
        old_values = _fit(data.get(key), len(generated))
        new_values = _fit(new_data.get(key), count)
        old_values[array_start:array_end] = new_values
        result[key] = old_values

    start = target + int(insert)
    end = start if insert else start + 1
    texts[start:end] = new_texts
    result["infotexts"] = texts
    result["index_of_first_image"] = first
    return result
