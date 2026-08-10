import json
import os

from modules import paths


MODEL_ROOTS = (
    os.path.join(paths.models_path, "Stable-diffusion"),
    os.path.join(paths.models_path, "diffusers"),
)


def resolve_model_path(value: str) -> str:
    value = str(value or "").strip()
    if not value or os.path.exists(value):
        return os.path.abspath(value) if value and os.path.exists(value) else value

    for root in (*MODEL_ROOTS, paths.models_path):
        candidate = os.path.join(root, value)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return value


def is_compatible_model(value: str, marker: str) -> bool:
    model_path = resolve_model_path(value)
    model_index = os.path.join(model_path, "model_index.json")
    if not os.path.isdir(model_path) or not os.path.isfile(model_index):
        return False

    try:
        with open(model_index, encoding="utf-8") as file:
            return marker in str(json.load(file).get("_class_name", ""))
    except (OSError, ValueError):
        return False


def compatible_model_directories(marker: str) -> list[str]:
    values = []
    for root in MODEL_ROOTS:
        if not os.path.isdir(root):
            continue
        for entry in os.scandir(root):
            if entry.is_dir() and is_compatible_model(entry.path, marker):
                values.append(os.path.relpath(entry.path, root))
    return sorted(set(values), key=str.casefold)
