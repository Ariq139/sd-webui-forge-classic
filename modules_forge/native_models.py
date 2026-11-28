import os

from modules_forge.native_runtime import require_pipeline_directory

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
    try:
        require_pipeline_directory(resolve_model_path(value), marker)
        return True
    except ValueError:
        return False


def compatible_model_directories(marker: str) -> list[str]:
    values = []
    for root in MODEL_ROOTS:
        if not os.path.isdir(root):
            continue
        for entry in os.scandir(root):
            if entry.is_dir() and is_compatible_model(entry.path, marker):
                # Keep the root identity; equal names may contain different models.
                values.append(os.path.abspath(entry.path))
    return sorted(set(values), key=str.casefold)


def select_model_path(marker, *values):
    for value in values:
        if is_compatible_model(value, marker):
            return os.path.abspath(resolve_model_path(value))
    choices = compatible_model_directories(marker)
    return choices[0] if choices else ""
