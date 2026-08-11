"""Opt-in persistent exports and component cache for the MMGP path."""

import html
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import torch

from backend import memory_management, mmgp_loader


logger = logging.getLogger("mmgp_cache")
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


def _options():
    from modules.shared import opts

    return opts


def root() -> Path:
    from modules import paths

    value = str(getattr(_options(), "forge_memory_cache_directory", "models/MMGP-cache") or "models/MMGP-cache")
    path = Path(value)
    return path if path.is_absolute() else Path(paths.script_path) / path


def enabled() -> bool:
    return bool(
        memory_management.mmgp_enabled()
        and memory_management.feature_enabled("enabled")
        and getattr(_options(), "forge_memory_persistent_cache_enabled", False)
    )


def _safe_name(value: str) -> str:
    return _SAFE_NAME.sub("_", str(value)).strip("._") or "component"


def _component_key(model_path: str, name: str, files, dtype: torch.dtype) -> str:
    parts = [os.path.abspath(str(model_path)), str(name), str(dtype)]
    for file in files:
        try:
            stat = os.stat(file)
            parts.append(f"{os.path.abspath(file)}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            parts.append(os.path.abspath(file))
    options = _options()
    parts.extend(
        (
            str(getattr(options, "forge_memory_alternate_quantization", False)),
            str(getattr(options, "forge_memory_quantization_type", "qint8")),
        )
    )
    import hashlib

    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]


def cached_component_path(model_path: str, name: str, files, dtype: torch.dtype) -> Path | None:
    if not enabled():
        return None
    path = root() / "components" / _component_key(model_path, name, files, dtype) / f"{_safe_name(name)}.safetensors"
    return path if path.is_file() and path.stat().st_size > 0 else None


def _save_model(model, path: Path, config_path: Path | None = None):
    from mmgp import offload

    options = _options()
    kwargs = {
        "do_quantize": bool(getattr(options, "forge_memory_alternate_quantization", False)),
        "quantizationType": mmgp_loader.quantization_type(),
        "verboseLevel": 0,
    }
    if config_path is not None and config_path.is_file():
        kwargs["config_file_path"] = str(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f"{path.stem}-", suffix=".safetensors", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        offload.save_model(model, str(temporary), **kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_pipeline_components(pipeline, model_path: str, component_files: dict[str, list[str]], dtype: torch.dtype) -> list[Path]:
    """Persist native components after a successful opt-in MMGP load."""
    if not enabled():
        return []
    saved = []
    for name, files in component_files.items():
        model = getattr(pipeline, name, None)
        if not isinstance(model, torch.nn.Module) or not files:
            continue
        path = root() / "components" / _component_key(model_path, name, files, dtype) / f"{_safe_name(name)}.safetensors"
        try:
            config_path = Path(model_path) / name / "config.json"
            _save_model(model, path, config_path=config_path)
            saved.append(path)
            logger.info("MMGP cached native component %s at %s", name, path)
        except Exception:
            logger.warning("Could not cache native MMGP component %s", name, exc_info=True)
    return saved


def _loaded_models():
    from modules import shared

    try:
        from mmgp import offload
    except Exception:
        return []

    models = []
    seen = set()

    def add(prefix, obj):
        if obj is None:
            return
        try:
            extracted = offload.extract_models(obj, prefix=prefix)
        except Exception:
            logger.debug("Could not inspect loaded MMGP object", exc_info=True)
            return
        for name, model in extracted.items():
            if isinstance(model, torch.nn.Module) and id(model) not in seen:
                seen.add(id(model))
                models.append((name, model))

    add("forge", getattr(shared, "sd_model", None))
    for module_name, prefix in (("modules.ui_ltx2_video", "ltx2"), ("modules.ui_ideogram", "ideogram")):
        module = sys.modules.get(module_name)
        add(prefix, getattr(module, "_pipeline", None) if module is not None else None)
    return models


def export_loaded_models(destination: str = "") -> str:
    """Export currently loaded models into a timestamped MMGP snapshot."""
    if not memory_management.mmgp_enabled() or not memory_management.feature_enabled("enabled"):
        return "MMGP is not active; launch with --mmgp and enable the MMGP master switch first."

    export_root = Path(destination.strip()) if str(destination or "").strip() else root() / "exports"
    if not export_root.is_absolute():
        from modules import paths

        export_root = Path(paths.script_path) / export_root
    snapshot = export_root / time.strftime("%Y%m%d-%H%M%S")
    models = _loaded_models()
    if not models:
        return "No loaded models were found to export."

    manifest = {"created": time.time(), "models": []}
    for name, model in models:
        filename = f"{_safe_name(name)}.safetensors"
        path = snapshot / filename
        try:
            _save_model(model, path)
            manifest["models"].append({"name": name, "file": filename, "class": model.__class__.__name__})
        except Exception as error:
            logger.warning("Could not export MMGP model %s", name, exc_info=True)
            manifest["models"].append({"name": name, "error": str(error)})

    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    exported = sum("file" in item for item in manifest["models"])
    return html.escape(f"Exported {exported} model(s) to {snapshot}")


def status() -> str:
    cache_root = root()
    files = list(cache_root.rglob("*.safetensors")) if cache_root.is_dir() else []
    total = sum(file.stat().st_size for file in files if file.is_file())
    return f"{len(files)} cached/exported file(s), {total / (1024 * 1024):.1f} MB in {cache_root}"
