"""Opt-in persistent exports and component cache for the MMGP path."""

import html
import hashlib
import json
import logging
import os
import re
import stat as stat_module
import tempfile
import time
from pathlib import Path

import torch

from backend import memory_management, mmgp_loader


logger = logging.getLogger("mmgp_cache")
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")
_STATUS_MANIFEST = ".status.json"


def _options():
    from modules.shared import opts

    return opts


def root() -> Path:
    from modules import paths

    value = str(getattr(_options(), "forge_memory_cache_directory", "models/MMGP-cache") or "models/MMGP-cache")
    path = Path(value)
    return path if path.is_absolute() else Path(paths.script_path) / path


def _read_status_manifest(cache_root: Path):
    try:
        data = json.loads((cache_root / _STATUS_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    entries = data.get("files") if isinstance(data, dict) else None
    return entries if isinstance(entries, dict) else None


def _scan_status(cache_root: Path):
    entries = {}
    if cache_root.is_dir():
        for file in cache_root.rglob("*.safetensors"):
            try:
                info = file.stat()
                if stat_module.S_ISREG(info.st_mode):
                    entries[str(file.relative_to(cache_root))] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
            except OSError:
                continue
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        (cache_root / _STATUS_MANIFEST).write_text(json.dumps({"files": entries}), encoding="utf-8")
    except OSError:
        logger.debug("Could not write MMGP cache status manifest", exc_info=True)
    return entries


def _record_status_files(files):
    cache_root = root()
    entries = _read_status_manifest(cache_root)
    if entries is None:
        entries = _scan_status(cache_root)
    for file in files:
        try:
            path = Path(file)
            info = path.stat()
            if stat_module.S_ISREG(info.st_mode):
                entries[str(path.relative_to(cache_root))] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
        except (OSError, ValueError):
            continue
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        (cache_root / _STATUS_MANIFEST).write_text(json.dumps({"files": entries}), encoding="utf-8")
    except OSError:
        logger.debug("Could not update MMGP cache status manifest", exc_info=True)


def enabled() -> bool:
    return bool(
        memory_management.mmgp_runtime_enabled()
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
    from modules_forge.mmgp_profiles import get_mmgp_quantization

    quantize, quantization_type = get_mmgp_quantization(options)
    parts.extend(
        (
            str(quantize),
            str(quantization_type),
        )
    )
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]


def cached_component_path(model_path: str, name: str, files, dtype: torch.dtype) -> Path | None:
    if not enabled():
        return None
    path = root() / "components" / _component_key(model_path, name, files, dtype) / f"{_safe_name(name)}.safetensors"
    try:
        file_info = path.stat()
        return path if stat_module.S_ISREG(file_info.st_mode) and file_info.st_size > 0 else None
    except OSError:
        return None


def _save_model(model, path: Path, config_path: Path | None = None):
    from mmgp import offload

    options = _options()
    from modules_forge.mmgp_profiles import get_mmgp_quantization

    quantize, _quantization_type = get_mmgp_quantization(options)
    kwargs = {
        "do_quantize": quantize,
        "quantizationType": mmgp_loader.quantization_type(_quantization_type),
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
    cache_root = root() / "components"
    for name, files in component_files.items():
        model = getattr(pipeline, name, None)
        if not isinstance(model, torch.nn.Module) or not files:
            continue
        path = cache_root / _component_key(model_path, name, files, dtype) / f"{_safe_name(name)}.safetensors"
        try:
            # The loader already used this exact cache key when it loaded the
            # component. Avoid serializing it again after a cache hit.
            file_info = path.stat()
            if stat_module.S_ISREG(file_info.st_mode) and file_info.st_size > 0:
                saved.append(path)
                continue
            config_path = Path(model_path) / name / "config.json"
            _save_model(model, path, config_path=config_path)
            saved.append(path)
            logger.info("MMGP cached native component %s at %s", name, path)
        except Exception:
            logger.warning("Could not cache native MMGP component %s", name, exc_info=True)
    _record_status_files(saved)
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
    try:
        from backend import mmgp_native

        add("native", mmgp_native.active_pipeline())
    except Exception:
        logger.debug("Could not inspect the active native MMGP pipeline", exc_info=True)
    return models


def export_loaded_models(destination: str = "") -> str:
    """Export currently loaded models into a timestamped MMGP snapshot."""
    if not memory_management.mmgp_runtime_enabled():
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
    _record_status_files([Path(snapshot) / item["file"] for item in manifest["models"] if "file" in item])
    exported = sum("file" in item for item in manifest["models"])
    return html.escape(f"Exported {exported} model(s) to {snapshot}")


def status() -> str:
    cache_root = root()
    entries = _read_status_manifest(cache_root)
    if entries is None:
        entries = _scan_status(cache_root)

    stale = False
    total = 0
    valid_files = 0
    for relative, expected in entries.items():
        try:
            file_info = (cache_root / relative).stat()
            if not stat_module.S_ISREG(file_info.st_mode) or file_info.st_size != expected.get("size") or file_info.st_mtime_ns != expected.get("mtime_ns"):
                stale = True
                break
            total += file_info.st_size
            valid_files += 1
        except (OSError, TypeError, AttributeError):
            stale = True
            break
    if stale:
        entries = _scan_status(cache_root)
        valid_files = len(entries)
        total = sum(item.get("size", 0) for item in entries.values())
    return f"{valid_files} cached/exported file(s), {total / (1024 * 1024):.1f} MB in {cache_root}"
