"""Opt-in low-RAM component loader backed by MMGP.

This is an alternative loader for native Diffusers pipelines. Forge's
checkpoint loader remains responsible for classic and special-format models.
"""

import json
import logging
import os
from collections.abc import MutableMapping
from pathlib import Path

import torch

from backend import memory_management


logger = logging.getLogger("mmgp_loader")


class MMGPLoaderUnavailable(RuntimeError):
    """The model layout is not safe for the alternate loader."""


def _special_format_reason(keys, metadata=None) -> str | None:
    """Return a reason when a safetensors file needs a format-aware loader."""
    keys = tuple(str(key).lower() for key in keys)
    key_markers = (
        ".qweight",
        ".qzeros",
        ".quant_state",
        ".weight_scale",
        ".weight_zero_point",
        ".absmax",
        ".comfy_quant",
        "scaled_fp8",
        "nunchaku",
    )
    if any(any(marker in key for marker in key_markers) for key in keys):
        return "tensor keys indicate a quantized or format-specific layout"

    metadata_text = " ".join(f"{key}:{value}" for key, value in (metadata or {}).items()).lower()
    metadata_markers = ("gguf", "ggml", "nunchaku", "quanto", "nf4", "fp4", "quantization_map")
    if any(marker in metadata_text for marker in metadata_markers):
        return "metadata indicates a quantized or format-specific layout"
    return None


def _reader_metadata(reader) -> dict:
    metadata = getattr(reader, "metadata", None)
    try:
        return metadata() if callable(metadata) else {}
    except Exception:
        return {}


class _LazySafeTensorStateDict(MutableMapping):
    """Key-only view that reads a tensor only when it is requested."""

    def __init__(self, loader, keys):
        self._loader = loader
        self._keys = list(keys)
        self._key_set = set(self._keys)
        self._values = {}

    def __getitem__(self, key):
        if key not in self._key_set:
            raise KeyError(key)
        if key not in self._values:
            self._values[key] = self._loader.get_tensor(key)
        return self._values[key]

    def __contains__(self, key):
        return key in self._key_set

    def _replace_keys(self, keys):
        self._keys = list(keys)
        self._key_set = set(self._keys)

    def _append_key(self, key):
        self._keys.append(key)
        self._key_set.add(key)

    def __setitem__(self, key, value):
        if key not in self._key_set:
            self._append_key(key)
        self._values[key] = value

    def __delitem__(self, key):
        if key not in self._key_set:
            raise KeyError(key)
        self._keys.remove(key)
        self._key_set.remove(key)
        self._values.pop(key, None)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


class StreamingStateDict(MutableMapping):
    """Lazy safetensors mapping used by LoRA and component loaders."""

    def __init__(self, loader):
        self.loader = loader
        self._keys = list(loader.keys())
        self._key_set = set(self._keys)
        self._values = {}

    def __getitem__(self, key):
        if key not in self._key_set:
            raise KeyError(key)
        if key not in self._values:
            self._values[key] = self.loader.get_tensor(key)
        return self._values[key]

    def __contains__(self, key):
        return key in self._key_set

    def _append_key(self, key):
        self._keys.append(key)
        self._key_set.add(key)

    def __setitem__(self, key, value):
        if key not in self._key_set:
            self._append_key(key)
        self._values[key] = value

    def __delitem__(self, key):
        if key not in self._key_set:
            raise KeyError(key)
        self._keys.remove(key)
        self._key_set.remove(key)
        self._values.pop(key, None)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def close(self):
        loader = self.loader
        self.loader = None
        if loader is not None:
            try:
                loader.close()
            except Exception:
                logger.debug("Streaming safetensors close failed", exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def open_streaming_state_dict(path: str):
    """Open a safetensors file lazily when the optional MMGP path is active."""
    if not memory_management.mmgp_runtime_enabled():
        return None
    if os.path.splitext(str(path))[1].lower() not in {".safetensors", ".sft"}:
        return None

    try:
        from mmgp import safetensors2

        loader = safetensors2.safe_open(
            os.path.abspath(path),
            framework="pt",
            device="cpu",
            writable_tensors=False,
            streaming=True,
        )
        reason = _special_format_reason(loader.keys(), _reader_metadata(loader))
        if reason:
            loader.close()
            return None
        return StreamingStateDict(loader)
    except Exception:
        logger.debug("Streaming safetensors is unavailable for %s", path, exc_info=True)
        return None


class _StreamingForgeCheckpoint:
    """Read a plain Forge safetensors checkpoint one component at a time."""

    def __init__(self, path: str, additional_state_dicts=None):
        if os.path.splitext(path)[1].lower() not in {".safetensors", ".sft"}:
            raise MMGPLoaderUnavailable("Streaming loader only supports safetensors checkpoints")

        try:
            from mmgp import safetensors2
        except ImportError as error:
            raise MMGPLoaderUnavailable("The MMGP safetensors reader is unavailable") from error

        self.path = os.path.abspath(path)
        self.loader = safetensors2.safe_open(
            self.path,
            framework="pt",
            device="cpu",
            writable_tensors=False,
            streaming=True,
        )
        self._extra_sources = {}
        self._has_external_vae = False
        source_keys = tuple(self.loader.keys())
        self._text_state = None
        metadata_reader = getattr(self.loader, "metadata", None)
        metadata = metadata_reader() if callable(metadata_reader) else {}

        if not source_keys:
            self.close()
            raise MMGPLoaderUnavailable("Checkpoint contains no tensors")
        reason = _special_format_reason(source_keys, metadata)
        if reason:
            self.close()
            raise MMGPLoaderUnavailable(reason)

        prefixes = ("model.diffusion_model.", "net.")
        if any(key.startswith(prefix) for prefix in prefixes for key in source_keys):
            self._source_by_key = {key: key for key in source_keys}
        else:
            # Forge's normal preprocess_state_dict adds this prefix to raw DiT files.
            self._source_by_key = {f"model.diffusion_model.{key}": key for key in source_keys}
        self.keys = tuple(self._source_by_key)

        self.state = _LazySafeTensorStateDict(self, self.keys)
        try:
            import huggingface_guess

            self.guess = huggingface_guess.guess(self.state)
        except Exception as error:
            self.close()
            raise MMGPLoaderUnavailable("Could not identify checkpoint layout") from error

        if getattr(self.guess, "nunchaku", False) or self.guess.huggingface_repo in {"nvidia/PiD"}:
            self.close()
            raise MMGPLoaderUnavailable("Model requires format-specific state-dict conversion")

        self.guess.model_type = self.guess.model_type(self.state)
        self.guess.ztsnr = "ztsnr" in self.state
        self._merge_additional_state_dicts(additional_state_dicts or ())
        self.guess.clip_target = self.guess.clip_target(self.state)

    def get_tensor(self, key):
        if key in self.state._values:
            return self.state._values[key]
        if key in self._extra_sources:
            loader, source_key = self._extra_sources[key]
            return loader.get_tensor(source_key)
        return self.loader.get_tensor(self._source_by_key[key])

    def _remove_state_prefix(self, prefix):
        keys = tuple(key for key in self.state if not key.startswith(prefix))
        self.state._replace_keys(keys)
        self.state._values = {key: value for key, value in self.state._values.items() if key in keys}
        self._extra_sources = {key: value for key, value in self._extra_sources.items() if key in keys}

    def _register_streaming_state(self, source, prefix, skip=()):
        self._remove_state_prefix(prefix)
        skipped = set(skip)
        for source_key in source:
            if source_key in skipped:
                continue
            key = prefix + source_key
            self.state._append_key(key)
            self._extra_sources[key] = (source.loader, source_key)

        source.loader = None
        source._values.clear()

    def _integrated_text_adapter_prefixes(self):
        prefixes = getattr(self.guess, "unet_key_prefix", ()) or ()
        return tuple(
            prefix
            for prefix in prefixes
            if any(key.startswith(prefix + "llm_adapter.") for key in self.keys)
        )

    def _merge_streaming_additional(self, path):
        source = open_streaming_state_dict(path)
        if source is None:
            return False

        keys = set(source.keys())
        try:
            reason = _special_format_reason(keys, _reader_metadata(source.loader))
            if reason:
                raise MMGPLoaderUnavailable(reason)

            vae_prefix = (getattr(self.guess, "vae_key_prefix", ()) or (None,))[0]
            if vae_prefix and ({"decoder.conv_in.weight", "decoder.conv1.weight"} & keys or "decoder.middle.0.residual.0.gamma" in keys):
                self._has_external_vae = True
                self._register_streaming_state(source, vae_prefix)
                return True

            text_prefix = (getattr(self.guess, "text_encoder_key_prefix", ()) or (None,))[0]
            if not text_prefix:
                return False

            target_prefix = None
            skip = ()
            if "encoder.block.0.layer.0.SelfAttention.k.weight" in keys and "shared.weight" in keys:
                hidden = source["shared.weight"].shape[0]
                target_prefix = text_prefix + ("umt5xxl" if hidden == 256384 else "t5xxl") + ".transformer."
                skip = ("spiece_model",)
            elif "model.layers.0.post_feedforward_layernorm.weight" in keys:
                target_prefix = text_prefix + "gemma2_2b."
            elif "model.visual.deepstack_merger_list.0.norm.weight" in keys and "model.visual.merger.linear_fc2.weight" in keys:
                target_prefix = text_prefix + "qwen3vl_4b.transformer."
            elif "model.layers.0.self_attn.k_proj.bias" in keys:
                if source["model.layers.0.self_attn.k_proj.bias"].shape[0] == 512:
                    target_prefix = text_prefix + "qwen25_7b."
            elif "model.layers.0.post_attention_layernorm.weight" in keys:
                if "model.layers.0.self_attn.q_norm.weight" in keys:
                    hidden = source["model.layers.0.post_attention_layernorm.weight"].shape[0]
                    size = "06b" if hidden == 1024 else ("4b" if hidden == 2560 else "8b")
                    target_prefix = text_prefix + f"qwen3_{size}.transformer."
                elif source["model.layers.0.post_attention_layernorm.weight"].shape[0] == 3072:
                    target_prefix = text_prefix + "ministral3_3b."
            elif "visual.blocks.0.attn.proj.weight" in keys:
                target_prefix = text_prefix + "qwen25_7b."

            if target_prefix is None:
                return False
            self._register_streaming_state(source, target_prefix, skip=skip)
            return True
        finally:
            source.close()

    def _merge_additional_state_dicts(self, paths):
        if not paths:
            return

        from backend.loader import _is_detectable_standalone_vae, load_torch_file, replace_state_dict
        from backend.state_dict import convert_quantization

        for path in paths:
            if not str(path).lower().endswith((".safetensors", ".sft")):
                self.close()
                raise MMGPLoaderUnavailable("Additional module is not a safetensors file")
            if self._merge_streaming_additional(str(path)):
                continue
            extra, metadata = load_torch_file(path, return_metadata=True)
            extra, metadata = convert_quantization(extra, metadata)
            self._has_external_vae |= _is_detectable_standalone_vae(extra)
            reason = _special_format_reason(extra, metadata)
            if reason:
                self.close()
                raise MMGPLoaderUnavailable(reason)
            replace_state_dict(self.state, extra, self.guess, str(path))
            del extra
            memory_management.soft_empty_cache()
        self.keys = tuple(self.state.keys())
        self.guess._forge_external_vae = self._has_external_vae

    def close(self):
        loader = getattr(self, "loader", None)
        if loader is not None:
            try:
                loader.close()
            except Exception:
                pass
            self.loader = None
        for loader, _ in self._extra_sources.values():
            try:
                loader.close()
            except Exception:
                pass
        self._extra_sources.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _materialize_prefixes(self, prefixes):
        prefixes = tuple(prefixes or ())
        adapter_prefixes = self._integrated_text_adapter_prefixes()
        result = {}
        for key in self.keys:
            prefix = next((prefix for prefix in prefixes if key.startswith(prefix)), None)
            if prefix is not None:
                if prefix in adapter_prefixes and key.startswith(prefix + "llm_adapter."):
                    continue
                result[key[len(prefix) :]] = self.get_tensor(key)
        return result

    def _load_text_state(self):
        if self._text_state is not None:
            return self._text_state

        text_state = {}
        prefixes = tuple(getattr(self.guess, "text_encoder_key_prefix", ()) or ())
        for key in self.keys:
            if any(key.startswith(prefix) for prefix in prefixes):
                text_state[key] = self.get_tensor(key)

        self._text_state = self.guess.process_clip_state_dict(text_state) if text_state else {}
        for prefix in self._integrated_text_adapter_prefixes():
            for key in self.keys:
                if key.startswith(prefix + "llm_adapter."):
                    self._text_state[key[len(prefix) :]] = self.get_tensor(key)
        return self._text_state

    def pop(self, component_name, default=None):
        if component_name in {"scheduler", "tokenizer", "tokenizer_2", "tokenizer_3", "feature_extractor", "safety_checker"}:
            return default

        if component_name == getattr(self.guess, "vae_target", "vae"):
            state_dict = self._materialize_prefixes(getattr(self.guess, "vae_key_prefix", ()))
            return self.guess.process_vae_state_dict(state_dict) if state_dict else default

        if component_name == getattr(self.guess, "unet_target", "unet"):
            state_dict = self._materialize_prefixes(getattr(self.guess, "unet_key_prefix", ()))
            return state_dict or default

        target_prefixes = [prefix for prefix, target in (getattr(self.guess, "clip_target", {}) or {}).items() if target == component_name]
        if target_prefixes:
            text_state = self._load_text_state()
            for prefix in target_prefixes:
                prefix = prefix + "."
                selected = {key[len(prefix) :]: value for key, value in text_state.items() if key.startswith(prefix)}
                if selected:
                    for key in list(text_state):
                        if key.startswith(prefix):
                            del text_state[key]
                    if self._integrated_text_adapter_prefixes():
                        selected.update({key: value for key, value in text_state.items() if key.startswith("llm_adapter.")})
                    return selected

        return default


def open_forge_checkpoint(path: str, additional_state_dicts=None):
    """Open a classic checkpoint through the low-RAM path when it is safe."""
    if not memory_management.mmgp_runtime_enabled():
        return None
    try:
        return _StreamingForgeCheckpoint(path, additional_state_dicts=additional_state_dicts)
    except MMGPLoaderUnavailable:
        return None
    except Exception:
        logger.debug("MMGP low-RAM checkpoint probe failed; using the Forge loader", exc_info=True)
        return None


def quantization_type(name: str | None = None):
    from optimum import quanto

    if name is None:
        from modules.shared import opts
        from modules_forge.mmgp_profiles import get_mmgp_quantization

        _quantize, name = get_mmgp_quantization(opts)
    return getattr(quanto, name, quanto.qint8)


_NON_MODEL_COMPONENTS = {
    "feature_extractor",
    "image_processor",
    "processor",
    "safety_checker",
    "scheduler",
}
_WEIGHT_EXTENSIONS = {".bin", ".ckpt", ".pt", ".safetensors", ".sft"}


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def _component_file_map(model_path: str) -> dict[str, list[str]]:
    root = Path(model_path)
    model_index = _read_json(root / "model_index.json") or {}
    names = [name for name in model_index if not name.startswith("_")]
    result = {}
    for name in names:
        lowered = str(name).lower()
        component_path = root / name
        files = _weight_files(component_path) if component_path.is_dir() else []
        if (
            lowered in _NON_MODEL_COMPONENTS
            or lowered.startswith("tokenizer")
            or lowered.endswith("_tokenizer")
            or not component_path.is_dir()
            or not (component_path / "config.json").is_file()
            or not files
        ):
            continue
        result[name] = files
    return result


def _component_names(model_path: str) -> list[str]:
    return list(_component_file_map(model_path))


def component_names(model_path: str) -> list[str]:
    """Return native component names for one loader-attempt decision."""
    if not (memory_management.mmgp_runtime_enabled() and os.path.isdir(model_path)):
        return []
    return list(component_files(model_path))


def component_files(model_path: str) -> dict[str, list[str]]:
    """Return the validated component index used by the native loader."""
    if not (memory_management.mmgp_runtime_enabled() and os.path.isdir(model_path)):
        return {}
    return _component_file_map(model_path)


def _weight_files(component_path: Path) -> list[str]:
    if not component_path.is_dir():
        return []

    index_files = sorted(component_path.glob("*.index.json"))
    if index_files:
        index = _read_json(index_files[0]) or {}
        weight_map = index.get("weight_map", {})
        files = sorted({component_path / name for name in weight_map.values()})
        if files and all(path.is_file() for path in files):
            return [str(path) for path in files]

    files = [path for path in component_path.iterdir() if path.is_file() and path.suffix.lower() in _WEIGHT_EXTENSIONS]
    return [str(path) for path in sorted(files)]


def _model_class(component_path: Path):
    config = _read_json(component_path / "config.json")
    if not config:
        raise MMGPLoaderUnavailable(f"Missing component config: {component_path / 'config.json'}")

    names = config.get("architectures") or []
    if not names and config.get("_class_name"):
        names = [config["_class_name"]]
    for name in names:
        for library in ("diffusers", "transformers"):
            try:
                module = __import__(library)
                model_class = getattr(module, name, None)
                if model_class is not None and hasattr(model_class, "from_config"):
                    return model_class
            except (ImportError, AttributeError):
                continue
    raise MMGPLoaderUnavailable(f"Could not resolve a model class for {component_path}")


def component_kind(name: str, model=None) -> str:
    """Classify a module without treating every unknown component as a denoiser."""
    identifiers = [str(name).lower()]
    if model is not None:
        model_class = model if isinstance(model, type) else model.__class__
        identifiers.append(model_class.__name__.lower())
    identifier = " ".join(identifiers)

    if any(token in identifier for token in ("controlnet", "control_net")):
        return "controlnet"
    if any(token in identifier for token in ("image_encoder", "vision", "audio_encoder", "feature_extractor")):
        return "auxiliary"
    if any(token in identifier for token in ("vae", "autoencoder", "vocoder", "decoder")):
        return "vae"
    if any(token in identifier for token in ("text_encoder", "tokenizer", "clip", "t5", "llama", "gemma", "qwen")):
        return "text_encoder"
    if any(token in identifier for token in ("transformer", "unet", "denoiser", "diffusion_model", "diffusion", "prior")):
        return "unet"
    return "auxiliary"


def has_llm_adapter(model) -> bool:
    """Return whether a text encoder embeds an LLM adapter module."""
    return bool(model and not isinstance(model, type) and any("llmadapter" in module.__class__.__name__.lower() for module in model.modules()))


_LARGE_TEXT_ENCODER_MARKERS = ("t5", "umt5", "llama", "llm", "gemma", "qwen", "mistral", "phi")


def is_large_text_encoder(model, name: str = "") -> bool:
    """Match MMGP's large language-encoder quantization scope, not CLIP."""
    if component_kind(name, model) != "text_encoder" or has_llm_adapter(model):
        return False

    model_class = model if isinstance(model, type) else model.__class__
    identifiers = {
        str(name).lower(),
        model_class.__name__.lower(),
        getattr(model_class, "__module__", "").lower(),
    }
    return any(marker in identifier for marker in _LARGE_TEXT_ENCODER_MARKERS for identifier in identifiers)


def _can_quantize_forge_component(model, name: str, quantize: bool) -> bool:
    return bool(quantize and is_large_text_encoder(model, name))


def _component_options(name: str, dtype: torch.dtype, model=None) -> dict:
    from modules.shared import opts
    from modules_forge.mmgp_profiles import get_mmgp_quantization

    quantize, _quantization_type = get_mmgp_quantization(opts)
    kind = component_kind(name, model)
    pinned = bool(memory_management.feature_enabled("pinned_memory") and kind in memory_management.MEMORY_PINNED_COMPONENTS)
    return {
        "do_quantize": quantize and (kind == "unet" or is_large_text_encoder(model, name)),
        "quantizationType": quantization_type(_quantization_type),
        "pinToMemory": pinned,
        "partialPinning": bool(getattr(opts, "forge_memory_partial_pinning_enabled", False)),
        "default_dtype": dtype,
        "verboseLevel": 0,
    }


def load_component(model_path: str, name: str, dtype: torch.dtype, files: list[str] | None = None):
    component_path = Path(model_path) / name
    if not component_path.is_dir():
        raise MMGPLoaderUnavailable(f"Missing component directory: {component_path}")

    files = _weight_files(component_path) if files is None else files
    if not files:
        raise MMGPLoaderUnavailable(f"No supported weight files found in {component_path}")

    model_class = _model_class(component_path)
    try:
        from mmgp import offload
    except ImportError as error:
        raise MMGPLoaderUnavailable("The mmgp package is unavailable") from error

    config_path = component_path / "config.json"
    from backend import mmgp_cache

    cached = mmgp_cache.cached_component_path(model_path, name, files, dtype)
    load_files = [str(cached)] if cached is not None else files
    logger.info("MMGP low-RAM loading native component %s%s", name, " from persistent cache" if cached else "")
    try:
        return offload.fast_load_transformers_model(
            load_files,
            modelClass=model_class,
            forcedConfigPath=str(config_path),
            **_component_options(name, dtype, model_class),
        )
    except Exception:
        if cached is None:
            raise
        logger.warning("Invalid MMGP component cache for %s; retrying from source weights", name, exc_info=True)
        cached.unlink(missing_ok=True)
        return offload.fast_load_transformers_model(
            files,
            modelClass=model_class,
            forcedConfigPath=str(config_path),
            **_component_options(name, dtype, model_class),
        )


def load_pipeline(pipeline_class, model_path: str, pipeline_kwargs: dict, dtype: torch.dtype, component_names: list[str] | dict[str, list[str]] | None = None):
    """Load a native Diffusers pipeline through MMGP component loading.

    Every component must have a resolvable config and supported weight files.
    A caller should catch ``MMGPLoaderUnavailable`` and use its normal loader.
    """
    if isinstance(component_names, dict):
        file_map = component_names
        names = list(component_names)
    else:
        file_map = _component_file_map(model_path) if component_names is None else {}
        names = list(file_map) if component_names is None else component_names
    if not names:
        raise MMGPLoaderUnavailable(f"No supported Diffusers components found in {model_path}")

    loaded = {}
    component_files = {
        name: file_map.get(name) or _weight_files(Path(model_path) / name)
        for name in names
        if not (name in pipeline_kwargs and isinstance(pipeline_kwargs[name], torch.nn.Module))
    }
    try:
        for name, files in component_files.items():
            loaded[name] = load_component(model_path, name, dtype, files=files)

        kwargs = dict(pipeline_kwargs)
        kwargs.update(loaded)
        pipeline = pipeline_class.from_pretrained(model_path, **kwargs)
        from backend import mmgp_cache

        mmgp_cache.cache_pipeline_components(pipeline, model_path, component_files, dtype)
        return pipeline
    except Exception:
        for module in loaded.values():
            del module
        memory_management.soft_empty_cache()
        raise


def can_attempt(model_path: str, names: list[str] | None = None) -> bool:
    names = component_names(model_path) if names is None else names
    return bool(memory_management.mmgp_runtime_enabled() and os.path.isdir(model_path) and names)


def load_forge_component(model, state_dict: dict, name: str, ignore_start: str | None = None, ignore_errors=()) -> bool:
    """Try MMGP's state-dict loader for a Forge-built component.

    Forge still constructs the architecture and owns all format-specific
    conversion. MMGP only receives the already-normalized state dict here.
    """
    if not memory_management.mmgp_runtime_enabled():
        return False
    if not isinstance(state_dict, dict) or not state_dict:
        return False
    tensor_count = 0
    for value in state_dict.values():
        if not isinstance(value, torch.Tensor):
            continue
        tensor_count += 1
        if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
    if tensor_count == 0:
        return False

    try:
        from mmgp import offload

        from modules.shared import opts
        from modules_forge.mmgp_profiles import get_mmgp_quantization

        dtype = getattr(model, "storage_dtype", None)
        if not isinstance(dtype, torch.dtype):
            dtype = next((parameter.dtype for parameter in model.parameters() if parameter.device.type != "meta"), torch.bfloat16)
        dtype = memory_management.mmgp_compute_dtype(model, fallback=dtype)
        kind = component_kind(name, model)
        quantize, _quantization_type = get_mmgp_quantization(opts)
        pinned = bool(memory_management.feature_enabled("pinned_memory") and kind in memory_management.MEMORY_PINNED_COMPONENTS)
        ignored = set(ignore_errors or ())
        filtered_state_dict = {
            key: value
            for key, value in state_dict.items()
            if not (ignore_start and key.startswith(ignore_start)) and key not in ignored
        }
        target_state_dict = model.state_dict()
        for key, value in filtered_state_dict.items():
            target = target_state_dict.get(key)
            if target is not None and hasattr(value, "shape") and tuple(target.shape) != tuple(value.shape):
                logger.debug("MMGP state-dict shape mismatch for %s.%s; using Forge loader", name, key)
                return False
        missing = [
            key
            for key in target_state_dict
            if key not in filtered_state_dict
            and key not in ignored
            and not (ignore_start and key.startswith(ignore_start))
        ]
        if missing:
            logger.debug("MMGP state-dict missing %s keys for %s; using Forge loader", len(missing), name)
            return False
        do_quantize = quantize and (kind == "unet" or _can_quantize_forge_component(model, name, quantize))
        offload.load_model_data(
            model,
            filtered_state_dict,
            do_quantize=do_quantize,
            quantizationType=quantization_type(_quantization_type),
            pinToMemory=pinned,
            partialPinning=bool(getattr(opts, "forge_memory_partial_pinning_enabled", False)),
            default_dtype=dtype,
            verboseLevel=0,
            ignore_missing_keys=bool(ignore_start or ignored),
        )
        logger.info("MMGP low-RAM state-dict loading active for Forge %s", name)
        return True
    except Exception:
        logger.debug("MMGP state-dict loading is incompatible with Forge %s; using Forge loader", name, exc_info=True)
        return False
