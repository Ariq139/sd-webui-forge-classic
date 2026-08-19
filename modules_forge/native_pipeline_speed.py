import logging
import threading
from collections import OrderedDict
from collections.abc import MutableMapping
from functools import wraps
from pathlib import Path

import torch


class _LazySafetensorStateDict(MutableMapping):
    def __init__(self, files):
        from safetensors import safe_open

        self._readers = [safe_open(str(path), framework="pt", device="cpu") for path in files]
        self._keys = []
        self._key_set = set()
        self._sources = {}
        self._values = {}
        for reader in self._readers:
            for key in reader.keys():
                if key not in self._key_set:
                    self._keys.append(key)
                    self._key_set.add(key)
                self._sources.setdefault(key, reader)

    def __getitem__(self, key):
        if key not in self._sources and key not in self._values:
            raise KeyError(key)
        if key not in self._values:
            self._values[key] = self._sources[key].get_tensor(key)
        return self._values[key]

    def __setitem__(self, key, value):
        if key not in self._key_set:
            self._keys.append(key)
            self._key_set.add(key)
        self._values[key] = value
        self._sources.pop(key, None)

    def __delitem__(self, key):
        if key not in self._sources and key not in self._values:
            raise KeyError(key)
        self._sources.pop(key, None)
        self._values.pop(key, None)
        self._keys.remove(key)
        self._key_set.remove(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def close(self):
        for reader in self._readers:
            close = getattr(reader, "close", None)
            if close is not None:
                close()
            else:
                reader.__exit__(None, None, None)
        self._readers.clear()
        self._keys.clear()
        self._key_set.clear()
        self._sources.clear()
        self._values.clear()


def _load_native_quantized_component(model_path: str, component_name: str, dtype: torch.dtype, logger):
    """Build a Diffusers component with Forge's native quantized operations."""
    from backend import mmgp_loader
    from backend.state_dict import convert_quantization, detect_quantization, load_state_dict

    component_path = Path(model_path) / component_name
    files = mmgp_loader._weight_files(component_path)
    if not files or not all(str(path).lower().endswith((".safetensors", ".sft")) for path in files):
        return None

    lazy_state_dict = _LazySafetensorStateDict(files)
    special_keys = set(lazy_state_dict)
    if not any(key.endswith((".comfy_quant", ".weight_scale_2", "scaled_fp8")) for key in special_keys):
        lazy_state_dict.close()
        return None

    try:
        state_dict, _ = convert_quantization(lazy_state_dict, {})
        quant_config = detect_quantization(state_dict, is_unet=component_name in {"transformer", "unet"})
        if quant_config is None:
            return None

        config = mmgp_loader._read_json(component_path / "config.json")
        try:
            model_class = mmgp_loader._model_class(component_path)
        except mmgp_loader.MMGPLoaderUnavailable:
            model_class = None
        if not config or model_class is None:
            return None

        from backend.operations import using_forge_operations

        try:
            from transformers.initialization import no_init_weights
        except ImportError:
            from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            with using_forge_operations(
                device=torch.device("cpu"),
                dtype=dtype,
                manual_cast_enabled=True,
                extra_dtype=dict(quant_config),
            ):
                model = model_class.from_config(config)

        load_state_dict(model, state_dict, log_name=f"Native {component_name}")
        model.eval()
        logger.info("Loaded native quantized component %s with Forge operations", component_name)
        return model
    finally:
        lazy_state_dict.close()


ATTENTION_BACKEND_CHOICES = ("automatic", "native", "sage", "flash")
OFFLOAD_CHOICES = ("none", "model", "group", "sequential")


def load_native_pipeline(pipeline_class, model_path: str, pipeline_kwargs: dict, dtype: torch.dtype, logger: logging.Logger, label: str):
    from backend import mmgp_loader

    pipeline_kwargs = dict(pipeline_kwargs)
    native_quantized = _load_native_quantized_component(model_path, "transformer", dtype, logger)
    if native_quantized is not None:
        pipeline_kwargs["transformer"] = native_quantized

    component_files = mmgp_loader.component_files(model_path)
    component_names = list(component_files)
    if native_quantized is None and mmgp_loader.can_attempt(model_path, component_names):
        try:
            pipeline = mmgp_loader.load_pipeline(pipeline_class, model_path, pipeline_kwargs, dtype, component_names=component_files)
            logger.info("%s native pipeline loaded through MMGP's low-RAM component loader", label)
            return pipeline
        except mmgp_loader.MMGPLoaderUnavailable:
            logger.info("%s model layout is not supported by the MMGP loader; using Diffusers loader", label)
        except Exception:
            logger.exception("MMGP native loading failed for %s; using Diffusers loader", label)

    return pipeline_class.from_pretrained(model_path, **pipeline_kwargs)


def resolve_vae_tiling_mode(auto: bool, force: bool) -> str:
    return "automatic" if auto else "enabled" if force else "disabled"


def _auto_tile_vae(device: torch.device, width: int, height: int, frames: int) -> bool:
    width, height, frames = int(width or 0), int(height or 0), int(frames or 1)
    if device.type != "cuda":
        return width > 1024 or height > 1024 or frames > 121

    try:
        free_memory, _ = torch.cuda.mem_get_info(device)
    except (RuntimeError, AttributeError):
        return True

    eight_gb = 8 * 1024**3
    twelve_gb = 12 * 1024**3
    if free_memory < eight_gb:
        return True
    if frames > 121:
        return True
    if frames > 1:
        return (width > 768 or height > 512) and free_memory < twelve_gb
    return (width > 1024 or height > 1024) and free_memory < twelve_gb


def configure_vae_tiling(vae, mode: str, device: torch.device, width: int, height: int, frames: int = 1) -> bool:
    if not vae or not hasattr(vae, "enable_tiling"):
        return False

    enabled = mode == "enabled" or (mode == "automatic" and _auto_tile_vae(device, width, height, frames))
    method = "enable_tiling" if enabled else "disable_tiling"
    getattr(vae, method)()
    return enabled


def _attention_components(pipe):
    return [
        component
        for component in (getattr(pipe, "transformer", None), getattr(pipe, "unconditional_transformer", None))
        if component is not None and hasattr(component, "set_attention_backend")
    ]


def configure_attention(pipe, requested: str, device: torch.device, logger: logging.Logger) -> str:
    requested = str(requested or "automatic").strip().lower()
    candidates = ("sage", "flash", "native") if requested == "automatic" and device.type == "cuda" else (requested,)
    if requested not in ATTENTION_BACKEND_CHOICES:
        candidates = ("native",)

    components = _attention_components(pipe)
    for backend in candidates:
        try:
            for component in components:
                component.set_attention_backend(backend)
            if components:
                logger.info("Native pipeline attention backend: %s", backend)
            return backend
        except Exception:
            logger.debug("Attention backend %s is unavailable", backend, exc_info=True)

    if requested != "native":
        for component in components:
            component.set_attention_backend("native")
    return "native"


def _enable_group_offload(pipe, device: torch.device):
    pipe.enable_group_offload(
        onload_device=device,
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=2,
        non_blocking=True,
        use_stream=True,
    )


def configure_offload(pipe, device: torch.device, offload: str, logger: logging.Logger) -> str:
    if device.type != "cuda" or offload == "none":
        pipe.to(device)
        return "none"

    if offload == "group" and hasattr(pipe, "enable_group_offload"):
        try:
            _enable_group_offload(pipe, device)
            return "group"
        except Exception:
            logger.warning("Group offload is unavailable; falling back to model offload", exc_info=True)
            offload = "model"

    if offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
        return "sequential"
    if offload == "model":
        pipe.enable_model_cpu_offload(device=device)
        return "model"

    pipe.to(device)
    return "none"


def compile_denoisers(pipe, enabled: bool, device: torch.device, offload: str, logger: logging.Logger) -> bool:
    if not enabled:
        return False
    if device.type != "cuda" or offload != "none" or not hasattr(torch, "compile"):
        logger.warning("Denoiser compilation requires CUDA and CPU offload set to none; disabled for this run")
        return False

    for name in ("transformer", "unconditional_transformer"):
        module = getattr(pipe, name, None)
        if module is None or getattr(module, "_forge_native_compiled", False):
            continue
        compiled = torch.compile(module, mode="max-autotune", dynamic=True)
        compiled._forge_native_compiled = True
        setattr(pipe, name, compiled)
    logger.info("Native pipeline denoiser compilation enabled; the first generation will compile the graph")
    return True


def configure_pipeline(pipe, device: torch.device, offload: str, attention: str, compile_enabled: bool, logger: logging.Logger):
    backend = configure_attention(pipe, attention, device, logger)
    actual_offload = configure_offload(pipe, device, offload, logger)
    compiled = compile_denoisers(pipe, compile_enabled, device, actual_offload, logger)
    return backend, actual_offload, compiled


def _contains_tensor(value) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _freeze(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


class PromptEmbeddingCache:
    def __init__(self, max_entries: int = 4):
        self.max_entries = max(1, int(max_entries))
        self._values = OrderedDict()
        self._lock = threading.RLock()

    def clear(self):
        with self._lock:
            self._values.clear()

    def get_or_compute(self, key, device, factory):
        with self._lock:
            if key in self._values:
                value = self._values.pop(key)
                self._values[key] = value
                return _to_device(value, device)

        with torch.no_grad():
            value = _to_cpu(factory())

        with self._lock:
            self._values[key] = value
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)
        return _to_device(value, device)


def configure_prompt_cache(pipe, enabled: bool, logger: logging.Logger) -> bool:
    current = getattr(pipe, "_forge_prompt_cache_state", None)
    if not enabled:
        if current is not None:
            pipe.encode_prompt = current["original"]
            delattr(pipe, "_forge_prompt_cache_state")
        return False
    if current is not None:
        return True

    original = pipe.encode_prompt
    cache = PromptEmbeddingCache()

    @wraps(original)
    def cached_encode_prompt(*args, **kwargs):
        if _contains_tensor(args) or _contains_tensor(kwargs):
            return original(*args, **kwargs)

        prompt = kwargs.get("prompt", args[0] if args else None)
        device = kwargs.get("device")
        if prompt is None or device is None:
            return original(*args, **kwargs)

        key = _freeze((args, kwargs))
        return cache.get_or_compute(key, device, lambda: original(*args, **kwargs))

    pipe.encode_prompt = cached_encode_prompt
    pipe._forge_prompt_cache_state = {"original": original, "cache": cache}
    logger.info("Native pipeline prompt embedding cache enabled (maximum 4 entries)")
    return True
