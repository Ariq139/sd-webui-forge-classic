import importlib.util
import json
import logging
import os
import threading
from collections import OrderedDict
from functools import wraps

import torch

from modules_forge import native_runtime


ATTENTION_BACKEND_CHOICES = ("automatic", "native", "sage", "flash")
OFFLOAD_CHOICES = ("none", "model", "group", "sequential")


def require_dependency(name, feature):
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"{feature} requires optional dependency '{name}' in the WebUI environment.")


def native_quantization_kwargs(model_path, mode, dtype, device, offload, denoisers):
    """Apply only explicit quantization; do not silently dequantize or download weights."""
    from modules_forge.native_recipes import QUANTIZATION_MODES

    if mode not in QUANTIZATION_MODES:
        raise ValueError("Unknown native quantization mode.")
    prequantized = []
    for name in (*denoisers, "text_encoder"):
        path = os.path.join(model_path, name, "config.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as file:
                quant = json.load(file).get("quantization_config") or {}
            if quant:
                prequantized.append(name)
                method = str(quant.get("quant_method", "")).lower()
                if "bitsandbytes" in method or quant.get("load_in_4bit") or quant.get("load_in_8bit"):
                    require_dependency("bitsandbytes", "Prequantized native checkpoint")
    if mode == "Checkpoint":
        return {}
    if prequantized:
        raise ValueError("Checkpoint already contains quantized components; select Checkpoint mode.")
    if device.type != "cuda" or offload not in ("model", "none"):
        raise ValueError("Experimental on-load NF4 requires CUDA and model or none offload. Prefer a prequantized checkpoint on low VRAM.")
    require_dependency("bitsandbytes", "NF4 quantization")
    require_dependency("accelerate", "NF4 quantization")
    from diffusers import BitsAndBytesConfig, PipelineQuantizationConfig

    options = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    mapping = {name: BitsAndBytesConfig(**options) for name in denoisers}
    if mode.endswith("text encoder"):
        from transformers import BitsAndBytesConfig as TextBitsAndBytesConfig
        mapping["text_encoder"] = TextBitsAndBytesConfig(**options)
    return {"quantization_config": PipelineQuantizationConfig(quant_mapping=mapping)}


def load_native_pipeline(pipeline_class, model_path: str, pipeline_kwargs: dict, dtype: torch.dtype, logger: logging.Logger, label: str):
    # The model selector accepts complete local pipeline directories only.
    # Missing weights must never trigger an implicit Hub download.
    return pipeline_class.from_pretrained(model_path, **dict(pipeline_kwargs, local_files_only=True))


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
    candidates = ("sage", "flash", "native") if requested == "automatic" and device.type == "cuda" else ("native",) if requested == "automatic" else (requested,)
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

    if offload == "group":
        if not hasattr(pipe, "enable_group_offload"):
            logger.warning("Group offload is unavailable; falling back to sequential offload")
            offload = "sequential"
        else:
            # Failed setup can leave partially installed hooks. Do not stack
            # another offload policy or silently load the pipeline to CUDA.
            _enable_group_offload(pipe, device)
            return "group"

    if offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
        return "sequential"
    if offload == "model":
        pipe.enable_model_cpu_offload(device=device)
        return "model"

    raise ValueError(f"Unsupported native offload mode: {offload}")


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
    if getattr(pipe, "unconditional_transformer", None) is not None:
        # Ideogram alternates denoisers and accesses encoder submodules
        # directly; the generic linear/sequential hooks are insufficient.
        import diffusers
        if actual_offload in ("model", "sequential") and diffusers.__version__ != "0.39.0":
            raise RuntimeError("Ideogram offload adapters require Diffusers 0.39.0.")
        if actual_offload == "model":
            native_runtime.install_alternating_offload(pipe)
        elif actual_offload == "sequential":
            from accelerate.hooks import remove_hook_from_module
            native_runtime.keep_sequential_prompt_on_cpu(pipe, remove_hook_from_module, device)
    compiled = compile_denoisers(pipe, compile_enabled, device, actual_offload, logger)
    pipe._forge_runtime = dict(attention_backend=backend, offload=actual_offload, compiled=compiled)
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
        return value.to(device, copy=True)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


class PromptEmbeddingCache:
    def __init__(self, max_entries: int = 4, max_bytes: int = 128 * 1024**2):
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self.bytes_used = 0
        self._values = OrderedDict()
        self._lock = threading.RLock()

    def clear(self):
        with self._lock:
            self._values.clear()
            self.bytes_used = 0

    def get_or_compute(self, key, device, factory):
        with self._lock:
            if key in self._values:
                value = self._values.pop(key)
                self._values[key] = value
                return _to_device(value, device)

        with torch.no_grad():
            computed = factory()
        size = _tensor_bytes(computed)
        if size > self.max_bytes or not size:
            return computed
        value = _to_cpu(computed)

        with self._lock:
            previous = self._values.pop(key, None)
            if previous is not None:
                self.bytes_used -= _tensor_bytes(previous)
            while self._values and (len(self._values) >= self.max_entries or self.bytes_used + size > self.max_bytes):
                _, evicted = self._values.popitem(last=False)
                self.bytes_used -= _tensor_bytes(evicted)
            self._values[key] = value
            self.bytes_used += size
        return computed


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
    logger.info("Native pipeline prompt embedding cache enabled (maximum 4 entries / 128 MiB of tensor payload)")
    return True
