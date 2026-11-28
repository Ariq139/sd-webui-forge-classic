"""Native pipeline lifecycle adapters for the pinned Diffusers/Accelerate APIs."""
from contextlib import contextmanager
import weakref


def _cpu_hook(pipe, module):
    for hook in getattr(pipe, "_all_hooks", ()):
        if hook.model is module:
            return hook
    raise RuntimeError("Missing native model CPU-offload hook.")


def install_alternating_offload(pipe):
    """Close the offload cycle: unconditional -> conditional on the next step.

    Diffusers rebuilds its Accelerate hooks after every request, so never
    capture a particular UserCpuOffloadHook in the forward-pre-hook.
    """
    conditional = pipe.transformer
    unconditional = pipe.unconditional_transformer
    _cpu_hook(pipe, conditional)
    _cpu_hook(pipe, unconditional)
    if getattr(pipe, "_forge_alternating_handle", None) is not None:
        return
    owner = weakref.ref(pipe)

    def before_conditional(module, inputs):
        current = owner()
        if current is not None:
            other = current.unconditional_transformer
            if next(other.parameters()).device.type != "cpu":
                _cpu_hook(current, other).offload()

    pipe._forge_alternating_handle = conditional.register_forward_pre_hook(before_conditional)


def _prompt_components(pipe):
    return [component for component in (pipe.text_encoder, getattr(pipe, "prompt_enhancer_head", None)) if component is not None]


def _require_cpu_component(component):
    if any(p.device.type != "cpu" for p in component.parameters()):
        raise RuntimeError("Ideogram prompt components must have materialized CPU weights.")


def keep_sequential_prompt_on_cpu(pipe, remove_hook, device):
    """The pinned encoder reads text_encoder.device and calls its children.

    Leave prompt components on CPU rather than allowing input IDs to become
    meta tensors. The denoisers/VAE retain sequential offload.
    """
    for component in _prompt_components(pipe):
        remove_hook(component, recurse=True)
        _require_cpu_component(component)
    if pipe._execution_device != device:
        raise RuntimeError("Ideogram sequential offload did not preserve the denoising execution device.")


@contextmanager
def ideogram_enhancement_device(pipe, remove_hook):
    """The enhancer grafts the encoder and LM head, bypassing head hooks."""
    mode = pipe._forge_runtime["offload"]
    if mode == "none":
        yield pipe._execution_device
        return
    if mode not in ("model", "sequential"):
        raise ValueError("Ideogram prompt enhancement requires model/sequential CPU offload or no offload; group hooks are not supported for the grafted enhancer.")
    components = _prompt_components(pipe)
    if any(getattr(c, "is_quantized", False) or getattr(c, "hf_quantizer", None) is not None for c in components):
        raise ValueError("CPU prompt enhancement with quantized prompt components is not supported.")
    execution = pipe._execution_device
    if mode == "model":
        for component in components:
            _cpu_hook(pipe, component).offload()
            remove_hook(component, recurse=True)
    try:
        for component in components:
            _require_cpu_component(component)
        yield pipe.text_encoder.device
    finally:
        if mode == "model":
            pipe.enable_model_cpu_offload(device=execution)


def refinement_geometry(before, after, audio, *, width, height, frames, spatial_ratio, temporal_ratio):
    """Reject incorrect latent upsamplers before stage two silently infers size."""
    expected = (1, (frames - 1) // temporal_ratio + 1, height // spatial_ratio, width // spatial_ratio)
    shape = tuple(before.shape)
    if len(shape) != 5 or (shape[0], *shape[2:]) != expected:
        raise ValueError("Stage-one LTX video latent geometry does not match the request.")
    if tuple(after.shape) != (*shape[:3], shape[3] * 2, shape[4] * 2):
        raise ValueError("Selected LTX upsampler must double spatial dimensions without changing temporal/channel dimensions.")
    if audio is None or len(audio.shape) != 4 or audio.shape[0] != shape[0]:
        raise ValueError("Stage-one LTX audio latents do not match the video batch.")


def release_runtime(pipe):
    """Break Forge-owned reference cycles without materializing offloaded weights."""
    handle = getattr(pipe, "_forge_alternating_handle", None)
    if handle is not None:
        handle.remove()
        del pipe._forge_alternating_handle
    state = getattr(pipe, "_forge_prompt_cache_state", None)
    if state is not None:
        pipe.encode_prompt = state["original"]
        state["cache"].clear()
        del pipe._forge_prompt_cache_state
    # The optional enhancer shares the encoder/head and may retain hooks.
    for name in ("_prompt_enhancer", "_caption_logits_processor"):
        if hasattr(pipe, name):
            setattr(pipe, name, None)


def require_pipeline_directory(path, marker):
    """A local full pipeline, not an upsampler/decoder repo or a remote model ID."""
    import json
    import os

    classes = {
        "LTX2": ("LTX2Pipeline", "LTX2ImageToVideoPipeline", "LTX2ConditionPipeline"),
        "Ideogram4": ("Ideogram4Pipeline",),
    }
    required = {"scheduler", "vae", "text_encoder", "tokenizer", "transformer"}
    required.update({"audio_vae", "vocoder", "connectors"} if marker == "LTX2" else {"unconditional_transformer"})
    try:
        if not path or not os.path.isdir(path):
            raise ValueError("not a local directory")
        with open(os.path.join(path, "model_index.json"), encoding="utf-8") as handle:
            index = json.load(handle)
        if not isinstance(index, dict) or index.get("_class_name") not in classes[marker]:
            raise ValueError("wrong pipeline class")
        missing = sorted(name for name in required if not isinstance(index.get(name), (list, tuple)) or len(index[name]) != 2 or not all(index[name]))
        if missing:
            raise ValueError("missing components: " + ", ".join(missing))
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Select a complete local {marker} Diffusers pipeline directory ({error}).") from error
    return os.path.abspath(path)
