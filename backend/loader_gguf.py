# https://github.com/city96/ComfyUI-GGUF/blob/main/loader.py
# (c) City96


import logging
import os
import warnings

import torch


def dequantize(p: torch.nn.Parameter, dtype: torch.dtype) -> torch.Tensor:
    from backend.operations_gguf import dequantize_tensor

    gguf_cls = getattr(p, "gguf_cls", None)
    if gguf_cls is not None:
        gguf_cls.bake(p)

    return dequantize_tensor(p).to(dtype=dtype)


def gguf_remapping(state_dict: dict[str, torch.Tensor], architecture: str | None = None) -> dict[str, torch.Tensor]:
    architecture = (architecture or "").lower()

    if "enc.blk.0.attn_k.weight" in state_dict:
        gguf_t5_format = {
            "enc.": "encoder.",
            ".blk.": ".block.",
            "token_embd": "shared",
            "output_norm": "final_layer_norm",
            "attn_q": "layer.0.SelfAttention.q",
            "attn_k": "layer.0.SelfAttention.k",
            "attn_v": "layer.0.SelfAttention.v",
            "attn_o": "layer.0.SelfAttention.o",
            "attn_norm": "layer.0.layer_norm",
            "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
            "ffn_up": "layer.1.DenseReluDense.wi_1",
            "ffn_down": "layer.1.DenseReluDense.wo",
            "ffn_gate": "layer.1.DenseReluDense.wi_0",
            "ffn_norm": "layer.1.layer_norm",
        }
        new_sd = {}
        for k, v in state_dict.items():
            for s, d in gguf_t5_format.items():
                k = k.replace(s, d)
            new_sd[k] = v
        new_sd["shared.weight"] = new_sd["shared.weight"].dequantize_as_pytorch_parameter()
        state_dict.clear()
        state_dict = new_sd

    if "blk.0.attn_norm.weight" in state_dict:
        gguf_llm_format = {
            "blk.": "model.layers.",
            "attn_norm": "input_layernorm",
            "attn_q_norm.": "self_attn.q_norm.",
            "attn_k_norm.": "self_attn.k_norm.",
            "attn_v_norm.": "self_attn.v_norm.",
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_norm": "post_attention_layernorm",
            "token_embd": "model.embed_tokens",
            "output_norm": "model.norm",
            "output.weight": "lm_head.weight",
        }
        if architecture == "gemma3":
            gguf_llm_format.update(
                {
                    "ffn_norm": "pre_feedforward_layernorm",
                    "post_ffw_norm": "post_feedforward_layernorm",
                    "post_attention_norm": "post_attention_layernorm",
                }
            )
        new_sd = {}
        for k, v in state_dict.items():
            for s, d in gguf_llm_format.items():
                k = k.replace(s, d)
            new_sd[k] = v
        new_sd["model.embed_tokens.weight"] = new_sd["model.embed_tokens.weight"].dequantize_as_pytorch_parameter()
        state_dict.clear()
        state_dict = new_sd

        if architecture == "gemma3":
            state_dict = _gemma3_norm_corrections(state_dict)

    if "v.patch_embd.weight.1" in state_dict:
        w1 = dequantize(state_dict.pop("v.patch_embd.weight"), torch.float32)
        w2 = dequantize(state_dict.pop("v.patch_embd.weight.1"), torch.float32)
        state_dict["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    if any(key.startswith(("mm.", "v.")) for key in state_dict):
        gguf_clip_vision_format = {
            "mm.": "visual.merger.mlp.",
            "v.post_ln.": "visual.merger.ln_q.",
            "v.patch_embd": "visual.patch_embed.proj",
            "v.blk.": "visual.blocks.",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate": "mlp.gate_proj",
            "attn_out.": "attn.proj.",
            "ln1.": "norm1.",
            "ln2.": "norm2.",
        }
        new_sd = {}
        for k, v in state_dict.items():
            for s, d in gguf_clip_vision_format.items():
                k = k.replace(s, d)
            new_sd[k] = v
        state_dict.clear()
        state_dict = new_sd

        if architecture == "qwen3vl":
            state_dict = {
                key if key.startswith("model.") else f"model.{key}": value
                for key, value in state_dict.items()
            }

    vision_prefix = "model." if "model.visual.blocks.0.attn_q.weight" in state_dict else ""
    if f"{vision_prefix}visual.blocks.0.attn_q.weight" in state_dict:
        attns = {}

        _keys = list(state_dict.keys())
        _sd = {}

        for k in _keys:
            if any(x in k for x in ["attn_q", "attn_k", "attn_v"]):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                if k_attn not in attns:
                    attns[k_attn] = {}
                value = state_dict.pop(k)
                dtype = torch.bfloat16 if getattr(value, "gguf_cls", None) is not None else torch.float16
                attns[k_attn][k_name] = dequantize(value, dtype)
            else:
                _sd[k] = state_dict.pop(k)

        del state_dict
        state_dict = _sd

        for k, v in attns.items():
            suffix = k.split(".")[-1]
            state_dict[k] = torch.cat(
                [
                    v[f"q.{suffix}"],
                    v[f"k.{suffix}"],
                    v[f"v.{suffix}"],
                ],
                dim=0,
            )

        del attns

    return state_dict


from modules_forge.packages import gguf


def get_orig_shape(reader: gguf.GGUFReader, tensor_name: str) -> torch.Size | None:
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) == 2 and field.types[0] == gguf.GGUFValueType.ARRAY and field.types[1] == gguf.GGUFValueType.INT32:
        return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))
    return None


# Upstream-compatible architecture metadata and standalone text-encoder helpers.
logger = logging.getLogger("loader_gguf")
IMAGE_ARCHITECTURES = frozenset({
    "flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos",
    "ltxv", "hyvid", "wan", "lumina2", "qwen_image",
})
TEXT_ARCHITECTURES = frozenset({
    "t5", "t5encoder", "llama", "qwen2vl", "qwen3", "qwen3vl", "gemma3",
})
VISION_TYPES = frozenset({"clip-vision", "mmproj"})


def _decode_field_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8")
    if hasattr(value, "tobytes"):
        return value.tobytes().decode("utf-8")
    return str(value)


def _field_value(field, index=None):
    if field is None or not field.data:
        return None
    value = field.parts[field.data[-1] if index is None else index]
    if field.types and field.types[-1] == gguf.GGUFValueType.STRING:
        return _decode_field_text(value)
    if hasattr(value, "item"):
        value = value.item()
    return value


def get_field(reader: gguf.GGUFReader, field_name: str, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    value = _field_value(field)
    if field_type is str:
        return str(value)
    if field_type in (int, float, bool):
        return field_type(value)
    raise TypeError(f"Unsupported GGUF field type: {field_type}")


def get_list_field(reader: gguf.GGUFReader, field_name: str, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    values = [_field_value(field, part_index) for part_index in field.data]
    if field_type is str:
        return tuple(str(value) for value in values)
    if field_type in (int, float, bool):
        return tuple(field_type(value) for value in values)
    raise TypeError(f"Unsupported GGUF list field type: {field_type}")


def get_gguf_metadata(reader: gguf.GGUFReader) -> dict:
    metadata = {}
    for field_name in reader.fields:
        field = reader.get_field(field_name)
        if field is None or len(field.types) != 1:
            continue
        try:
            value_type = field.types[0]
            if value_type == gguf.GGUFValueType.STRING:
                metadata[field_name] = get_field(reader, field_name, str)
            elif value_type == gguf.GGUFValueType.INT32:
                metadata[field_name] = get_field(reader, field_name, int)
            elif value_type == gguf.GGUFValueType.FLOAT32:
                metadata[field_name] = get_field(reader, field_name, float)
            elif value_type == gguf.GGUFValueType.BOOL:
                metadata[field_name] = get_field(reader, field_name, bool)
        except (TypeError, ValueError, UnicodeDecodeError):
            logger.debug("Unable to decode GGUF metadata field %s", field_name, exc_info=True)
    return metadata


def _validate_architecture(architecture: str | None, model_type: str | None, *, is_text_model: bool) -> None:
    if architecture in (None, "pig", "cow"):
        if is_text_model:
            raise ValueError("GGUF text encoder has no compatible architecture metadata")
        return
    supported = TEXT_ARCHITECTURES if is_text_model else IMAGE_ARCHITECTURES
    if architecture not in supported and not (is_text_model and model_type in VISION_TYPES):
        kind = "text encoder" if is_text_model else "diffusion model"
        raise ValueError(f"Unsupported GGUF {kind} architecture: {architecture}")


def gguf_sd_loader(path: os.PathLike, handle_prefix: str | None = None, *, is_text_model=False, strict_architecture=False):
    """Read a GGUF file as Forge-compatible quantized parameters."""
    reader = gguf.GGUFReader(path)
    architecture = get_field(reader, "general.architecture", str)
    model_type = get_field(reader, "general.type", str)
    if strict_architecture:
        _validate_architecture(architecture, model_type, is_text_model=is_text_model)

    tensor_names = {str(tensor.name) for tensor in reader.tensors}
    has_prefix = bool(handle_prefix and any(name.startswith(handle_prefix) for name in tensor_names))
    prefix_length = len(handle_prefix or "")
    state_dict = {}
    qtype_counts = {}

    from backend.operations_gguf import ParameterGGUF

    for tensor in reader.tensors:
        tensor_name = str(tensor.name)
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            state_key = tensor_name[prefix_length:]
        else:
            state_key = tensor_name

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(value) for value in reversed(tensor.shape)))
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)

        parameter = ParameterGGUF(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            parameter = dequantize(parameter, dtype=torch.float32)
        state_dict[state_key] = parameter

        type_name = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_counts[type_name] = qtype_counts.get(type_name, 0) + 1

    quantized = {key: value for key, value in state_dict.items() if getattr(value, "gguf_cls", None) is not None}
    if quantized:
        largest_key = max(quantized, key=lambda key: quantized[key].numel())
        quantized[largest_key].is_largest_weight = True
    logger.info("GGUF architecture=%s type=%s qtypes=%s", architecture, model_type, qtype_counts)
    return state_dict, {"arch_str": architecture, "type_str": model_type, "metadata": get_gguf_metadata(reader)}


def _gemma3_norm_corrections(state_dict):
    patterns = (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight", "model.norm.weight",
    )
    for key in list(state_dict):
        if any(pattern in key for pattern in patterns):
            value = state_dict[key]
            if getattr(value, "gguf_cls", None) is not None:
                value = dequantize(value, dtype=torch.float32)
            state_dict[key] = value.float() - 1.0
    return state_dict


def gguf_clip_loader(path: os.PathLike) -> dict[str, torch.Tensor]:
    """Load and remap a standalone GGUF text encoder for Forge."""
    state_dict, extra = gguf_sd_loader(path, is_text_model=True, strict_architecture=True)
    return gguf_remapping(state_dict, architecture=extra.get("arch_str"))
