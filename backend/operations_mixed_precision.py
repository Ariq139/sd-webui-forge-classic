# https://github.com/Comfy-Org/ComfyUI/blob/v0.27.0/comfy/ops.py#L1163

import json

import torch

from backend import memory_management
from backend.memory_management import cast_to_device, logger

from .operations import (
    ForgeOperations,
    ForgeWeights,
    main_stream_worker,
    weights_manual_cast,
)
from .quant_ops import (  # noqa
    QUANT_ALGOS,
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorWiseINT8Layout,
    awq_w4a16_embedding_lookup,
    get_layout_class,
    nvfp4_embedding_lookup,
    w4a8_embedding_lookup,
)


def _quantized_apply(module: torch.nn.Module, fn, recurse=True):
    if recurse:
        for child in module.children():
            child._apply(fn)
    for key, param in module._parameters.items():
        if param is None:
            continue
        p: torch.Tensor = fn(param)
        try:
            module.register_parameter(key, torch.nn.Parameter(p, requires_grad=False))
        except RuntimeError:
            module.register_parameter(key, torch.nn.Parameter(p.clone(), requires_grad=False))
    for key, buf in module._buffers.items():
        if buf is not None:
            module._buffers[key] = fn(buf)
    return module


def _infer_quant_format(weight: torch.Tensor, state_dict: dict[str, torch.Tensor], prefix: str) -> str | None:
    """Recover the format used by older/incomplete Comfy quant metadata."""
    scale = state_dict.get(f"{prefix}weight_scale")
    scale_2 = state_dict.get(f"{prefix}weight_scale_2")
    zero = state_dict.get(f"{prefix}weight_zero")
    if zero is None:
        zero = state_dict.get(f"{prefix}weight_zeros")
    relative_scale = state_dict.get(f"{prefix}weight_s_rel")
    channel_scale = state_dict.get(f"{prefix}weight_s_channel")
    if weight.dtype == torch.int8 and scale is not None and zero is not None:
        return "awq_w4a16"
    if weight.dtype == torch.int8 and relative_scale is not None and channel_scale is not None:
        return "asym_w4a8_int8"
    if scale_2 is not None and weight.dtype == torch.uint8 and weight.ndim == 2:
        return "nvfp4"
    if scale is not None and getattr(torch, "float8_e8m0fnu", None) is not None and scale.dtype == torch.float8_e8m0fnu:
        return "mxfp8"
    if weight.dtype == torch.float8_e4m3fn and scale is not None and scale.dtype == torch.uint8 and scale.ndim == 2:
        return "mxfp8"
    if weight.dtype == torch.int8 and scale is not None:
        return "int8_tensorwise"
    if weight.dtype == torch.float8_e4m3fn:
        return "float8_e4m3fn"
    if weight.dtype == torch.float8_e5m2:
        return "float8_e5m2"
    # Some older files stored FP8 bytes in uint8 containers. Without a
    # format marker, a single weight scale is the only safe legacy signal.
    if weight.dtype == torch.uint8 and scale is not None:
        return "float8_e4m3fn"
    return None


def _normalize_quant_format(value):
    if not isinstance(value, str):
        return None
    return {
        "int8": "int8_tensorwise",
        "fp8": "float8_e4m3fn",
        "fp8_e4m3": "float8_e4m3fn",
        "fp8_e5m2": "float8_e5m2",
        "nvfp4_mixed": "nvfp4",
        "w4a8": "asym_w4a8_int8",
        "w4a16": "awq_w4a16",
        "awq": "awq_w4a16",
    }.get(value.strip().lower(), value.strip().lower())


def _load_quantized_module(module: torch.nn.Module, super_load, state_dict: dict[str, torch.Tensor], prefix: str, local_metadata, strict, missing_keys, unexpected_keys, error_msgs, load_extra_params=False):
    device = module.factory_kwargs["device"]
    compute_dtype = module.factory_kwargs["dtype"]
    disabled_formats = module._disabled_formats
    layer_name = prefix.rstrip(".")

    weight = state_dict.pop(f"{prefix}weight", None)
    if weight is None:
        module.weight = None
        return
    manually_loaded_keys = [f"{prefix}weight"]

    def pop_scale(name, dtype=None):
        key = f"{prefix}{name}"
        v = state_dict.pop(key, None)
        if v is not None:
            v = v.to(device=device)
            if dtype is not None:
                v = v.view(dtype=dtype)
            manually_loaded_keys.append(key)
        return v

    layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
    if layer_conf is not None:
        if isinstance(layer_conf, torch.Tensor):
            layer_conf = json.loads(layer_conf.detach().cpu().numpy().tobytes())
        elif isinstance(layer_conf, (bytes, bytearray)):
            layer_conf = json.loads(layer_conf)
        if not isinstance(layer_conf, dict):
            layer_conf = {}

    if layer_conf is None:
        module.weight = torch.nn.Parameter(weight.to(device=device, dtype=compute_dtype), requires_grad=False)
    else:
        module.quant_format = _normalize_quant_format(layer_conf.get("format"))
        if module.quant_format is None:
            module.quant_format = _infer_quant_format(weight, state_dict, prefix)
        module._full_precision_mm_config = layer_conf.get("full_precision_matrix_mult", False)
        if not module._full_precision_mm:
            module._full_precision_mm = module._full_precision_mm_config
        if module.quant_format in disabled_formats:
            module._full_precision_mm = True
        if module.quant_format is None:
            raise ValueError(
                f"Unknown quantization format for layer {layer_name}; "
                "the checkpoint has incomplete quantization metadata"
            )
        if module.quant_format not in QUANT_ALGOS:
            raise ValueError(f"Unsupported quantization format for layer {layer_name}: {module.quant_format}")

        qconfig = QUANT_ALGOS[module.quant_format]
        module.layout_type = qconfig["comfy_tensor_layout"]
        layout_cls = get_layout_class(module.layout_type)

        # Per-format scales; fp8 dtype views handle both legacy uint8-on-disk and native fp8.
        if module.quant_format in ("float8_e4m3fn", "float8_e5m2"):
            scales = {"scale": pop_scale("weight_scale")}
        elif module.quant_format == "mxfp8":
            bs = pop_scale("weight_scale", torch.float8_e8m0fnu)
            if bs is None:
                raise ValueError(f"Missing MXFP8 block scales for layer {layer_name}")
            scales = {"scale": bs}
        elif module.quant_format == "nvfp4":
            ts = pop_scale("weight_scale_2")
            bs = pop_scale("weight_scale", torch.float8_e4m3fn)
            if ts is None or bs is None:
                raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")
            scales = {"scale": ts, "block_scale": bs}
        elif module.quant_format == "int8_tensorwise":
            scale = pop_scale("weight_scale")
            if scale is None:
                raise ValueError(f"Missing INT8 weight scale for layer {layer_name}")
            module._per_row = scale.dim() == 2 and scale.shape[1] == 1
            scales = {"scale": scale}
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            if layer_conf.get("convrot", params_conf.get("convrot", False)):
                scales["convrot"] = True
                scales["convrot_groupsize"] = int(layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)))
        elif module.quant_format == "convrot_w4a4":
            scale = pop_scale("weight_scale")
            if scale is None:
                raise ValueError(f"Missing ConvRot W4A4 weight scale for layer {layer_name}")
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            scales = {
                "scale": scale,
                "convrot_groupsize": int(layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))),
                "quant_group_size": 64,
                "linear_dtype": layer_conf.get("linear_dtype", params_conf.get("linear_dtype", "int4")),
            }
        elif module.quant_format == "asym_w4a8_int8":
            scale = pop_scale("weight_s_rel")
            if scale is None:
                raise ValueError(f"Missing W4A8 group scale (weight_s_rel) for layer {layer_name}")
            if scale.dtype == torch.uint8:
                scale = scale.view(torch.float8_e4m3fn)
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            scales = {
                "scale": scale,
                "s_channel": pop_scale("weight_s_channel"),
                "correction": pop_scale("weight_correction"),
                "codebook": pop_scale("weight_codebook"),
                "group_size": int(layer_conf.get("group_size", params_conf.get("group_size", 16))),
                "convrot_groupsize": int(layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))),
            }
            if scales["s_channel"] is None:
                raise ValueError(f"Missing W4A8 channel scale for layer {layer_name}")
        elif module.quant_format == "awq_w4a16":
            scale = pop_scale("weight_scale")
            zero = pop_scale("weight_zero")
            if zero is None:
                zero = pop_scale("weight_zeros")
            if scale is None or zero is None:
                raise ValueError(f"Missing AWQ W4A16 scales for layer {layer_name}")
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            scales = {
                "scale": scale,
                "zeros": zero,
                "group_size": int(layer_conf.get("group_size", params_conf.get("group_size", 64))),
            }
        else:
            raise ValueError(f"Unsupported quantization format: {module.quant_format}")

        params = layout_cls.Params(**scales, orig_dtype=compute_dtype, orig_shape=module._orig_shape)
        module.weight = torch.nn.Parameter(
            QuantizedTensor(weight.to(device=device, dtype=qconfig["storage_t"]), module.layout_type, params),
            requires_grad=False,
        )

        if load_extra_params:
            for param_name in qconfig["parameters"]:
                if param_name in {"weight_scale", "weight_scale_2"}:
                    continue
                param_key = f"{prefix}{param_name}"
                _v = state_dict.pop(param_key, None)
                if _v is None:
                    continue
                module.register_parameter(param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
                manually_loaded_keys.append(param_key)

    super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    for key in manually_loaded_keys:
        if key in missing_keys:
            missing_keys.remove(key)


def _quantized_weight_state_dict(module: torch.nn.Module, sd: dict[str, torch.Tensor], prefix: str, extra_quant_conf: dict = None, extra_quant_params: tuple[str] = ()):
    if not hasattr(module, "weight"):
        logger.warning(f"uninitialized op {prefix}")
        return sd
    bias = getattr(module, "bias", None)
    if bias is not None:
        sd[f"{prefix}bias"] = bias
    if module.weight is None:
        return sd

    if not isinstance(module.weight, QuantizedTensor):
        sd[f"{prefix}weight"] = module.weight
    else:
        sd.update(module.weight.state_dict(f"{prefix}weight"))
        quant_conf = {"format": module.quant_format}
        if getattr(module, "_full_precision_mm_config", False):
            quant_conf["full_precision_matrix_mult"] = True
        params = getattr(module.weight, "_params", None)
        if module.quant_format == "int8_tensorwise" and getattr(params, "convrot", False):
            quant_conf["convrot"] = True
            quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
        elif module.quant_format == "convrot_w4a4":
            quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
            linear_dtype = getattr(params, "linear_dtype", "int4")
            if linear_dtype != "int4":
                quant_conf["linear_dtype"] = linear_dtype
        elif module.quant_format == "asym_w4a8_int8":
            params = getattr(module.weight, "_params", None)
            if params is not None:
                quant_conf["group_size"] = getattr(params, "group_size", 16)
                quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
            for name, attribute in (
                ("weight_s_rel", "scale"),
                ("weight_s_channel", "s_channel"),
                ("weight_correction", "correction"),
                ("weight_codebook", "codebook"),
            ):
                value = getattr(params, attribute, None)
                if value is not None:
                    sd[f"{prefix}{name}"] = value
        elif module.quant_format == "awq_w4a16":
            params = getattr(module.weight, "_params", None)
            if params is not None:
                quant_conf["group_size"] = getattr(params, "group_size", 64)
            plural_key = f"{prefix}weight_zeros"
            singular_key = f"{prefix}weight_zero"
            if plural_key in sd:
                sd[singular_key] = sd.pop(plural_key)
        if extra_quant_conf:
            quant_conf.update(extra_quant_conf)
        sd[f"{prefix}comfy_quant"] = torch.tensor(list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8)
        for name in extra_quant_params:
            value = getattr(module, name, None)
            if value is not None:
                sd[f"{prefix}{name}"] = value

    return sd


def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[]):
    class MixedPrecisionOps(ForgeOperations):
        _quant_config = quant_config
        _compute_dtype = compute_dtype
        _full_precision_mm = full_precision_mm
        _disabled = disabled

        class Linear(torch.nn.Module, ForgeWeights):
            _disabled_formats = disabled

            def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
                super().__init__()

                self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}

                self.in_features = in_features
                self.out_features = out_features
                self._orig_shape = (out_features, in_features)
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)

                self._full_precision_mm = MixedPrecisionOps._full_precision_mm
                self._full_precision_mm_config = False

            def reset_parameters(self):
                return None

            def _load_from_state_dict(self, *args):
                _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=True)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix, extra_quant_params=("input_scale",))

            def forward(self, input, *args, **kwargs):
                input_shape = input.shape
                reshaped_nd = False

                _use_quantized = getattr(self, "layout_type", None) is not None and not isinstance(input, QuantizedTensor) and not self._full_precision_mm and not getattr(self, "forge_force_cast_weights", False) and len(self.weight_function) == 0 and len(self.bias_function) == 0
                quantize_input = QUANT_ALGOS.get(getattr(self, "quant_format", None), {}).get("quantize_input", True)

                assert not input.requires_grad

                if _use_quantized and quantize_input:
                    input_reshaped = input.reshape(-1, input_shape[-1]) if input.ndim >= 3 else input

                    if input_reshaped.ndim == 2:
                        reshaped_nd = input.ndim >= 3
                        scale = getattr(self, "input_scale", None)
                        if scale is not None:
                            scale = cast_to_device(scale, input.device, None)
                        input = QuantizedTensor.from_float(input_reshaped, self.layout_type, scale=scale)

                weight_only_quant = _use_quantized and not quantize_input and isinstance(self.weight, QuantizedTensor)

                if weight_only_quant:
                    weight, bias, signal = weights_manual_cast(self, x=None, dtype=self.weight.dtype, device=input.device, bias_dtype=input.dtype)
                    weight = weight.to(dtype=input.dtype)
                else:
                    weight, bias, signal = weights_manual_cast(self, x=input)

                with main_stream_worker(weight, bias, signal):
                    output = torch.nn.functional.linear(input, weight, bias)

                if reshaped_nd:
                    output = output.reshape((*input_shape[:-1], self.weight.shape[0]))

                return output

            def convert_weight(self, weight, inplace=False, **kwargs):
                if isinstance(weight, QuantizedTensor):
                    return weight.dequantize()
                else:
                    return weight

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                if getattr(self, "layout_type", None) is not None:
                    weight = self.weight.requantize_from_float(weight, scale="recalculate", stochastic_rounding=seed, inplace_ops=True).to(self.weight.dtype)
                else:
                    weight = weight.to(self.weight.dtype)
                if return_weight:
                    return weight

                assert inplace_update is False
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def _apply(self, fn, recurse=True):
                return _quantized_apply(self, fn, recurse)

        class Embedding(ForgeOperations.Embedding):
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = f"{prefix}weight"
                layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
                if layer_conf is not None:
                    if isinstance(layer_conf, torch.Tensor):
                        layer_conf = json.loads(layer_conf.detach().cpu().numpy().tobytes())
                    elif isinstance(layer_conf, (bytes, bytearray)):
                        layer_conf = json.loads(layer_conf)
                    if not isinstance(layer_conf, dict):
                        layer_conf = {}

                quant_format = _normalize_quant_format(layer_conf.get("format")) if layer_conf is not None else None
                if quant_format is None and weight_key in state_dict:
                    quant_format = _infer_quant_format(state_dict[weight_key], state_dict, prefix)
                manually_loaded_keys = []

                embedding_quant_formats = {"float8_e4m3fn", "float8_e5m2", "int8_tensorwise", "nvfp4", "mxfp8", "asym_w4a8_int8", "awq_w4a16"}
                if quant_format in embedding_quant_formats and weight_key in state_dict:
                    self.quant_format = quant_format
                    qconfig = QUANT_ALGOS[quant_format]
                    self.layout_type = qconfig["comfy_tensor_layout"]
                    layout_cls = get_layout_class(self.layout_type)
                    weight = state_dict.pop(weight_key)
                    manually_loaded_keys.append(weight_key)

                    scales = {}
                    scale_names = ("weight_scale", "weight_scale_2")
                    if quant_format == "awq_w4a16":
                        scale_names += ("weight_zero", "weight_zeros")
                    for scale_name in scale_names:
                        scale_key = f"{prefix}{scale_name}"
                        scale = state_dict.pop(scale_key, None)
                        if scale is not None:
                            if quant_format == "nvfp4" and scale_name == "weight_scale":
                                scale = scale.view(torch.float8_e4m3fn)
                            elif quant_format == "mxfp8" and scale_name == "weight_scale":
                                scale = scale.view(torch.float8_e8m0fnu)
                            else:
                                scale = scale.float()
                            scales[scale_name] = scale
                            manually_loaded_keys.append(scale_key)

                    extra = {}
                    if quant_format == "nvfp4":
                        scales = {"scale": scales.get("weight_scale_2"), "block_scale": scales.get("weight_scale")}
                    elif quant_format == "asym_w4a8_int8":
                        relative_key = f"{prefix}weight_s_rel"
                        relative_scale = state_dict.pop(relative_key, None)
                        if relative_scale is not None:
                            relative_scale = relative_scale.view(torch.float8_e4m3fn) if relative_scale.dtype == torch.uint8 else relative_scale
                            manually_loaded_keys.append(relative_key)
                        scales = {"scale": relative_scale}
                    elif quant_format == "awq_w4a16":
                        zero = scales.get("weight_zero")
                        if zero is None:
                            zero = scales.get("weight_zeros")
                        scales = {"scale": scales.get("weight_scale"), "zeros": zero}
                    else:
                        scales = {"scale": scales.get("weight_scale")}
                    if quant_format == "mxfp8":
                        scales = {"scale": scales.get("weight_scale")}
                    elif quant_format == "asym_w4a8_int8":
                        for param_name, scale_name in (
                            ("s_channel", "weight_s_channel"),
                            ("correction", "weight_correction"),
                            ("codebook", "weight_codebook"),
                        ):
                            param_key = f"{prefix}{scale_name}"
                            value = state_dict.pop(param_key, None)
                            scales[param_name] = value
                            if value is not None:
                                manually_loaded_keys.append(param_key)
                        params_conf = (layer_conf or {}).get("params", {})
                        if not isinstance(params_conf, dict):
                            params_conf = {}
                        extra["group_size"] = int((layer_conf or {}).get("group_size", params_conf.get("group_size", 16)))
                        extra["convrot_groupsize"] = int((layer_conf or {}).get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)))
                        if scales["scale"] is None or scales.get("s_channel") is None:
                            raise ValueError("Missing W4A8 embedding scales")
                    elif quant_format == "awq_w4a16":
                        params_conf = (layer_conf or {}).get("params", {})
                        if not isinstance(params_conf, dict):
                            params_conf = {}
                        extra["group_size"] = int((layer_conf or {}).get("group_size", params_conf.get("group_size", 64)))
                        if scales["scale"] is None or scales.get("zeros") is None:
                            raise ValueError("Missing AWQ W4A16 embedding scales")
                    elif quant_format == "nvfp4" and (scales["scale"] is None or scales["block_scale"] is None):
                        raise ValueError("Missing NVFP4 embedding scales")
                    elif quant_format == "mxfp8" and scales["scale"] is None:
                        raise ValueError("Missing MXFP8 embedding scales")
                    if quant_format == "int8_tensorwise" and (layer_conf or {}).get("convrot", False):
                        extra["convrot"] = True
                        extra["convrot_groupsize"] = int((layer_conf or {}).get("convrot_groupsize", 256))

                    parameter_values = {key: value for key, value in scales.items() if value is not None}
                    if quant_format in {"float8_e4m3fn", "float8_e5m2", "int8_tensorwise"} and "scale" not in parameter_values:
                        parameter_values["scale"] = torch.ones((), dtype=torch.float32)
                    params = layout_cls.Params(
                        **parameter_values,
                        orig_dtype=MixedPrecisionOps._compute_dtype,
                        orig_shape=(self.num_embeddings, self.embedding_dim),
                        **extra,
                    )
                    self.weight = torch.nn.Parameter(QuantizedTensor(weight.to(dtype=qconfig["storage_t"]), qconfig["comfy_tensor_layout"], params), requires_grad=False)
                elif layer_conf is not None:
                    state_dict[f"{prefix}comfy_quant"] = torch.tensor(list(json.dumps(layer_conf).encode("utf-8")), dtype=torch.uint8)

                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

                for k in manually_loaded_keys:
                    if k in missing_keys:
                        missing_keys.remove(k)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix)

            def forward(self, input):
                weight = self.weight

                if isinstance(weight, QuantizedTensor) and len(self.weight_function) == 0:
                    qdata, _, signal = weights_manual_cast(self, device=input.device, dtype=weight.dtype)
                    if isinstance(qdata, QuantizedTensor):
                        params = qdata._params
                        scale = params.scale
                        raw_qdata = qdata._qdata
                    else:
                        params = weight._params
                        scale = None
                        raw_qdata = qdata

                    with main_stream_worker(qdata, None, signal):
                        if self.quant_format == "int8_tensorwise":
                            return get_layout_class(self.layout_type).dequantize_embedding(qdata, params, input)

                        if self.quant_format == "nvfp4":
                            if (
                                memory_management.nvfp4_embedding_rowwise_enabled()
                                and self.padding_idx is None
                                and self.max_norm is None
                                and not self.scale_grad_by_freq
                                and not self.sparse
                            ):
                                return nvfp4_embedding_lookup(raw_qdata, params, input)

                            qdata = raw_qdata
                            qdata = get_layout_class(self.layout_type).dequantize(qdata, params)
                            scale = None
                        elif self.quant_format == "asym_w4a8_int8":
                            if (
                                memory_management.w4a8_embedding_rowwise_enabled()
                                and self.padding_idx is None
                                and self.max_norm is None
                                and not self.scale_grad_by_freq
                                and not self.sparse
                            ):
                                return w4a8_embedding_lookup(raw_qdata, params, input)

                            qdata = raw_qdata
                            qdata = get_layout_class(self.layout_type).dequantize(qdata, params)
                            scale = None
                        elif self.quant_format == "awq_w4a16":
                            if (
                                memory_management.w4a16_embedding_rowwise_enabled()
                                and self.padding_idx is None
                                and self.max_norm is None
                                and not self.scale_grad_by_freq
                                and not self.sparse
                            ):
                                return awq_w4a16_embedding_lookup(raw_qdata, params, input)

                            qdata = raw_qdata
                            qdata = get_layout_class(self.layout_type).dequantize(qdata, params)
                            scale = None

                        x = torch.nn.functional.embedding(input, qdata, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

                    target_dtype = weight._params.orig_dtype
                    x = x.to(dtype=target_dtype)
                    if scale is not None and scale != 1.0:
                        x = x * scale.to(dtype=target_dtype)

                    return x

                return super().forward(input)

    return MixedPrecisionOps
