# https://github.com/Comfy-Org/ComfyUI/blob/v0.27.0/comfy/quant_ops.py

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import (  # noqa
    AsymW4A8Int8Layout,
    QuantizedTensor,
    TensorCoreConvRotW4A4Layout,
    TensorCoreFP8Layout as _CKTensorCoreFP8Layout,
    TensorCoreMXFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
    get_layout_class,
    register_layout_class,
)

if torch.version.cuda is None:
    ck.registry.disable("cuda")
else:
    cuda_version = tuple(map(int, str(torch.version.cuda).split(".")))
    if cuda_version < (13,):
        ck.registry.disable("cuda")

from backend.args import args

if args.enable_triton_backend:
    try:
        import triton  # noqa
    except ImportError:
        ck.registry.disable("triton")
else:
    ck.registry.disable("triton")

import importlib.metadata

ver = importlib.metadata.version("comfy-kitchen")

print(f"Comfy-Kitchen {ver}:", {k: v["available"] and not v["disabled"] for k, v in ck.list_backends().items()})


# region Registry


class _TensorCoreFP8LayoutBase(_CKTensorCoreFP8Layout):
    """Keep Comfy's E4M3/E5M2 layouts distinct when re-quantizing weights."""

    FP8_DTYPE = None

    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if cls.FP8_DTYPE is None:
            raise NotImplementedError(f"{cls.__name__} must define FP8_DTYPE")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)
        if isinstance(scale, str) and scale == "recalculate":
            scale = torch.amax(tensor.abs()).to(dtype=torch.float32) / torch.finfo(cls.FP8_DTYPE).max
            if tensor.dtype not in (torch.float32, torch.bfloat16):
                tensor_info = torch.finfo(tensor.dtype)
                scale = 1.0 / torch.clamp(1.0 / scale, min=tensor_info.min, max=tensor_info.max)
        if scale is None:
            scale = torch.ones((), device=tensor.device, dtype=torch.float32)
        elif not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=torch.float32)

        if stochastic_rounding > 0:
            scaled = tensor * (1.0 / scale).to(tensor.dtype)
            qdata = globals()["stochastic_rounding"](scaled, dtype=cls.FP8_DTYPE, seed=stochastic_rounding)
        else:
            qdata = ck.quantize_per_tensor_fp8(tensor, scale, cls.FP8_DTYPE)

        return qdata, cls.Params(scale=scale.float(), orig_dtype=orig_dtype, orig_shape=orig_shape)


class TensorCoreFP8E4M3Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e4m3fn


class TensorCoreFP8E5M2Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e5m2


# Backward-compatible default used by older Forge metadata.
TensorCoreFP8Layout = TensorCoreFP8E4M3Layout


register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8E5M2Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)
register_layout_class("TensorWiseINT8Layout", TensorWiseINT8Layout)
register_layout_class("TensorCoreConvRotW4A4Layout", TensorCoreConvRotW4A4Layout)
register_layout_class("AsymW4A8Int8Layout", AsymW4A8Int8Layout)


QUANT_ALGOS = {
    "float8_e4m3fn": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E4M3Layout",
    },
    "float8_e5m2": {
        "storage_t": torch.float8_e5m2,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E5M2Layout",
    },
    "nvfp4": {
        "storage_t": torch.uint8,
        "parameters": {"weight_scale", "weight_scale_2", "input_scale", "pre_quant_scale"},
        "comfy_tensor_layout": "TensorCoreNVFP4Layout",
        "group_size": 16,
    },
    "mxfp8": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreMXFP8Layout",
        "group_size": 32,
    },
    "int8_tensorwise": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "TensorWiseINT8Layout",
        "quantize_input": False,
    },
    "convrot_w4a4": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "TensorCoreConvRotW4A4Layout",
        "quantize_input": False,
    },
    "asym_w4a8_int8": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "AsymW4A8Int8Layout",
        "quantize_input": False,
    },
}


# region float


def stochastic_rounding(value: torch.Tensor, dtype: torch.dtype, seed: int = 0):
    if dtype is torch.float32:
        return value.to(dtype=torch.float32)
    if dtype is torch.float16:
        return value.to(dtype=torch.float16)
    if dtype is torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        rng = torch.randint(0, 256, value.size(), dtype=torch.uint8, layout=value.layout, device=value.device, generator=generator)
        return ck.stochastic_rounding_fp8(value, rng, dtype)

    return value.to(dtype=dtype)
