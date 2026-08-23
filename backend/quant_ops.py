# https://github.com/Comfy-Org/ComfyUI/blob/v0.33.1/comfy/quant_ops.py

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import (  # noqa
    AsymW4A8Int8Layout,
    QuantizedTensor,
    TensorCoreAWQW4A16Layout,
    TensorCoreConvRotW4A4Layout,
    TensorCoreFP8Layout,
    TensorCoreMXFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
    dequantize_w4a8_int8_weight,
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

from . import float

# region Layouts


class TensorCoreFP8LayoutBase(TensorCoreFP8Layout):
    FP8_DTYPE = None

    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        assert cls.FP8_DTYPE is not None

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if isinstance(scale, str) and scale == "recalculate":
            scale = torch.amax(tensor.abs()).to(dtype=torch.float32) / torch.finfo(cls.FP8_DTYPE).max
            if tensor.dtype not in [torch.float32, torch.bfloat16]:
                tensor_info = torch.finfo(tensor.dtype)
                scale = 1.0 / torch.clamp((1.0 / scale), min=tensor_info.min, max=tensor_info.max)

        if scale is None:
            scale = torch.ones((), device=tensor.device, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=torch.float32)

        if stochastic_rounding > 0:
            if inplace_ops:
                tensor *= (1.0 / scale).to(tensor.dtype)
            else:
                tensor = tensor * (1.0 / scale).to(tensor.dtype)
            qdata = float.stochastic_rounding(tensor, dtype=cls.FP8_DTYPE, seed=stochastic_rounding)
        else:
            qdata = ck.quantize_per_tensor_fp8(tensor, scale, cls.FP8_DTYPE)

        params = cls.Params(scale=scale.float(), orig_dtype=orig_dtype, orig_shape=orig_shape)
        return qdata, params


class TensorCoreMXFP8Layout_(TensorCoreMXFP8Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        assert tensor.dim() == 2

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = float.stochastic_round_quantize_mxfp8_by_block(tensor, pad_32x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_mxfp8(tensor, pad_32x=needs_padding)

        params = cls.Params(
            scale=block_scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
        )
        return qdata, params


class TensorCoreNVFP4Layout_(TensorCoreNVFP4Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        assert tensor.dim() == 2

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if scale is None or (isinstance(scale, str) and scale == "recalculate"):
            scale = torch.amax(tensor.abs()) / (ck.float_utils.F8_E4M3_MAX * ck.float_utils.F4_E2M1_MAX)

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = float.stochastic_round_quantize_nvfp4_by_block(tensor, scale, pad_16x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_nvfp4(tensor, scale, pad_16x=needs_padding)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            block_scale=block_scale,
        )
        return qdata, params


class TensorCoreFP8E4M3Layout(TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e4m3fn


class TensorCoreFP8E5M2Layout(TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e5m2


# region Registry


register_layout_class("TensorCoreFP8Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8E5M2Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout_)
register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout_)
register_layout_class("TensorWiseINT8Layout", TensorWiseINT8Layout)
register_layout_class("TensorCoreConvRotW4A4Layout", TensorCoreConvRotW4A4Layout)
register_layout_class("AsymW4A8Int8Layout", AsymW4A8Int8Layout)
register_layout_class("TensorCoreAWQW4A16Layout", TensorCoreAWQW4A16Layout)


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
    "awq_w4a16": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale", "weight_zero"},
        "comfy_tensor_layout": "TensorCoreAWQW4A16Layout",
        "quantize_input": False,
        "group_size": 64,
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


_NVFP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nvfp4_from_blocked(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Undo the cuBLAS block-scale layout used by NVFP4."""
    row_blocks = (rows + 127) // 128
    col_blocks = (cols + 3) // 4
    step = blocked.reshape(-1, 32, 16)
    step = step.reshape(-1, 32, 4, 4).transpose(1, 2)
    step = step.reshape(row_blocks, col_blocks, 4, 32, 4)
    step = step.reshape(row_blocks, col_blocks, 128, 4).permute(0, 2, 1, 3)
    return step.reshape(row_blocks * 128, col_blocks * 4)[:rows, :cols]


def nvfp4_embedding_lookup(qdata: torch.Tensor, params, input: torch.Tensor) -> torch.Tensor:
    """Decode only the unique NVFP4 embedding rows referenced by ``input``."""
    if qdata.ndim != 2 or qdata.dtype is not torch.uint8:
        raise ValueError("NVFP4 embedding data must be a two-dimensional uint8 tensor")
    input = input.to(device=qdata.device)

    original_rows, original_cols = params.orig_shape
    padded_cols = qdata.shape[1] * 2
    block_count = (padded_cols + 15) // 16
    scale_cols = ((block_count + 3) // 4) * 4
    if params.block_scale.numel() % scale_cols != 0:
        raise ValueError("NVFP4 embedding block scales have an invalid shape")

    scale_rows = params.block_scale.numel() // scale_cols
    if scale_rows < original_rows or qdata.shape[0] < original_rows:
        raise ValueError("NVFP4 embedding storage is smaller than its logical shape")

    token_ids, inverse = torch.unique(input, sorted=False, return_inverse=True)
    if token_ids.numel() == 0:
        return torch.empty((*input.shape, original_cols), device=qdata.device, dtype=params.orig_dtype)
    if token_ids.min() < 0 or token_ids.max() >= original_rows:
        raise IndexError("NVFP4 embedding token index is outside the logical embedding table")

    selected_qdata = qdata.index_select(0, token_ids)
    high = selected_qdata >> 4
    low = selected_qdata & 0x0F
    codes = torch.stack((high, low), dim=-1).reshape(selected_qdata.shape[0], padded_cols)

    values = torch.tensor(_NVFP4_E2M1_VALUES, device=qdata.device, dtype=torch.float32)
    decoded = values[(codes & 0x07).long()]
    decoded = torch.where((codes & 0x08) != 0, -decoded, decoded)

    blocked_scales = params.block_scale.reshape(scale_rows, scale_cols)
    block_scales = _nvfp4_from_blocked(blocked_scales, scale_rows, scale_cols)
    block_scales = block_scales.index_select(0, token_ids)[:, :block_count].to(dtype=torch.float32)
    decoded = decoded.reshape(selected_qdata.shape[0], block_count, 16)
    decoded = (decoded * block_scales.unsqueeze(-1)).reshape(selected_qdata.shape[0], padded_cols)
    decoded = decoded[:, :original_cols] * params.scale.to(device=qdata.device, dtype=torch.float32)
    decoded = decoded.to(dtype=params.orig_dtype)

    return torch.nn.functional.embedding(inverse, decoded)


def _embedding_token_rows(input: torch.Tensor, rows: int, device: torch.device):
    input = input.to(device=device)
    token_ids, inverse = torch.unique(input, sorted=False, return_inverse=True)
    if token_ids.numel() and (token_ids.min() < 0 or token_ids.max() >= rows):
        raise IndexError("Quantized embedding token index is outside the logical embedding table")
    return token_ids, inverse


def w4a8_embedding_lookup(qdata: torch.Tensor, params, input: torch.Tensor) -> torch.Tensor:
    """Decode only the unique W4A8 embedding rows referenced by ``input``."""
    if qdata.ndim != 2 or qdata.dtype is not torch.int8:
        raise ValueError("W4A8 embedding data must be a two-dimensional int8 tensor")

    original_rows, original_cols = params.orig_shape
    token_ids, inverse = _embedding_token_rows(input, original_rows, qdata.device)
    if token_ids.numel() == 0:
        return torch.empty((*input.shape, original_cols), device=qdata.device, dtype=params.orig_dtype)

    selected_qdata = qdata.index_select(0, token_ids)
    s_rel = params.scale.index_select(0, token_ids)
    s_channel = params.s_channel.index_select(0, token_ids)
    correction = params.correction.index_select(1, token_ids) if params.correction is not None else None
    decoded = dequantize_w4a8_int8_weight(
        selected_qdata,
        s_rel,
        s_channel,
        codebook=params.codebook,
        correction=correction,
        group_size=params.group_size,
        convrot_groupsize=params.convrot_groupsize,
        output_dtype=params.orig_dtype,
    )
    return torch.nn.functional.embedding(inverse, decoded[:, :original_cols])


def awq_w4a16_embedding_lookup(qdata: torch.Tensor, params, input: torch.Tensor) -> torch.Tensor:
    """Decode only the unique AWQ W4A16 embedding rows referenced by ``input``."""
    if qdata.ndim != 2 or qdata.dtype is not torch.int8:
        raise ValueError("AWQ W4A16 embedding data must be a two-dimensional int8 tensor")
    if params.transposed:
        raise ValueError("AWQ W4A16 embedding lookup does not support transposed weights")

    original_rows, original_cols = params.orig_shape
    token_ids, inverse = _embedding_token_rows(input, original_rows, qdata.device)
    if token_ids.numel() == 0:
        return torch.empty((*input.shape, original_cols), device=qdata.device, dtype=params.orig_dtype)

    _, packed_cols = qdata.shape
    columns = packed_cols * 2
    if columns % params.group_size != 0:
        raise ValueError("AWQ W4A16 embedding width is not divisible by its group size")

    selected_qdata = qdata.index_select(0, token_ids)
    packed = selected_qdata.to(torch.int32) & 0xFF
    values = torch.empty(selected_qdata.shape[0], columns, dtype=torch.int32, device=qdata.device)
    values[:, 0::2] = packed & 0x0F
    values[:, 1::2] = packed >> 4

    groups = values.view(selected_qdata.shape[0], columns // params.group_size, params.group_size).to(params.orig_dtype)
    scales = params.scale.index_select(1, token_ids).transpose(0, 1).unsqueeze(-1)
    zeros = params.zeros.index_select(1, token_ids).transpose(0, 1).unsqueeze(-1)
    decoded = ((groups - 8.0) * scales + zeros).reshape(selected_qdata.shape[0], columns).to(dtype=params.orig_dtype)
    return torch.nn.functional.embedding(inverse, decoded[:, :original_cols])
