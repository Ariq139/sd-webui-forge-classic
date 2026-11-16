import logging
import os

import torch
import gguf
from gguf import GGMLQuantizationType

from modules_forge.packages.gguf import quants

QUANTS_MAPPING: dict[GGMLQuantizationType, type[quants.TorchQuant]] = {
    GGMLQuantizationType.Q2_K: quants.Q2_K,
    GGMLQuantizationType.Q3_K: quants.Q3_K,
    GGMLQuantizationType.Q4_0: quants.Q4_0,
    GGMLQuantizationType.Q4_1: quants.Q4_1,
    GGMLQuantizationType.Q4_K: quants.Q4_K,
    GGMLQuantizationType.Q5_0: quants.Q5_0,
    GGMLQuantizationType.Q5_1: quants.Q5_1,
    GGMLQuantizationType.Q5_K: quants.Q5_K,
    GGMLQuantizationType.Q6_K: quants.Q6_K,
    GGMLQuantizationType.Q8_0: quants.Q8_0,
    GGMLQuantizationType.BF16: quants.BF16,
    GGMLQuantizationType.IQ1_S: quants.IQ1_S,
    GGMLQuantizationType.IQ1_M: quants.IQ1_M,
    GGMLQuantizationType.IQ2_XXS: quants.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS: quants.IQ2_XS,
    GGMLQuantizationType.IQ2_S: quants.IQ2_S,
    GGMLQuantizationType.IQ3_XXS: quants.IQ3_XXS,
    GGMLQuantizationType.IQ3_S: quants.IQ3_S,
    GGMLQuantizationType.IQ4_NL: quants.IQ4_NL,
    GGMLQuantizationType.IQ4_XS: quants.IQ4_XS,
    GGMLQuantizationType.NVFP4: quants.NVFP4,
}

_TORCH_COMPATIBLE_QTYPES = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}


class _FallbackGGUFQuant:
    """Reference dequantizer for GGUF types not known to the fast mapping."""

    def __init__(self, tensor_type):
        self.tensor_type = tensor_type

    def bake(self, tensor):
        tensor.baked = True

    def dequantize_pytorch(self, tensor):
        data = gguf.quants.dequantize(tensor.detach().cpu().numpy(), self.tensor_type)
        return torch.from_numpy(data).to(device=tensor.device, dtype=tensor.computation_dtype)

logger = logging.getLogger("operations_gguf")
_CUDA_MODULE = None
_CUDA_PROBED = False


def _llamacpp_cuda_module():
    global _CUDA_MODULE, _CUDA_PROBED

    if _CUDA_PROBED:
        return _CUDA_MODULE
    _CUDA_PROBED = True
    if os.environ.get("FORGE_GGUF_CUDA", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        import llamacpp_gguf_cuda

        if not torch.cuda.is_available() or not hasattr(llamacpp_gguf_cuda, "linear"):
            return None
        _CUDA_MODULE = llamacpp_gguf_cuda
        logger.info("GGUF CUDA linear kernels are available")
    except Exception:
        logger.debug("GGUF CUDA kernels are unavailable; using dequantization", exc_info=True)
    return _CUDA_MODULE


def try_llamacpp_cuda_linear(weight, input_tensor, bias):
    if input_tensor.device.type != "cuda" or getattr(weight, "gguf_cls", None) is None:
        return None

    module = _llamacpp_cuda_module()
    tensor_type = getattr(weight, "tensor_type", None)
    qtype_name = getattr(tensor_type, "name", None)
    raw_weight = getattr(weight, "data", None)
    if module is None or not qtype_name or raw_weight is None or not raw_weight.is_contiguous():
        return None

    try:
        if not module.may_support_linear_qtype_name(qtype_name):
            return None
        return module.linear(raw_weight, qtype_name, tuple(weight.real_shape), input_tensor, bias, input_tensor.dtype)
    except Exception:
        logger.debug("GGUF CUDA linear failed for %s; using dequantization", qtype_name, exc_info=True)
        return None


def try_llamacpp_cuda_embedding(weight, input_tensor, dtype):
    if input_tensor.device.type != "cuda" or getattr(weight, "gguf_cls", None) is None:
        return None

    module = _llamacpp_cuda_module()
    tensor_type = getattr(weight, "tensor_type", None)
    qtype_name = getattr(tensor_type, "name", None)
    raw_weight = getattr(weight, "data", None)
    if module is None or not qtype_name or raw_weight is None or not raw_weight.is_contiguous():
        return None

    try:
        if not hasattr(module, "embedding") or not module.may_support_embedding_qtype_name(qtype_name):
            return None
        return module.embedding(raw_weight, qtype_name, tuple(weight.real_shape), input_tensor, dtype)
    except Exception:
        logger.debug("GGUF CUDA embedding failed for %s; using dequantization", qtype_name, exc_info=True)
        return None


class ParameterGGUF(torch.nn.Parameter):
    def __init__(self, torch_tensor, *, tensor_type=None, tensor_shape=None, no_init=False):
        super().__init__()
        if no_init:
            return

        if tensor_type in _TORCH_COMPATIBLE_QTYPES:
            self.gguf_cls = None
        else:
            self.gguf_cls = QUANTS_MAPPING.get(tensor_type, _FallbackGGUFQuant(tensor_type))
        self.tensor_type = tensor_type
        self.real_shape: torch.Size = tensor_shape
        self.computation_dtype = torch.float16
        self.baked = False

    @property
    def shape(self):
        return self.real_shape

    def __new__(cls, torch_tensor, *, tensor_type=None, tensor_shape=None, no_init=False):
        return super().__new__(cls, torch_tensor, requires_grad=False)

    def dequantize_as_pytorch_parameter(self):
        if self.gguf_cls is not None:
            self.gguf_cls.bake(self)
        data = dequantize_tensor(self)
        if isinstance(data, ParameterGGUF):
            data = torch.Tensor(data)
        return torch.nn.Parameter(data, requires_grad=False)

    def copy_with_data(self, data):
        new = ParameterGGUF(data, no_init=True)
        new.tensor_type = self.tensor_type
        new.gguf_cls = self.gguf_cls
        new.real_shape = self.real_shape
        new.computation_dtype = self.computation_dtype
        new.baked = self.baked
        new.is_largest_weight = getattr(self, "is_largest_weight", False)
        return new

    def to(self, *args, **kwargs):
        return self.copy_with_data(self.data.to(*args, **kwargs))

    def pin_memory(self, device=None):
        return self.copy_with_data(self.data.pin_memory(device=device))


def dequantize_tensor(tensor: "ParameterGGUF", *, input: torch.Tensor = None) -> torch.Tensor:
    if tensor is None:
        return None

    if not hasattr(tensor, "gguf_cls"):
        return tensor

    if (gguf_cls := tensor.gguf_cls) is not None:
        if not getattr(tensor, "baked", False):
            gguf_cls.bake(tensor)
        if input is not None and input.is_cuda and gguf_cls in (quants.Q2_K, quants.Q3_K, quants.Q4_K, quants.Q5_K, quants.Q6_K):
            from modules_forge.packages.gguf.kquant_triton import try_dequantize

            if (weight := try_dequantize(tensor, input)) is not None:
                return weight
        return gguf_cls.dequantize_pytorch(tensor)

    return tensor
