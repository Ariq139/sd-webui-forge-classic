"""Unit coverage of storage contracts; actual dependency cases below are optional."""
from dataclasses import dataclass, replace

import pytest
import torch

from backend import quantized_storage as qs


@dataclass(frozen=True)
class FP8Params:
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: tuple


@dataclass(frozen=True)
class AWQParams(FP8Params):
    zeros: torch.Tensor
    group_size: int = 64
    transposed: bool = False


@dataclass(frozen=True)
class ConvRotParams(FP8Params):
    convrot_groupsize: int = 256
    quant_group_size: int = 64
    linear_dtype: str = "int4"
    transposed: bool = False


@dataclass(frozen=True)
class GroupedParams(FP8Params):
    s_channel: torch.Tensor
    correction: torch.Tensor | None = None
    codebook: torch.Tensor | None = None
    group_size: int = 16
    convrot_groupsize: int = 256
    mma_packed: bool = False
    transposed: bool = False


class QuantizedTensor:
    """Fake wrapper only for testing the adapter contract without Comfy-Kitchen."""
    def __init__(self, q, layout, p):
        self._qdata, self._layout_cls, self._params = q, layout, p
        self._version = 0


def samples(classes, cls):
    fp8, awq, convrot, grouped = classes
    shape = (8, 256)
    dtype = torch.bfloat16
    for label, dt in (("TensorCoreFP8E4M3Layout", torch.float8_e4m3fn), ("TensorCoreFP8E5M2Layout", torch.float8_e5m2)):
        yield cls(torch.zeros(shape, dtype=dt), label, fp8(torch.ones(()), dtype, shape))
    yield cls(torch.zeros(8, 128, dtype=torch.int8), "TensorCoreAWQW4A16Layout",
              awq(torch.ones(4, 8, dtype=dtype), dtype, shape, zeros=torch.zeros(4, 8, dtype=dtype)))
    yield cls(torch.zeros(8, 128, dtype=torch.int8), "TensorCoreConvRotW4A4Layout",
              convrot(torch.ones(8, 1), dtype, shape))
    for bits in (4, 6):
        yield cls(torch.zeros(8, 256 * bits // 8, dtype=torch.int8), "AsymW4A8Int8Layout",
                  grouped(torch.ones(8, 16), dtype, shape, s_channel=torch.ones(8),
                          correction=torch.ones(16, 8) if bits == 4 else None,
                          codebook=torch.arange(16, dtype=torch.float32) if bits == 4 else None))


@pytest.fixture
def weights(monkeypatch):
    classes = FP8Params, AWQParams, ConvRotParams, GroupedParams
    monkeypatch.setattr(qs, "_extra_layout_types", lambda: (QuantizedTensor, {
        "TensorCoreFP8E4M3Layout": FP8Params, "TensorCoreFP8E5M2Layout": FP8Params,
        "TensorCoreAWQW4A16Layout": AWQParams, "TensorCoreConvRotW4A4Layout": ConvRotParams,
        "AsymW4A8Int8Layout": GroupedParams,
    }))
    return list(samples(classes, QuantizedTensor))


@pytest.mark.parametrize("index", range(6))
def test_packed_roundtrip_and_mutation(weights, index):
    weight = weights[index]
    snapshot = qs.packed_storage(weight)
    assert snapshot is not None and snapshot.matches(weight)
    rebuilt = snapshot.rebuild(tuple(t.clone() for t in snapshot.parts))
    assert rebuilt._layout_cls == weight._layout_cls
    other = qs.packed_storage(rebuilt)
    assert other.metadata == snapshot.metadata
    for a, b in zip(other.parts, snapshot.parts):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)) if a.ndim else torch.equal(a, b)
    # Every metadata field participates in invalidation, not just qdata.
    weight._params.scale.add_(1)
    assert not snapshot.matches(weight)


def test_reject_partial_storage_and_unknown_layout(weights):
    weight = weights[2]
    weight._qdata = torch.zeros(9, 128, dtype=torch.int8)[:8]
    assert qs.packed_storage(weight) is None
    weight._qdata = torch.zeros(8, 128, dtype=torch.int8)
    weight._layout_cls = "UnknownLayout"
    assert qs.packed_storage(weight) is None


def test_reject_transposed_and_packed_grouped(weights):
    weight = weights[2]
    weight._params = replace(weight._params, transposed=True)
    assert qs.packed_storage(weight) is None
    weight = weights[4]
    weight._params = replace(weight._params, mma_packed=True)
    assert qs.packed_storage(weight) is None


def test_metadata_rebind_retains_actual_storage(weights):
    weight = weights[2]
    snapshot = qs.packed_storage(weight)
    weight._params.zeros.data = torch.ones_like(weight._params.zeros)
    assert not snapshot.matches(weight)
    assert not snapshot.transfer_parts[2].any()


@pytest.mark.parametrize("index", range(6))
def test_real_pinned_dependency_roundtrip(index):
    types = qs._extra_layout_types()
    if types is None:
        pytest.skip("requires pinned comfy-kitchen 0.2.40")
    cls, layouts = types
    classes = tuple(layouts[name] for name in ("TensorCoreFP8E4M3Layout", "TensorCoreAWQW4A16Layout",
                    "TensorCoreConvRotW4A4Layout", "AsymW4A8Int8Layout"))
    weight = list(samples(classes, cls))[index]
    snapshot = qs.packed_storage(weight)
    assert snapshot is not None and snapshot.matches(weight)
    rebuilt = snapshot.rebuild(tuple(t.clone() for t in snapshot.parts))
    assert rebuilt._layout_cls == weight._layout_cls and rebuilt.shape == weight.shape
    assert qs.packed_storage(rebuilt) is not None


def test_untrusted_wrapper_name_is_not_a_layout_contract():
    impostor = type("QuantizedTensor", (), {})()
    assert qs.packed_storage(impostor) is None
