import pytest
import torch
from backend.quantized_storage import int8_storage, _layout_types


def test_dense_storage_rejected():
    assert int8_storage(torch.ones(2, 3)) is None


def test_int8_storage_roundtrip():
    types = _layout_types()
    if types is None:
        pytest.skip("requires pinned comfy-kitchen 0.2.40")
    cls, params = types
    q = torch.arange(12, dtype=torch.int8).reshape(3, 4)
    scale = torch.ones(3, dtype=torch.float32)
    w = cls(q, "TensorWiseINT8Layout", params(scale=scale, orig_dtype=torch.float16, orig_shape=(3, 4)))
    snapshot = int8_storage(w)
    assert snapshot is not None and snapshot.matches(w)
    copied = snapshot.rebuild(tuple(part.clone() for part in snapshot.parts))
    assert torch.equal(copied._qdata, q)
    assert torch.equal(copied._params.scale, scale)
    scale.add_(1)
    assert not snapshot.matches(w)
