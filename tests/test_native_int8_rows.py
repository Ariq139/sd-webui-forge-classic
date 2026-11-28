import pytest
import torch

from backend import native_int8_rows as rows


def test_row_cache_groups_stream_rotation_and_mutation():
    x = torch.randn(3, 4, 16)
    calls = []
    def quantize(flat, rotate):
        calls.append(rotate)
        return flat.to(torch.int8), torch.ones(flat.shape[0], 1)
    cache = {}
    a = rows.shared_rows(x, False, 1, cache, quantize)
    assert rows.shared_rows(x, False, 1, cache, quantize) is a
    assert rows.shared_rows(x, True, 1, cache, quantize) is not a
    assert rows.shared_rows(x, False, 2, cache, quantize) is not a
    x.add_(1)
    assert rows.shared_rows(x, False, 1, cache, quantize) is not a
    assert calls == [False, True, False, False]


def test_row_cache_budget_and_cpu_dispatch(monkeypatch):
    x = torch.zeros(4, 16)
    monkeypatch.setattr(rows, "MAX_ROW_BYTES", 80)
    cache = {}
    def quantize(flat, rotate):
        return flat.to(torch.int8), torch.ones(flat.shape[0], 1)
    assert rows.shared_rows(x, False, 0, cache, quantize) is not None
    assert rows.shared_rows(x, True, 0, cache, quantize) is None
    assert len(cache) == 1
    assert rows.try_native_cutlass(x, torch.zeros(16, 16, dtype=torch.int8), torch.ones(16), None, False, cache, None) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("convrot", [False, True])
def test_pinned_sm86_native_epilogue_matches_original(convrot):
    cuda = rows._cuda_backend()
    if cuda is None or torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("requires Comfy-Kitchen 0.2.40 and SM86")
    x = torch.randn(1, 256, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (256, 256), device="cuda", dtype=torch.int8)
    scale = torch.rand(256, device="cuda")
    bias = torch.randn(256, device="cuda", dtype=x.dtype)
    cache = {}
    with torch.no_grad():
        for w, s, b in ((weight, scale, None), (weight.clone(), scale, bias)):
            actual = rows.try_native_cutlass(x, w, s, b, convrot, cache, cuda)
            if actual is None:
                pytest.skip("native CUTLASS route declined")
            expected = cuda.int8_linear(x, w, s, b, x.dtype, convrot=convrot)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(cache) == 1
