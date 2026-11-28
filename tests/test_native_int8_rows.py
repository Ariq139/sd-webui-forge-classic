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


@pytest.mark.parametrize("convrot", [False, True])
def test_mocked_native_dispatch_reuses_rows_and_discards_on_decline(monkeypatch, convrot):
    from types import SimpleNamespace
    from unittest.mock import Mock

    x = torch.zeros(1, 4, 256, dtype=torch.bfloat16)

    class SimulatedDeviceTensor:
        is_cuda = True
        def __getattr__(self, name):
            return getattr(x, name)
        def get_device(self):
            return 0

    source = SimulatedDeviceTensor()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=12))
    producer = Mock(side_effect=lambda flat: (flat.to(torch.int8), torch.ones(flat.shape[0], 1)))
    def rotated(flat, q, scales, group, stochastic, act, seed, act_weight, eps, stream):
        assert (group, stochastic, act, seed, eps, stream) == (256, False, 0, 0, 0.0, 12)
        q.copy_(flat.to(torch.int8))
        scales.fill_(1)
    rotated_producer = Mock(side_effect=rotated)
    decline = [False]
    def epilogue(q, w, rs, ws, b, out, dtype_code, stream):
        assert dtype_code == 5 and stream == 12
        out.copy_((q.float() @ w.float().T).to(out.dtype))
        return not decline[0]
    cutlass = Mock(side_effect=epilogue)
    cuda = SimpleNamespace(
        quantize_int8_rowwise=producer, _DISABLE_CUTLASS_INT8=False,
        _CONVROT_FUSED_MAX_K=4096, _convrot_fused_shared_memory_fits=lambda *args: True,
        _cuda_device_supports_cutlass_int8_dequant=lambda *args: True,
        _prefer_cublas_int8_fallback=lambda *args: False,
        _prefer_turing_fused_int8=lambda *args: False,
        _int8_weight_scale_arg=lambda s, d: s,
        _gemm_vector_arg=lambda v, d, dtype: v,
        _empty_cuda_tensor=lambda d, dtype: torch.empty(0, device=d, dtype=dtype),
        _wrap_for_dlpack=lambda t: t, _input_act_code=lambda act: 0,
        DTYPE_TO_CODE={torch.bfloat16: 5},
        _C=SimpleNamespace(cutlass_int8_dequant=cutlass, quantize_int8_rowwise_convrot64=rotated_producer),
    )
    cache = {}
    w = torch.ones(16, 256, dtype=torch.int8)
    scale = torch.ones(16)
    for _ in range(2):
        out = rows.try_native_cutlass(source, w, scale, None, convrot, cache, cuda)
        assert out.shape == (1, 4, 16)
    assert producer.call_count == (0 if convrot else 1)
    assert rotated_producer.call_count == (1 if convrot else 0)
    assert cutlass.call_count == 2
    decline[0] = True
    assert rows.try_native_cutlass(source, w, scale, None, convrot, cache, cuda) is None
    assert not cache
