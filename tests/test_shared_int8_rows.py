from unittest import mock

import pytest
import torch

from backend import activation_quantization as aq


def quantize(x):
    scale = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 127
    return (x.float() / scale).round().clamp(-127, 127).to(torch.int8), scale


def reference(x, weight, scale, bias):
    q, row_scale = quantize(x.reshape(-1, x.shape[-1]))
    accum = q.to(torch.int32) @ weight.to(torch.int32).T
    result = (accum.float() * (row_scale * scale.reshape(1, -1))).to(x.dtype)
    if bias is not None:
        result = result + bias.to(x.dtype).reshape(1, -1)
    return result.reshape(*x.shape[:-1], weight.shape[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("per_channel", [False, True])
def test_shared_rowwise_eager_matches_reference_and_reuses_quantization(dtype, per_channel):
    torch.manual_seed(5)
    x = torch.randn(2, 3, 16, dtype=dtype)
    a = torch.randint(-127, 128, (8, 16), dtype=torch.int8)
    b = torch.randint(-127, 128, (16, 16), dtype=torch.int8)
    scale_a = torch.rand(8 if per_channel else 1)
    scale_b = torch.rand(16 if per_channel else 1)
    bias = torch.randn(16, dtype=dtype)
    cache = {}
    counted = mock.Mock(side_effect=quantize)
    for w, s, z in ((a, scale_a, None), (b, scale_b, bias)):
        out = aq._eager_linear_from_shared_rows(x, w, s, z, cache, counted, torch._int_mm)
        torch.testing.assert_close(out, reference(x, w, s, z), rtol=0, atol=0)
    assert counted.call_count == 1
    x.add_(1)
    out = aq._eager_linear_from_shared_rows(x, a, scale_a, None, cache, counted, torch._int_mm)
    torch.testing.assert_close(out, reference(x, a, scale_a, None), rtol=0, atol=0)
    assert counted.call_count == 2
    aq._eager_linear_from_shared_rows(x, a, scale_a, None, cache, counted, torch._int_mm, stream_key=1)
    assert counted.call_count == 3


def test_cpu_and_no_scope_keep_ordinary_linear():
    x = torch.ones(2, 16)
    w = torch.ones(8, 16, dtype=torch.int8)
    scale = torch.ones(1)
    with mock.patch.object(aq, "_eager_int8_backend") as eager:
        assert aq.try_shared_int8_linear(x, w, scale, None) is None
        token = aq._ACTIVE.set((x, {}))
        try:
            assert aq.try_shared_int8_linear(x, w, scale, None) is None
        finally:
            aq._ACTIVE.reset(token)
        eager.assert_not_called()
