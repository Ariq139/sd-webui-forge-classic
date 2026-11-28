from unittest import mock

import pytest
import torch
from torch._dynamo.exc import TorchDynamoException

from backend import fused_pointwise as fused


@pytest.mark.parametrize("inner", [torch.OutOfMemoryError("allocation failed"), RuntimeError("CUDA error: illegal memory access")])
def test_compiler_wrapped_device_failure_does_not_retry(inner):
    error = TorchDynamoException("compiler failed")
    error.inner_exception = inner
    eager = mock.Mock()
    with mock.patch.object(fused, "fusion_enabled", return_value=True), mock.patch.object(fused, "_compile", side_effect=error), mock.patch.object(fused, "_FAILED", set()):
        with pytest.raises(TorchDynamoException) as raised:
            fused._run(eager, torch.ones(1))
        assert raised.value is error
        assert eager not in fused._FAILED
        eager.assert_not_called()


def test_compiler_unsupported_falls_back_and_stays_disabled():
    def eager(x):
        return x + 1
    x = torch.ones(1)
    with mock.patch.object(fused, "fusion_enabled", return_value=True), mock.patch.object(fused, "_compile", side_effect=ImportError("compiler unavailable")) as compile_mock, mock.patch.object(fused, "_FAILED", set()):
        for _ in range(2):
            torch.testing.assert_close(fused._run(eager, x), x + 1)
        compile_mock.assert_called_once()


def test_actual_pinned_kitchen_eager_equivalence():
    pytest.importorskip("comfy_kitchen")
    from backend import activation_quantization as aq
    eager = aq._eager_int8_backend()
    if eager is None:
        pytest.skip("requires pinned comfy-kitchen 0.2.40")
    torch.manual_seed(11)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(2, 3, 16, dtype=dtype)
        weight = torch.randint(-127, 128, (8, 16), dtype=torch.int8)
        scale = torch.rand(8)
        bias = torch.randn(8, dtype=dtype)
        expected = eager.int8_linear(x, weight, scale, bias, dtype)
        actual = aq._eager_linear_from_shared_rows(x, weight, scale, bias, {}, eager.quantize_int8_rowwise, eager.mm_int8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_actual_cuda_eager_shared_dispatch_and_equivalence():
    pytest.importorskip("comfy_kitchen")
    from comfy_kitchen import registry
    from backend import activation_quantization as aq
    eager = aq._eager_int8_backend()
    if eager is None:
        pytest.skip("requires pinned comfy-kitchen 0.2.40")
    x = torch.randn(1, 64, 128, device="cuda", dtype=torch.float16)
    weight = torch.randint(-127, 128, (64, 128), device=x.device, dtype=torch.int8)
    scale = torch.rand(64, device=x.device)
    with torch.no_grad(), registry.use_backend("eager"), mock.patch.object(aq.args, "shared_activation_quantization", True, create=True):
        expected = eager.int8_linear(x, weight, scale, None, x.dtype)
        with aq.shared_activation_scope(x):
            with mock.patch.object(eager, "quantize_int8_rowwise", wraps=eager.quantize_int8_rowwise) as quantize:
                for _ in range(2):
                    actual = aq.try_shared_int8_linear(x, weight, scale, None)
                    assert actual is not None
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert quantize.call_count == 1


def test_compiler_non_exception_diagnostic_is_safe():
    error = TorchDynamoException("compiler rejected graph")
    error.inner_exception = "unsupported operator"
    assert not fused._device_failure(error)
    error.inner_exception = "CUDA error: illegal memory access"
    assert fused._device_failure(error)
