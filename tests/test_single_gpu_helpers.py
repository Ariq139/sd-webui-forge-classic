"""CPU helper regressions and optional CUDA stream/compilation smoke tests."""
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from backend import activation_quantization as aq
from backend import fused_pointwise as fused
from backend import weight_prefetch as wp
from backend.image_postprocessing import chw_to_uint8
from backend.sampling.invariant_cache import (
    cached_invariant, invariant_condition, invariant_revision, sampling_invariant_scope,
)


def test_invariant_mutation_hooks_and_revision_transition():
    module = torch.nn.Identity().eval()
    x = torch.ones(4)
    with torch.no_grad(), sampling_invariant_scope(1024) as cache, invariant_condition("prompt"):
        with invariant_revision("r1"):
            first = cached_invariant(module, "text", x, {}, lambda: x.clone())
            x.add_(1)
            second = cached_invariant(module, "text", x, {}, lambda: x.clone())
            assert not torch.equal(first, second)
            with mock.patch.object(module, "_forward_hooks", {1: lambda *args: None}):
                bypass = cached_invariant(module, "text", x, {}, lambda: x + 10)
                torch.testing.assert_close(bypass, x + 10)
        with invariant_revision(None):
            assert not cache.entries
        with invariant_revision("r1"):
            fresh = cached_invariant(module, "text", x, {}, lambda: x + 5)
            torch.testing.assert_close(fresh, x + 5)
    assert not cache.entries


def test_invariant_cleanup_on_exception():
    module = torch.nn.Identity().eval()
    x = torch.ones(4)
    with pytest.raises(ValueError), torch.no_grad(), sampling_invariant_scope(1024) as cache:
        with invariant_revision(1), invariant_condition(1):
            cached_invariant(module, "rope", x, {}, lambda: x.clone())
            raise ValueError("interrupt")
    assert not cache.entries and cache.bytes_used == 0


def test_activation_cache_matches_layout_scale_source_and_version():
    x = torch.ones(2, 3)
    quantize = mock.Mock(side_effect=lambda t, layout, scale: t.clone())
    token = aq._ACTIVE.set((x, {}))  # Exercise key logic without a CUDA device.
    try:
        a = aq.quantize_shared_activation(x, x, "TensorWiseINT8Layout", None, quantize)
        b = aq.quantize_shared_activation(x, x, "TensorWiseINT8Layout", None, quantize)
        assert a is b and quantize.call_count == 1
        x.add_(2)
        c = aq.quantize_shared_activation(x, x, "TensorWiseINT8Layout", None, quantize)
        assert c is not a and torch.equal(c, x)
        aq.quantize_shared_activation(x, x, "TensorWiseINT8Layout", torch.ones(1), quantize)
        aq.quantize_shared_activation(x, x, "different-layout", None, quantize)
        y = x.clone()
        aq.quantize_shared_activation(y, y, "TensorWiseINT8Layout", None, quantize)
        assert quantize.call_count == 5
    finally:
        aq._ACTIVE.reset(token)
    assert aq._ACTIVE.get() is None


def test_shared_activation_scope_cpu_disabled():
    x = torch.ones(2, 3)
    with mock.patch.object(aq.args, "shared_activation_quantization", True, create=True), torch.no_grad():
        with aq.shared_activation_scope(x):
            assert aq._ACTIVE.get() is None
    assert aq._ACTIVE.get() is None


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_uint8_matches_legacy_on_strided_images(channels, dtype):
    rng = np.random.default_rng(7)
    image = rng.random((channels, 19, 31)).astype(dtype)[:, ::-1, ::2]
    expected = np.repeat(image, 3, axis=0) if channels == 1 else image
    expected = (255.0 * np.moveaxis(expected, 0, 2)).astype(np.uint8)
    result = chw_to_uint8(image)
    np.testing.assert_array_equal(result, expected)
    assert result.flags.c_contiguous


def test_uint8_boundary_values_and_nan():
    values = np.arange(256, dtype=np.float32) / np.float32(255)
    values = np.concatenate((values, np.nextafter(values, -np.inf), np.array([np.nan, np.inf, -np.inf], dtype=np.float32)))
    image = values.reshape(1, 1, -1)
    with np.errstate(invalid="ignore", over="ignore"):
        expected = (255.0 * np.moveaxis(np.repeat(image, 3, axis=0), 0, 2)).astype(np.uint8)
        result = chw_to_uint8(image)
    np.testing.assert_array_equal(result, expected)


def test_pointwise_eager_preserves_norm_module_hooks_and_values():
    x = torch.randn(2, 3, 8)
    scale = torch.randn(2, 1, 8)
    shift = torch.randn_like(scale)
    norm = torch.nn.LayerNorm(8)
    hook = mock.Mock(return_value=None)
    handle = norm.register_forward_hook(hook)
    try:
        expected = torch.addcmul(shift, F.layer_norm(x, (8,), norm.weight, norm.bias, norm.eps), 1 + scale)
        with mock.patch.object(fused.args, "fused_denoiser_pointwise", True, create=True), torch.no_grad():
            actual = fused.norm_affine(norm, x, scale, shift)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert hook.call_count == 1
    finally:
        handle.remove()
    rms = torch.nn.RMSNorm(8)
    raw_scale = scale.squeeze(1)
    torch.testing.assert_close(fused.norm_scale(rms, x, raw_scale), rms(x) * (1 + scale), rtol=0, atol=0)


def test_pointwise_fp16_nan_and_infinity_fix():
    x = torch.tensor([1000, float("nan"), -1000], dtype=torch.float16)
    y = torch.tensor([1000, 1, 1], dtype=torch.float16)
    expected = torch.nan_to_num(F.silu(x) * y, nan=0.0, posinf=65504, neginf=-65504)
    torch.testing.assert_close(fused.silu_gate(x, y, fix_fp16=True), expected, rtol=0, atol=0)


def test_prefetch_disabled_on_cpu_and_unknown_options():
    with mock.patch.object(wp.args, "weight_prefetch_mib", 64, create=True), torch.no_grad():
        with mock.patch.object(torch.cuda, "Stream") as stream:
            with wp.weight_prefetch_scope(torch.device("cpu"), {}):
                assert wp._ACTIVE.get() is None
            with wp.weight_prefetch_scope(torch.device("cuda:0"), {"patches": {}}):
                assert wp._ACTIVE.get() is None
            stream.assert_not_called()


def test_prefetch_take_event_and_version_guard_with_mock_stream():
    source = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    module = torch.nn.Linear(2, 2)
    device = torch.device("cuda:0")
    value = mock.Mock(dtype=source.dtype)
    ready = mock.Mock()
    ready.query.return_value = False
    done = mock.Mock()
    done.query.return_value = False
    with mock.patch.object(torch.cuda, "Event", return_value=done), mock.patch.object(torch.cuda, "current_stream") as current:
        state = wp.WeightPrefetch(device, 1024)
        state.pending[module] = (source, source._version, value, ready, False, 16)
        state.bytes = 16
        assert state.take(module, source, device, source.dtype) is value
        current.return_value.wait_event.assert_called_once_with(ready)
        value.record_stream.assert_called_once_with(current.return_value)
        assert state.bytes == 16 and not state.pending and module in state.in_use
        state.discard(module)
        done.record.assert_called_once_with(current.return_value)
        assert state.bytes == 16 and not state.in_use
        done.query.return_value = True
        state._reap()
        assert state.bytes == 0 and not state.retired
        state.pending[module] = (source, source._version, value, ready, False, 16)
        state.bytes = 16
        source.add_(1)
        assert state.take(module, source, device, source.dtype) is None
        assert state.bytes == 16 and not state.pending
        current.return_value.wait_event.assert_called_once_with(ready)
        ready.query.return_value = True
        state._reap()
        assert state.bytes == 0
        state.close()
        current.return_value.wait_stream.assert_not_called()


def test_prefetch_close_waits_only_for_unfinished_copies():
    state = wp.WeightPrefetch(torch.device("cuda:0"), 1024)
    state.stream = mock.Mock()
    state.last_copy = mock.Mock()
    state.last_copy.query.return_value = False
    state.close()
    state.last_copy.synchronize.assert_called_once_with()
    state.stream.synchronize.assert_not_called()
    state.last_copy = None
    state.close()
    state.stream.synchronize.assert_called_once_with()


def test_prefetch_without_candidates_avoids_cuda_calls():
    import sys
    from types import SimpleNamespace

    device = torch.device("cuda:0")
    with mock.patch.dict(sys.modules, {"backend.operations_gguf": SimpleNamespace(ParameterGGUF=type("ParameterGGUF", (), {}))}), mock.patch.object(torch.cuda, "Stream") as stream, mock.patch.object(torch.cuda, "mem_get_info") as free:
        state = wp.WeightPrefetch(device, 1024)
        # A mismatched dtype is rejected before any transfer preparation.
        state.stage(torch.nn.Linear(2, 2, dtype=torch.float32), torch.float16)
        state.close()
        stream.assert_not_called()
        free.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefetch_cuda_pinned_copy_and_cleanup():
    device = torch.device("cuda:0")
    if torch.cuda.mem_get_info(device)[0] < wp._MIN_HEADROOM + 1024 * 1024:
        pytest.skip("not enough free GPU memory for experimental prefetch")
    layer = torch.nn.Linear(8, 4, bias=False, dtype=torch.float16)
    layer.weight = torch.nn.Parameter(layer.weight.detach().pin_memory(), requires_grad=False)
    state = wp.WeightPrefetch(device, 1024 * 1024)
    with torch.no_grad():
        try:
            state.stage(layer, torch.float16)
            assert layer in state.pending
            weight = state.take(layer, layer.weight, device, torch.float16)
            assert weight is not None
            torch.testing.assert_close(weight.cpu(), layer.weight)
            assert state.bytes > 0
            state.discard(layer)
            state.stage(layer, torch.float16)
            state.discard(layer)
            assert not state.pending
        finally:
            state.close()
    assert layer.weight.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_activation_cuda_scope_and_fused_values():
    x = torch.randn(2, 3, 8, device="cuda", dtype=torch.float16)
    quantize = mock.Mock(side_effect=lambda t, layout, scale: t.clone())
    with torch.no_grad(), mock.patch.object(aq.args, "shared_activation_quantization", True, create=True):
        with aq.shared_activation_scope(x):
            a = aq.quantize_shared_activation(x, x, "TensorWiseINT8Layout", None, quantize)
            b = aq.quantize_shared_activation(x, x, "TensorWiseINT8Layout", None, quantize)
            assert a is b and quantize.call_count == 1
    with torch.no_grad(), mock.patch.object(fused.args, "fused_denoiser_pointwise", True, create=True):
        y = torch.randn_like(x)
        torch.testing.assert_close(fused.silu_gate(x, y), F.silu(x) * y, rtol=0.01, atol=0.005)
        if fused._silu_gate in fused._FAILED:
            pytest.skip("torch.compile unavailable; eager fallback used")
