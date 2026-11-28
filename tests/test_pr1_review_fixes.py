"""Review regressions: preserve extension contracts and bound group-local cache ownership."""
from unittest import mock

import pytest
import torch

from backend import activation_quantization as aq
from backend import native_int8_rows as nr
from backend.sampling.condition import compile_conditions
from backend.sampling.reconstruction_cache import condition_source_key
from backend.sampling.invariant_cache import (
    cache_dependencies_safe, call_native_with_options, invariant_revision, sampling_invariant_scope,
)


def test_native_options_only_for_original_bound_method_in_active_scope():
    class Native:
        def prepare(self, x, transformer_options=None):
            return transformer_options

    class Legacy(Native):
        def prepare(self, x):
            return x

    x = object()
    for budget in (0, 1024):
        with sampling_invariant_scope(budget), invariant_revision(1):
            assert call_native_with_options(Legacy().prepare, Native.prepare, x, transformer_options={}) is x
            value = call_native_with_options(Native().prepare, Native.prepare, x, transformer_options={})
            assert value == ({} if budget else None)


def test_instance_forward_override_disqualifies_dependencies():
    module = torch.nn.Sequential(torch.nn.Identity()).eval()
    assert cache_dependencies_safe(module)
    module[0].forward = lambda x: x + 1
    assert not cache_dependencies_safe(module)


def test_extended_versioned_condition_has_device_key_not_native_key():
    cond = {"crossattn": torch.ones(1, 2, 4), "vector": torch.ones(1, 4),
            "custom": torch.zeros(1, 2)}
    key = condition_source_key(cond, native_only=False)
    assert key is not None and condition_source_key(cond) is None
    cache = {}
    a = compile_conditions(cond, key)[0]["model_conds"]["c_crossattn"]
    assert a.process_cond(1, torch.device("cpu"), cache=cache).cond is a.process_cond(1, torch.device("cpu"), cache=cache).cond
    assert len(cache) == 1
    cond["custom"].add_(1)
    assert condition_source_key(cond, native_only=False) != key
    with torch.inference_mode():
        cond["custom"] = torch.zeros(1)
    assert condition_source_key(cond, native_only=False) is None
    assert condition_source_key({"crossattn": torch.ones(1).as_subclass(torch.nn.Parameter)}, native_only=False) is None


def test_quantized_activation_budget_counts_versions_and_actual_storage(monkeypatch):
    x = torch.ones(2, 4)
    monkeypatch.setattr(aq, "MAX_SHARED_BYTES", 32)
    cache = {}
    token = aq._ACTIVE.set((x, cache))
    quantize = mock.Mock(side_effect=lambda t, layout, scale: t.clone())
    try:
        first = aq.quantize_shared_activation(x, x, "TensorWiseINT8Layout", None, quantize)
        assert aq.quantize_shared_activation(x, x, "TensorWiseINT8Layout", None, quantize) is first
        x.add_(1)
        second = aq.quantize_shared_activation(x, x, "TensorWiseINT8Layout", None, quantize)
        assert aq.quantize_shared_activation(x, x, "TensorWiseINT8Layout", None, quantize) is not second
        assert len(cache) == 1 and aq._retained_bytes(cache) == 32 and quantize.call_count == 3
    finally:
        aq._ACTIVE.reset(token)
    # Full backing storage, not the view's logical size, is charged.
    assert aq._storage_bytes(torch.empty(128)[:1]) == 512


def test_eager_budget_declines_before_quantizing_and_counts_native_rows(monkeypatch):
    x = torch.ones(2, 16)
    native = (torch.ones(2, 16, dtype=torch.int8), torch.ones(2, 1))
    monkeypatch.setattr(aq, "MAX_SHARED_BYTES", 79)
    cache = {aq._NATIVE_CACHE: {"native": native}}
    quantize = mock.Mock()
    matmul = mock.Mock()
    assert aq._eager_linear_from_shared_rows(x, torch.ones(16, 16, dtype=torch.int8),
                                            torch.ones(1), None, cache, quantize, matmul) is None
    quantize.assert_not_called()
    matmul.assert_not_called()
    assert aq._retained_bytes(cache) == 40
    native_cache = {}
    producer = mock.Mock(return_value=native)
    assert nr.shared_rows(x, False, 0, native_cache, producer, budget=39) is None
    producer.assert_not_called()
    assert nr.shared_rows(x, False, 0, native_cache, producer, budget=40) is native


def test_quantized_buffer_and_scale_storage_accounted():
    class QuantizedTensor:
        _layout_cls = "TensorWiseINT8Layout"
        _qdata = torch.ones(8, dtype=torch.int8)
        _params = type("Params", (), {"scale": torch.ones(2)})()
    assert aq._storage_bytes(QuantizedTensor()) == 16



def test_native_rows_charge_actual_backing_storage(monkeypatch):
    x = torch.ones(2, 16)
    rows = (torch.empty(512, dtype=torch.int8)[:32].reshape(2, 16), torch.ones(2, 1))
    cache = {}
    assert nr.shared_rows(x, False, 0, cache, lambda *_: rows, budget=40) is rows
    assert not cache
