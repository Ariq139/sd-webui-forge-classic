import torch

from backend.sampling.invariant_cache import (
    cached_adapter_cast, cached_invariant, invariant_condition, invariant_revision,
    sampling_invariant_scope,
)


def test_cache_is_scoped_revision_and_condition_sensitive():
    module = torch.nn.Linear(4, 4).eval()
    x = torch.ones(1, 4)
    calls = []

    def compute():
        calls.append(1)
        return x + len(calls)

    with torch.no_grad(), sampling_invariant_scope(4096) as cache, invariant_revision("r1"):
        with invariant_condition("p1"):
            a = cached_invariant(module, "text", x, {}, compute)
            b = cached_invariant(module, "text", x, {}, compute)
            assert a is b
            assert len(calls) == 1
            cached_invariant(module, "text", x, {"patches": {}}, compute)
            assert len(calls) == 2
        with invariant_condition("p2"):
            cached_invariant(module, "text", x, {}, compute)
        with invariant_revision("r2"), invariant_condition("p1"):
            cached_invariant(module, "text", x, {}, compute)
        assert len(calls) == 4
        assert cache.bytes_used > 0
    assert cache.bytes_used == 0 and not cache.entries


def test_budget_tuple_and_alias_protection():
    module = torch.nn.Identity().eval()
    x = torch.ones(4)
    with torch.no_grad(), sampling_invariant_scope(32) as cache, invariant_revision(1), invariant_condition(1):
        a = cached_invariant(module, "text", x, {}, lambda: x.clone(), clone=True)
        a.zero_()
        b = cached_invariant(module, "text", x, {}, lambda: x.zero_(), clone=True)
        torch.testing.assert_close(b, torch.ones(4))
        cached_invariant(module, "kv", x, {}, lambda: (x.clone(), x.clone()))
        assert cache.bytes_used == 16
        assert len(cache.entries) == 1


def test_geometry_and_no_scope_bypass():
    module = torch.nn.Identity().eval()
    x = torch.ones(4)
    with torch.no_grad(), sampling_invariant_scope(1024), invariant_revision(1), invariant_condition(1):
        a = cached_invariant(module, "rope", x, {}, lambda: x + 1, geometry=(2, 2))
        b = cached_invariant(module, "rope", x, {}, lambda: x + 2, geometry=(1, 4))
        assert not torch.equal(a, b)
    assert cached_invariant(module, "rope", x, {}, lambda: x + 3).equal(x + 3)


def test_adapter_version_and_inference_tensor_bypass():
    source = torch.ones(2, 2)
    inputs = torch.ones(1, 2, dtype=torch.float64)
    calls = []

    def cast(t, device, dtype):
        calls.append(1)
        return t.to(device=device, dtype=dtype)

    with torch.no_grad(), sampling_invariant_scope(1024):
        a = cached_adapter_cast(source, inputs, cast)
        b = cached_adapter_cast(source, inputs, cast)
        assert a is b and len(calls) == 1
        source.add_(1)
        c = cached_adapter_cast(source, inputs, cast)
        assert len(calls) == 2 and c.equal(torch.full_like(c, 2))
        with torch.inference_mode():
            t = torch.ones(2, 2)
            cached_adapter_cast(t, inputs, cast)
            cached_adapter_cast(t, inputs, cast)
        assert len(calls) == 4
