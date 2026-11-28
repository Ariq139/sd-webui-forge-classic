import torch

from backend.sampling.invariant_cache import (
    cached_invariant, invariant_condition, invariant_revision, sampling_invariant_scope,
)


def test_geometry_only_reuses_positions_across_conditions_and_versions():
    module = torch.nn.Identity().eval()
    x = torch.zeros(2, 3)
    with torch.no_grad(), sampling_invariant_scope(1024) as cache, invariant_revision(1):
        with invariant_condition("positive"):
            first = cached_invariant(module, "rope", x, {}, lambda: x.clone(), geometry=(2, 3), geometry_only=True)
        x.add_(1)
        with invariant_condition("negative"):
            second = cached_invariant(module, "rope", x, {}, lambda: x.clone(), geometry=(2, 3), geometry_only=True)
            other = cached_invariant(module, "rope", x, {}, lambda: x.clone(), geometry=(3, 2), geometry_only=True)
        assert first is second and other is not first
        assert len(cache.entries) == 2


def test_text_features_still_separate_conditions():
    module = torch.nn.Identity().eval()
    x = torch.zeros(2, 3)
    with torch.no_grad(), sampling_invariant_scope(1024), invariant_revision(1):
        with invariant_condition("positive"):
            first = cached_invariant(module, "text", x, {}, lambda: x.clone())
        with invariant_condition("negative"):
            second = cached_invariant(module, "text", x, {}, lambda: x + 1)
        assert first is not second
        torch.testing.assert_close(second, torch.ones_like(x))


def test_child_hook_bypasses_and_invalidates_cached_feature():
    module = torch.nn.Identity().eval()
    dependency = torch.nn.Sequential(torch.nn.Identity()).eval()
    x = torch.zeros(2, 3)
    with torch.no_grad(), sampling_invariant_scope(1024) as cache, invariant_revision(1), invariant_condition(1):
        first = cached_invariant(module, "text", x, {}, lambda: dependency(x).clone(), dependencies=(dependency,))
        hook = dependency[0].register_forward_hook(lambda m, inp, out: out + 1)
        try:
            second = cached_invariant(module, "text", x, {}, lambda: dependency(x).clone(), dependencies=(dependency,))
            assert not cache.entries
            torch.testing.assert_close(second, torch.ones_like(x))
        finally:
            hook.remove()
        third = cached_invariant(module, "text", x, {}, lambda: dependency(x).clone(), dependencies=(dependency,))
        assert third is not first
        torch.testing.assert_close(third, first)


def test_unnamespaced_conditions_cannot_cross_contaminate_device_cache():
    from backend.sampling.condition import compile_conditions, compile_weighted_conditions

    positive = torch.full((1, 2, 3), 3.0)
    negative = torch.full_like(positive, 1.0)
    cache = {}
    for compile_fn in (compile_conditions, lambda value: compile_weighted_conditions(value, [[(0, 1.0)]])):
        for value in (positive, negative):
            condition = compile_fn(value)[0]["model_conds"]["c_crossattn"]
            assert condition.cache_key is None
            result = condition.process_cond(1, value.device, cache=cache)
            torch.testing.assert_close(result.cond, value)
    assert not cache
    for namespace, value in (("positive", positive), ("negative", negative)):
        condition = compile_conditions(value, cache_namespace=namespace)[0]["model_conds"]["c_crossattn"]
        a = condition.process_cond(1, value.device, cache=cache)
        b = condition.process_cond(1, value.device, cache=cache)
        assert a.cond is b.cond
        torch.testing.assert_close(b.cond, value)
    assert len(cache) == 2
