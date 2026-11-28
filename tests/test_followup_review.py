import pytest
import torch

from backend.packed_weight_identity import TransferSource
from backend.sampling.reconstruction_cache import ReconstructionCache, condition_source_key
from backend.sampling.sigma_transform import SigmaTransform


def test_transfer_snapshot_keeps_rebound_storage_alive():
    weight = torch.nn.Parameter(torch.ones(3, 4))
    source = TransferSource(weight, weight.detach())
    assert source.matches(weight, weight.detach())
    weight.data = torch.zeros_like(weight)
    assert not source.matches(weight, weight.detach())
    torch.testing.assert_close(source.raw, torch.ones(3, 4))


def test_transfer_snapshot_detects_version_and_view():
    weight = torch.nn.Parameter(torch.ones(3, 4), requires_grad=False)
    source = TransferSource(weight, weight.detach())
    assert not source.matches(weight, weight.detach().T)
    weight.add_(1)
    assert not source.matches(weight, weight.detach())


def test_condition_reconstruction_reuses_only_current_schedule():
    value = torch.ones(4)
    cond = [value]
    cache = ReconstructionCache()
    calls = []
    def build():
        calls.append(1)
        return value.clone()
    key = condition_source_key(value)
    first = cache.get(key, cond, None, build, enabled=True)
    assert cache.get(key, cond, None, build, enabled=True) is first
    value.add_(1)
    key = condition_source_key(value)
    second = cache.get(key, cond, None, build, enabled=True)
    torch.testing.assert_close(second, value)
    assert second is not first
    assert cache.get(key, list(cond), None, build, enabled=True) is not second
    cache.get(key, cond, None, build, enabled=False)
    assert len(calls) == 4 and cache.value is None


def test_condition_budget_and_dict_key():
    value = {"crossattn": torch.ones(4), "guidance": torch.ones(1)}
    key = condition_source_key(value)
    value["guidance"].add_(1)
    assert condition_source_key(value) != key
    assert condition_source_key({"custom": torch.ones(1)}) is None
    cache = ReconstructionCache(max_bytes=1)
    cache.get(key, value, None, lambda: value, enabled=True)
    assert cache.value is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("flow", [False, True])
def test_sigma_transform_preserves_scalar_arithmetic_and_refreshes(dtype, flow):
    sigmas = torch.tensor([0.99, 0.7, 0.2, 0], dtype=dtype)
    transform = lambda value: value.logit().neg() if flow else value.log().neg()
    calls = []
    def counted(value):
        calls.append(value.shape)
        return transform(value)
    cache = SigmaTransform(sigmas, counted, True)
    for index in range(len(sigmas)):
        torch.testing.assert_close(cache.value(index), transform(sigmas[index]), rtol=0, atol=0)
    assert len(calls) == 1
    sigmas[1] = 0.4
    torch.testing.assert_close(cache.value(1), transform(sigmas[1]), rtol=0, atol=0)
    assert len(calls) == 2


def test_sigma_transform_fallback_and_predictor_revision():
    sigma = torch.tensor([0.5, 0.2])
    mode = [0]
    cache = SigmaTransform(sigma, lambda x: x + mode[0], True, lambda: mode[0])
    assert cache.value(0) == 0.5
    mode[0] = 1
    assert cache.value(0) == 1.5
    with torch.inference_mode():
        sigma = torch.tensor([0.5, 0.2])
        cache = SigmaTransform(sigma, lambda x: x + 1, True)
        assert cache.value(0) == 1.5 and cache.values is None


def test_storage_metadata_changes_reject_prefetch():
    weight = torch.ones(2, 3)
    stamp = TransferSource(weight, weight, ("Q4", False))
    assert stamp.matches(weight, weight, ("Q4", False))
    assert not stamp.matches(weight, weight, ("Q4", True))


def test_condition_key_detects_data_rebind_without_version_change():
    value = torch.ones(2, 3)
    old = condition_source_key(value)
    version = value._version
    value.data = torch.zeros_like(value)
    assert value._version == version
    assert condition_source_key(value) != old
