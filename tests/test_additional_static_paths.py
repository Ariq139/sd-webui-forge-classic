import torch
import pytest

from backend.single_gpu_fastpaths import requested_layer_stop, schedule_options_safe, uniform_unpatchify


def test_schedule_allowlist():
    assert schedule_options_safe({})
    assert schedule_options_safe({"sampling_sigmas": torch.ones(3)})
    assert not schedule_options_safe({"sampling_sigmas": torch.ones(3), "patches": {}})
    assert not schedule_options_safe(None)


@pytest.mark.parametrize("requested,n,expected", [
    ([9, 18, 27], 36, (27, True)), (-2, 36, (34, False)),
    ([0], 36, (0, True)), ([36], 36, None), (35, 36, None),
    ("all", 36, None), ([], 36, None), ([-1], 36, None),
])
def test_intermediate_layer_conventions(requested, n, expected):
    assert requested_layer_stop(requested, n) == expected


@pytest.mark.parametrize("batch,patch,channels", [(1, 2, 16), (3, 2, 4), (2, 1, 8)])
def test_uniform_unpatchify_matches_original(batch, patch, channels):
    h, w, start = 6, 10, 7
    x = torch.randn(batch, start + h * w // patch**2 + 4, patch**2 * channels)
    end = start + h * w // patch**2
    expected = torch.stack([
        row[start:end].view(h // patch, w // patch, patch, patch, channels)
        .permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2)
        for row in x
    ])
    actual = uniform_unpatchify(x, [(h, w)] * batch, [start] * batch, patch, channels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
    if batch > 1:
        assert uniform_unpatchify(x, [(h, w)] * batch, [start] + [start + 1] * (batch - 1), patch, channels) is None


def test_sigma_control_invalidation_and_churn_boundary(monkeypatch):
    from backend import single_gpu_fastpaths as fast
    monkeypatch.setattr(fast, "single_gpu_flag", lambda name: True)
    sigmas = torch.tensor([1, 0.5, 0], dtype=torch.float16)
    control = fast.SigmaControl(sigmas)
    control.enabled = True  # Test snapshot/version logic independently of CUDA.
    assert control.value(2) == 0
    assert control.in_range(1, 0.50001, 1.0) == bool(0.50001 <= sigmas[1] <= 1.0)
    sigmas[2] = 0.25
    assert control.value(2) == 0.25
    with torch.inference_mode():
        unversioned = torch.tensor([1.0, 0.0])
        other = fast.SigmaControl(unversioned)
        other.enabled = True
        assert isinstance(other.value(1), torch.Tensor)


def test_reference_signature_detects_mutations_and_versions_inference_input(monkeypatch):
    from backend import single_gpu_fastpaths as fast
    monkeypatch.setattr(fast, "single_gpu_flag", lambda name: True)
    ref = torch.zeros(1, 4, 8, 8)
    initial = fast.reference_signature([ref])
    ref.add_(1)
    assert fast.reference_signature([ref]) != initial
    with torch.inference_mode():
        source = torch.ones_like(ref)
        assert fast.reference_signature([source]) is None
        converted, = fast.version_reference_latents([source])
        assert not converted.is_inference()
        assert fast.reference_signature([converted]) is not None
        torch.testing.assert_close(converted, source, rtol=0, atol=0)
