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
