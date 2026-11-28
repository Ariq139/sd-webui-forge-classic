from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.image_postprocessing import can_decode_direct_pixels, chw_to_uint8


class NativeVAE:
    def decode(self, x):
        return x

    @staticmethod
    def process_output(x):
        return x


class NativeEngine:
    def __init__(self):
        self.forge_objects = SimpleNamespace(vae=NativeVAE())

    def decode_first_stage(self, x):
        return x

    def decode_first_stage_pixels(self, x):
        return x


@pytest.mark.parametrize("rank", [4, 5])
def test_direct_pixels_only_accepts_native_decode_contract(rank):
    engine = NativeEngine()
    x = torch.zeros([1] * rank)
    assert can_decode_direct_pixels(engine, x, NativeEngine, NativeVAE)
    assert not can_decode_direct_pixels(engine, x.to(torch.int64), NativeEngine, NativeVAE)
    assert not can_decode_direct_pixels(engine, torch.zeros(3), NativeEngine, NativeVAE)
    for target, name in ((engine, "decode_first_stage"), (engine, "decode_first_stage_pixels"),
                         (engine.forge_objects.vae, "decode"), (engine.forge_objects.vae, "process_output")):
        setattr(target, name, lambda v: v)
        assert not can_decode_direct_pixels(engine, x, NativeEngine, NativeVAE)
        delattr(target, name)
    assert can_decode_direct_pixels(engine, x, NativeEngine, NativeVAE)
    class CustomVAE(NativeVAE):
        pass
    engine.forge_objects.vae = CustomVAE()
    assert not can_decode_direct_pixels(engine, x, NativeEngine, NativeVAE)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_image_conversion_exact_on_strided_values_and_boundaries(channels, dtype):
    rng = np.random.default_rng(71)
    image = rng.uniform(0, 1, (channels, 17, 32)).astype(dtype)[:, ::-1, ::2]
    image[:, 0, :4] = [0, 1, 1 / 255, 254 / 255]
    expected = np.repeat(image, 3, axis=0) if channels == 1 else image
    expected = (255.0 * np.moveaxis(expected, 0, 2)).astype(np.uint8)
    actual = chw_to_uint8(image)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert not np.shares_memory(actual, image)
