import unittest
from unittest import mock

import torch

from backend.patcher.vae import VAE


class _FirstStage:
    def __init__(self):
        self.decoded = None
        self.encoded = None

    def decode(self, samples):
        self.decoded = torch.zeros((samples.shape[0], 3, samples.shape[2] * 2, samples.shape[3] * 2))
        return self.decoded

    def encode(self, pixels):
        self.encoded = torch.zeros((pixels.shape[0], 4, pixels.shape[2] // 2, pixels.shape[3] // 2))
        return self.encoded


def _vae():
    vae = VAE(no_init=True)
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.patcher = object()
    vae.is_wan = False
    vae.first_stage_model = _FirstStage()
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.memory_used_encode = lambda shape, dtype: 1
    return vae


class VAEAllocationPathTests(unittest.TestCase):
    @mock.patch("backend.patcher.vae.memory_management.load_models_gpu")
    @mock.patch("backend.patcher.vae.memory_management.get_free_memory", return_value=1024)
    def test_single_chunk_decode_reuses_decoder_storage(self, _free_memory, _load_models):
        vae = _vae()

        decoded = vae.decode(torch.zeros((2, 4, 2, 2)))

        self.assertEqual(decoded.data_ptr(), vae.first_stage_model.decoded.data_ptr())
        self.assertEqual(tuple(decoded.shape), (2, 4, 4, 3))

    @mock.patch("backend.patcher.vae.memory_management.load_models_gpu")
    @mock.patch("backend.patcher.vae.memory_management.get_free_memory", return_value=1024)
    def test_single_chunk_encode_reuses_encoder_storage(self, _free_memory, _load_models):
        vae = _vae()

        encoded = vae.encode(torch.zeros((2, 4, 4, 3)))

        self.assertIs(encoded, vae.first_stage_model.encoded)
        self.assertEqual(tuple(encoded.shape), (2, 4, 2, 2))

    @mock.patch("backend.patcher.vae.memory_management.load_models_gpu")
    @mock.patch("backend.patcher.vae.memory_management.get_free_memory", return_value=1024)
    @mock.patch("backend.patcher.vae.memory_management.soft_empty_cache")
    def test_decode_oom_retries_tiled_on_requested_device(self, soft_empty_cache, _free_memory, _load_models):
        vae = _vae()
        vae.first_stage_model.decode = mock.Mock(side_effect=torch.OutOfMemoryError("test"))
        expected = torch.zeros((1, 4, 4, 3))

        with mock.patch.object(vae, "decode_tiled", return_value=expected) as decode_tiled:
            decoded = vae.decode(torch.zeros((1, 4, 2, 2)), output_device=torch.device("cpu"))

        self.assertIs(decoded, expected)
        soft_empty_cache.assert_called_once_with(force=True)
        decode_tiled.assert_called_once_with(mock.ANY, output_device=torch.device("cpu"))

    @mock.patch("backend.patcher.vae.memory_management.load_models_gpu")
    @mock.patch("backend.patcher.vae.memory_management.get_free_memory", return_value=1024)
    @mock.patch("backend.patcher.vae.memory_management.soft_empty_cache")
    def test_encode_oom_retries_tiled_on_requested_device(self, soft_empty_cache, _free_memory, _load_models):
        vae = _vae()
        vae.first_stage_model.encode = mock.Mock(side_effect=torch.OutOfMemoryError("test"))
        expected = torch.zeros((1, 4, 2, 2))

        with mock.patch.object(vae, "encode_tiled", return_value=expected) as encode_tiled:
            encoded = vae.encode(torch.zeros((1, 4, 4, 3)), output_device=torch.device("cpu"))

        self.assertIs(encoded, expected)
        soft_empty_cache.assert_called_once_with(force=True)
        encode_tiled.assert_called_once_with(mock.ANY, output_device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
