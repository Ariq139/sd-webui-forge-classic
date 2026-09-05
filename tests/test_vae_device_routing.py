import types
import unittest

import torch

from backend.diffusion_engine.base import ForgeDiffusionEngine


class _FirstStage:
    @staticmethod
    def process_in(value):
        return value

    @staticmethod
    def process_out(value):
        return value


class _VAE:
    def __init__(self):
        self.first_stage_model = _FirstStage()
        self.encode_output_device = None
        self.decode_output_device = None

    def encode(self, value, output_device=None):
        self.encode_output_device = output_device
        return value.movedim(-1, 1)

    def decode(self, value, output_device=None):
        self.decode_output_device = output_device
        return torch.zeros((value.shape[0], value.shape[1], value.shape[2], 3), device=output_device)


class _Engine(ForgeDiffusionEngine):
    def get_learned_conditioning(self, prompt):
        raise NotImplementedError


def _engine():
    engine = object.__new__(_Engine)
    vae = _VAE()
    engine.forge_objects = types.SimpleNamespace(vae=vae)
    engine.is_wan = False
    return engine, vae


class VAEDeviceRoutingTests(unittest.TestCase):
    def test_encode_requests_input_device(self):
        engine, vae = _engine()
        image = torch.zeros((1, 3, 4, 4))

        encoded = engine.encode_first_stage(image)

        self.assertEqual(vae.encode_output_device, image.device)
        self.assertEqual(encoded.device, image.device)

    def test_decode_requests_explicit_output_device(self):
        engine, vae = _engine()
        latent = torch.zeros((1, 4, 2, 2), dtype=torch.float16)

        decoded = engine.decode_first_stage(latent, output_device=torch.device("cpu"))

        self.assertEqual(vae.decode_output_device, torch.device("cpu"))
        self.assertEqual(decoded.device, torch.device("cpu"))
        self.assertEqual(decoded.dtype, latent.dtype)

if __name__ == "__main__":
    unittest.main()
