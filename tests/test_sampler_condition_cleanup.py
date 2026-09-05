import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


class SamplerConditionCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules_forge" / "packages"))
        with mock.patch.object(sys, "argv", ["verify"]):
            from modules import shared_init
            shared_init.initialize()
            from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers_timesteps
        cls.common = sd_samplers_common
        cls.families = (sd_samplers_kdiffusion.KDiffusionSampler, sd_samplers_timesteps.CompVisSampler)

    def test_both_sampler_families_use_common_launch(self):
        for family in self.families:
            self.assertIs(family.launch_sampling, self.common.Sampler.launch_sampling)

    def check_cleanup(self, device):
        for error in (None, RuntimeError, torch.OutOfMemoryError, RecursionError, self.common.InterruptedException):
            with self.subTest(device=device, error=error):
                cache, references = {}, []
                clear = mock.Mock(wraps=cache.clear)
                sampler = SimpleNamespace(
                    model_wrap_cfg=SimpleNamespace(clear_conditioning_cache=clear),
                    config=SimpleNamespace(total_steps=lambda steps: steps),
                )

                def run():
                    tensor = torch.ones(16, device=device)
                    references.append(weakref.ref(tensor))
                    cache["condition"] = tensor
                    if error:
                        raise error("injected sampling failure")
                    return "result"

                with mock.patch.object(self.common.state, "current_latent", "interrupted result"):
                    if error in (RuntimeError, torch.OutOfMemoryError):
                        with self.assertRaises(error):
                            self.common.Sampler.launch_sampling(sampler, 1, run)
                    else:
                        result = self.common.Sampler.launch_sampling(sampler, 1, run)
                        self.assertEqual(result, "result" if error is None else "interrupted result")
                clear.assert_called_once_with()
                self.assertEqual(cache, {})
                self.assertIsNone(references[0]())

    def test_cleanup_on_success_and_failures(self):
        self.check_cleanup("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_cache_released_on_success_and_failures(self):
        self.check_cleanup("cuda")


if __name__ == "__main__":
    unittest.main()
