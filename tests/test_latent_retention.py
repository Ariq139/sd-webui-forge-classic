import unittest
import weakref
from types import SimpleNamespace

import torch

from modules.latent_retention import retain_latents, scripts_require_latents


class LatentRetentionTests(unittest.TestCase):
    def test_default_releases_tensors_on_success_and_failure(self):
        for fail in (False, True):
            p = SimpleNamespace(latents_after_sampling=[])
            reference = None
            try:
                with retain_latents(p, False):
                    tensor = torch.ones(1, 4, 2, 2, 2)
                    reference = weakref.ref(tensor)
                    p.latents_after_sampling.append(tensor)
                    del tensor
                    self.assertIsNotNone(reference())
                    if fail:
                        raise RuntimeError("interrupted processing")
            except RuntimeError:
                pass
            self.assertEqual(p.latents_after_sampling, [])
            self.assertIsNone(reference())

    def test_compatibility_mode_keeps_request_latents(self):
        p = SimpleNamespace(latents_after_sampling=[])
        with retain_latents(p, True):
            p.latents_after_sampling.extend([torch.ones(1), torch.zeros(1)])
        self.assertEqual(len(p.latents_after_sampling), 2)

    def test_only_active_script_capabilities_enable_retention(self):
        opt_in = SimpleNamespace(requires_all_latents=True)
        normal = SimpleNamespace(requires_all_latents=False)
        runner = SimpleNamespace(alwayson_scripts=[normal], selectable_scripts=[opt_in])
        self.assertFalse(scripts_require_latents(runner, [0]))
        self.assertTrue(scripts_require_latents(runner, [1]))
        runner.alwayson_scripts.append(opt_in)
        self.assertTrue(scripts_require_latents(runner, []))


if __name__ == "__main__":
    unittest.main()
