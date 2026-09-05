import unittest
from unittest import mock

import torch

from backend.args import dynamic_args
from backend.sampling.condition import compile_conditions
from backend.sampling.sampling_function import calc_cond_uncond_batch, sampling_function_inner


class Model:
    def memory_required(self, shape):
        return 0

    def apply_model(self, x, timestep, c_crossattn, **kwargs):
        return torch.ones_like(x) * c_crossattn[:, :1, :1, None]


class CFGAllocationTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.zeros(2, 1, 16, 16)
        self.sigma = torch.ones(2)
        self.cond = compile_conditions(torch.full((2, 1, 1), 3.0))
        self.uncond = compile_conditions(torch.full((2, 1, 1), 1.0))
        self.patch = mock.patch.object(dynamic_args, "context_handler", None)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def run_cfg(self, scale=1.0, **kwargs):
        return sampling_function_inner(Model(), self.x, self.sigma, self.uncond, self.cond, scale, **kwargs)

    def test_cfg1_avoids_unconditional_buffers_and_preserves_values(self):
        with mock.patch("torch.zeros_like", wraps=torch.zeros_like) as zeros:
            actual = self.run_cfg()
        self.assertEqual(zeros.call_count, 1)
        torch.testing.assert_close(actual, torch.full_like(self.x, 3), rtol=0, atol=0)

    def test_return_full_and_external_batch_call_preserve_zero_tensor(self):
        result, cond, uncond = self.run_cfg(return_full=True)
        torch.testing.assert_close(result, cond, rtol=0, atol=0)
        torch.testing.assert_close(uncond, torch.zeros_like(self.x), rtol=0, atol=0)
        _, uncond = calc_cond_uncond_batch(Model(), self.cond, None, self.x, self.sigma, {})
        self.assertIsInstance(uncond, torch.Tensor)

    def test_cfg_greater_than_one_and_forced_unconditional_evaluation(self):
        torch.testing.assert_close(self.run_cfg(2), torch.full_like(self.x, 5), rtol=0, atol=0)
        _, _, uncond = self.run_cfg(model_options={"disable_cfg1_optimization": True}, return_full=True)
        torch.testing.assert_close(uncond, torch.ones_like(self.x), rtol=0, atol=0)

    def test_custom_cfg_and_post_cfg_receive_zero_unconditional(self):
        def cfg(arguments):
            torch.testing.assert_close(arguments["uncond_denoised"], torch.zeros_like(self.x))
            return arguments["cond"]

        def post(arguments):
            torch.testing.assert_close(arguments["uncond_denoised"], torch.zeros_like(self.x))
            return arguments["denoised"] + 2

        result = self.run_cfg(model_options={"sampler_cfg_function": cfg, "sampler_post_cfg_function": [post]})
        torch.testing.assert_close(result, torch.full_like(self.x, 5), rtol=0, atol=0)

    def test_pre_cfg_can_restore_unconditional_and_regional_outputs_match(self):
        def pre(model, cond, uncond, x, sigma, options):
            return model, cond, self.uncond, x, sigma, options

        _, _, uncond = self.run_cfg(model_options={"sampler_pre_cfg_function": [pre]}, return_full=True)
        torch.testing.assert_close(uncond, torch.ones_like(self.x), rtol=0, atol=0)
        self.cond[0]["area"] = (8, 8, 0, 0)
        self.cond[0]["strength"] = 0.5
        cond, uncond = calc_cond_uncond_batch(Model(), self.cond, None, self.x, self.sigma, {})
        expected = uncond + (cond - uncond) * 0.5
        torch.testing.assert_close(self.run_cfg(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
