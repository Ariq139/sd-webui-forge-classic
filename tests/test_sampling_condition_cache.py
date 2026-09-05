import unittest

import torch

from backend.sampling.condition import Condition, ConditionNoiseShape, compile_conditions
from backend.sampling import sampling_function


class SamplingConditionCacheTests(unittest.TestCase):
    def test_condition_reuses_processed_tensor(self):
        cache = {}
        condition = Condition(torch.arange(4, dtype=torch.float32).reshape(1, 4), cache_key=("prompt", 0))

        first = condition.process_cond(batch_size=2, device=torch.device("cpu"), cache=cache)
        second = condition.process_cond(batch_size=2, device=torch.device("cpu"), cache=cache)

        self.assertIs(first.cond, second.cond)
        self.assertEqual(len(cache), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_condition_transfer_is_reused(self):
        cache = {}
        condition = Condition(torch.ones((1, 4)), cache_key=("prompt", 0))

        first = condition.process_cond(batch_size=2, device=torch.device("cuda"), cache=cache)
        second = condition.process_cond(batch_size=2, device=torch.device("cuda"), cache=cache)

        self.assertEqual(first.cond.device.type, "cuda")
        self.assertEqual(first.cond.data_ptr(), second.cond.data_ptr())
        self.assertEqual(len(cache), 1)

    def test_noise_shape_cache_is_area_specific(self):
        cache = {}
        condition = ConditionNoiseShape(torch.ones((1, 1, 4, 4)), cache_key=("image", 0))

        first = condition.process_cond(batch_size=1, device=torch.device("cpu"), area=(2, 2, 0, 0), cache=cache)
        second = condition.process_cond(batch_size=1, device=torch.device("cpu"), area=(2, 2, 2, 2), cache=cache)

        self.assertIsNot(first.cond, second.cond)
        self.assertEqual(len(cache), 2)

    def test_compiled_fields_have_distinct_cache_keys(self):
        compiled = compile_conditions(
            {
                "crossattn": torch.ones((1, 2, 3)),
                "vector": torch.ones((1, 3)),
                "guidance": torch.ones((1,)),
            },
            cache_namespace=("schedule", 0),
        )[0]["model_conds"]

        keys = {condition.cache_key for condition in compiled.values()}
        self.assertEqual(len(keys), 3)

    def test_custom_conditioning_hooks_disable_cache(self):
        for option, value in (
            ("conditioning_modifiers", [object()]),
            ("sampler_pre_cfg_function", [object()]),
            ("model_function_wrapper", object()),
        ):
            self.assertFalse(sampling_function.condition_cache_allowed({option: value}), option)

        self.assertTrue(sampling_function.condition_cache_allowed({}))


if __name__ == "__main__":
    unittest.main()
