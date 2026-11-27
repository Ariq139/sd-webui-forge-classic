import unittest

import torch

from backend.sampling.condition import ConditionCrossAttn
from backend.sampling.krea_cache import (
    _SamplingCache,
    cached_krea_text_features,
    clear_current_krea_cache,
    krea_text_fusion_condition,
    krea_text_fusion_revision,
    sampling_krea_cache_scope,
)
from backend.sampling.sampling_function import get_area_and_mult
from backend.nn.flux import EmbedND
from backend.nn.krea import _imgids, clear_krea_position_cache, krea_position_frequencies


class KreaTextFusionCacheTests(unittest.TestCase):
    def setUp(self):
        self.module = torch.nn.Identity().eval()
        self.calls = 0

    def compute(self):
        self.calls += 1
        return torch.full((2, 3), float(self.calls))

    def test_reuses_features_for_rebuilt_condition_tensor(self):
        first_context = torch.ones((1, 4, 8))
        second_context = torch.ones((1, 4, 8))
        options = {"cond_or_uncond": [0], "sigmas": torch.tensor([1.0]), "cond_mark": None,
                   "cond_indices": [0], "uncond_indices": []}

        with torch.no_grad(), sampling_krea_cache_scope(("model", "revision")):
            with krea_text_fusion_condition(("prompt", "positive")):
                first = cached_krea_text_features(self.module, first_context, options, self.compute)
                second = cached_krea_text_features(self.module, second_context, options, self.compute)

        self.assertEqual(self.calls, 1)
        self.assertIs(first, second)

    def test_in_place_text_fusion_reuses_only_equivalent_rebuilt_inputs(self):
        options = {"cond_or_uncond": [0]}

        def mutate(context):
            self.calls += 1
            context.add_(1)
            return context.clone()

        first_context = torch.ones((1, 4, 8))
        rebuilt_context = torch.ones_like(first_context)
        with torch.no_grad(), sampling_krea_cache_scope(("model", "revision")), krea_text_fusion_condition(("prompt",)):
            first = cached_krea_text_features(self.module, first_context, options, lambda: mutate(first_context))
            rebuilt = cached_krea_text_features(self.module, rebuilt_context, options, lambda: mutate(rebuilt_context))
            mutated_input = cached_krea_text_features(self.module, first_context, options, lambda: mutate(first_context))

        self.assertEqual(self.calls, 2)
        torch.testing.assert_close(first, torch.full_like(first, 2))
        torch.testing.assert_close(rebuilt, first)
        torch.testing.assert_close(mutated_input, torch.full_like(mutated_input, 3))
        torch.testing.assert_close(rebuilt_context, torch.ones_like(rebuilt_context))

    def test_conditioning_group_and_extension_options_invalidate_cache(self):
        context = torch.ones((1, 4, 8))
        options = {"cond_or_uncond": [0], "sigmas": torch.tensor([1.0]), "cond_mark": None,
                   "cond_indices": [0], "uncond_indices": []}
        with torch.no_grad(), sampling_krea_cache_scope(("model", "revision")):
            with krea_text_fusion_condition(("prompt", "positive")):
                cached_krea_text_features(self.module, context, options, self.compute)
            with krea_text_fusion_condition(("prompt", "negative")):
                cached_krea_text_features(self.module, context, options, self.compute)
                extended = dict(options, optimized_attention_override=object())
                cached_krea_text_features(self.module, context, extended, self.compute)

        self.assertEqual(self.calls, 3)

    def test_model_revision_invalidates_cached_features(self):
        context = torch.ones((1, 4, 8))
        with torch.no_grad(), sampling_krea_cache_scope(("model", "scope")), krea_text_fusion_condition(("prompt",)):
            with krea_text_fusion_revision(("patcher", "lora-a")):
                cached_krea_text_features(self.module, context, {}, self.compute)
            with krea_text_fusion_revision(("patcher", "lora-b")):
                cached_krea_text_features(self.module, context, {}, self.compute)
        self.assertEqual(self.calls, 2)

    def test_grad_enabled_and_training_module_bypass_cache(self):
        context = torch.ones((1, 4, 8))
        options = {"cond_or_uncond": [0]}
        with sampling_krea_cache_scope(("model", "revision")), krea_text_fusion_condition(("prompt",)):
            cached_krea_text_features(self.module, context, options, self.compute)
            cached_krea_text_features(self.module, context, options, self.compute)
        self.assertEqual(self.calls, 2)

        self.module.train()
        with torch.no_grad(), sampling_krea_cache_scope(("model", "revision")), krea_text_fusion_condition(("prompt",)):
            cached_krea_text_features(self.module, context, options, self.compute)
            cached_krea_text_features(self.module, context, options, self.compute)
        self.assertEqual(self.calls, 4)

    def test_scope_cleanup_and_explicit_oom_clear(self):
        with torch.no_grad(), sampling_krea_cache_scope(("model", "revision")) as cache:
            with krea_text_fusion_condition(("prompt",)):
                cached_krea_text_features(self.module, torch.ones((1, 4, 8)), {}, self.compute)
            self.assertTrue(cache.entries)
            clear_current_krea_cache()
            self.assertFalse(cache.entries)
        self.assertFalse(cache.entries)
        self.assertEqual(cache.bytes_used, 0)

    def test_cache_lru_respects_memory_bound(self):
        cache = _SamplingCache(("model",), max_bytes=16)
        first = torch.ones(4, dtype=torch.float32)
        second = torch.zeros(4, dtype=torch.float32)
        cache.put("first", first)
        cache.put("second", second)
        self.assertIsNone(cache.get("first"))
        self.assertIs(cache.get("second"), second)
        self.assertEqual(cache.bytes_used, 16)

    def test_krea_position_frequencies_reuse_shape_and_match_direct_embedding(self):
        embedder = EmbedND(dim=12, theta=10000, axes_dim=[4, 4, 4])
        device = torch.device("cpu")
        clear_krea_position_cache(embedder)
        with torch.no_grad():
            first = krea_position_frequencies(embedder, 3, 2, 3, (), device)
            second = krea_position_frequencies(embedder, 3, 2, 3, (), device)
            other_shape = krea_position_frequencies(embedder, 3, 3, 3, (), device)
        position_ids = torch.cat((torch.zeros(1, 3, 3), _imgids(1, 0, 2, 3, device)), dim=1)
        torch.testing.assert_close(first, embedder(position_ids))
        self.assertIs(first, second)
        self.assertNotEqual(tuple(first.shape), tuple(other_shape.shape))
        clear_krea_position_cache(embedder)

    def test_sampling_scope_releases_position_cache(self):
        embedder = EmbedND(dim=12, theta=10000, axes_dim=[4, 4, 4])
        device = torch.device("cpu")
        with torch.no_grad():
            with sampling_krea_cache_scope(("first",), position_embedder=embedder):
                first = krea_position_frequencies(embedder, 3, 2, 3, (), device)
                self.assertIs(first, krea_position_frequencies(embedder, 3, 2, 3, (), device))
            with sampling_krea_cache_scope(("second",), position_embedder=embedder):
                second = krea_position_frequencies(embedder, 3, 2, 3, (), device)
        self.assertIsNot(first, second)

    def test_area_group_carries_stable_condition_key(self):
        cond = torch.ones((1, 4, 8))
        condition = ConditionCrossAttn(cond, cache_key=(("scheduled", 12), "crossattn"))
        result = get_area_and_mult(
            {"model_conds": {"c_crossattn": condition}},
            torch.zeros((1, 4, 2, 2)),
            torch.tensor([1.0]),
        )
        self.assertEqual(result.condition_cache_key, (("c_crossattn", (("scheduled", 12), "crossattn")),))


if __name__ == "__main__":
    unittest.main()
