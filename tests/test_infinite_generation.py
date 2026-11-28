import unittest
from types import SimpleNamespace

from modules import infinite_generation as inf


class InfiniteGenerationTests(unittest.TestCase):
    def processing(self, **kw):
        values = dict(prompt="test", negative_prompt="negative", seed=100, subseed=200,
                      all_prompts=["test"], all_negative_prompts=["negative"],
                      all_seeds=[100, 101, 102], all_subseeds=[200, 201, 202],
                      subseed_strength=0, enable_hr=False, infinite_generation=True,
                      batch_size=3, iteration=2)
        values.update(kw)
        return SimpleNamespace(**values)

    def test_static_and_partial_cycles_align_all_arrays(self):
        p = self.processing()
        self.assertEqual(inf.normalize_cycle(p, 3, False, False), 3)
        self.assertEqual(p.all_prompts, ["test"] * 3)
        self.assertEqual(p.all_seeds, [100, 101, 102])
        p.all_prompts = ["a", "b"]
        self.assertEqual(inf.normalize_cycle(p, 3, False, False), 3)
        self.assertEqual(p.all_prompts, ["a", "b", "b"])

    def test_expanded_cycle_and_hr_arrays(self):
        p = self.processing(enable_hr=True, hr_prompt="hr", hr_negative_prompt="hr neg",
                            all_hr_prompts=["stale"], all_hr_negative_prompts=["stale neg"],
                            _infinite_hr_prompts_template=["hr A", "hr B", "hr C"],
                            _infinite_hr_negative_prompts_template=["hr neg"])
        inf.reset_hr_prompts(p, 3)
        p.all_prompts = ["a", "b", "c", "d", "e"]
        self.assertEqual(inf.normalize_cycle(p, 3, False, False), 5)
        self.assertEqual(p.all_hr_prompts, ["hr A", "hr B", "hr C", "hr C", "hr C"])
        self.assertEqual(p.all_hr_negative_prompts, ["hr neg"] * 5)
        self.assertEqual(p.all_seeds, [100, 101, 102, 103, 104])
        self.assertEqual(p.all_subseeds, [200, 201, 202, 203, 204])

    def test_explicit_seed_lists_and_variation(self):
        p = self.processing(all_seeds=[100, 7], all_subseeds=[200], subseed_strength=0.5)
        inf.normalize_cycle(p, 4, True, True)
        self.assertEqual(p.all_seeds, [100, 7, 100, 7])
        self.assertEqual(p.all_subseeds, [200] * 4)
        p = self.processing(all_seeds=[100], all_subseeds=[200], subseed_strength=0.5)
        inf.normalize_cycle(p, 4, False, False)
        self.assertEqual(p.all_seeds, [100] * 4)
        self.assertEqual(p.all_subseeds, [200, 201, 202, 203])

    def test_intermediate_metadata_indices(self):
        p = self.processing()
        self.assertEqual(inf.intermediate_index(p, 1), 1)
        p.infinite_generation = False
        self.assertEqual(inf.intermediate_index(p, 1), 7)
        p._current_batch_indices = [4, 1, 6]
        self.assertEqual(inf.intermediate_index(p, 1), 1)

    def test_retention_rejected_only_for_infinite_requests(self):
        for latents, pixels in ((True, False), (False, True), (True, True)):
            with self.assertRaisesRegex(ValueError, "cannot retain all"):
                inf.require_bounded_retention(True, latents, pixels)
            inf.require_bounded_retention(False, latents, pixels)
        inf.require_bounded_retention(True, False, False)

    def test_graceful_stop_and_state_cleanup(self):
        state = SimpleNamespace(infinite_generation=False, infinite_batch=0, job_count=-1, stopping_generation=False)
        self.assertFalse(inf.stop_after_current(state, True))
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with inf.generation_state(state, True):
                state.infinite_batch = 3
                self.assertEqual(inf.progress_text(state), "Infinite batch 3 (current pass)")
                self.assertTrue(inf.stop_after_current(state, True))
                self.assertFalse(inf.stop_after_current(state, False))
                state.stopping_generation = True
                self.assertFalse(inf.stop_after_current(state, True))
                raise RuntimeError("failure")
        self.assertFalse(state.infinite_generation)
        self.assertEqual(state.infinite_batch, 0)

    def test_completed_result_survives_inflight_mutation(self):
        p = self.processing(extra_result_images=["extra"], pixels_after_sampling=["pixel"], extra_generation_params={"LoRA": "A"})
        images, texts = ["image A"], ["metadata A"]
        snapshot = inf.capture_completed(p, images, texts, None)
        p.all_seeds[:] = [900, 901, 902]
        p.extra_result_images.clear()
        p.pixels_after_sampling.clear()
        p.extra_generation_params["LoRA"] = "B"
        images.clear()
        texts.clear()
        p.iteration = 3
        self.assertEqual(inf.restore_completed(p, snapshot), (["image A"], ["metadata A"], None))
        self.assertEqual(p.all_seeds, [100, 101, 102])
        self.assertEqual(p.extra_result_images, ["extra"])
        self.assertEqual(p.pixels_after_sampling, ["pixel"])
        self.assertEqual(p.extra_generation_params, {"LoRA": "A"})
        self.assertEqual(p.iteration, 2)


if __name__ == "__main__":
    unittest.main()
