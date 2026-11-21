import unittest
from types import SimpleNamespace
from unittest import mock

from modules.batch_planning import (
    effective_output_batch_index,
    effective_output_batch_size,
    effective_output_generation_index,
    effective_prompt_index,
    group_extra_network_batches,
    restore_grouped_output_order,
    select_batch_values,
)


class ProcessingBatchGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules_forge" / "packages"))

        with mock.patch.object(sys, "argv", ["verify"]):
            from modules import shared, shared_init
            if shared.opts is None:
                shared_init.initialize()
            from modules import processing

        cls.processing = processing

    def test_infinite_refresh_reruns_prompt_processing_after_lora_grouping(self):
        processing = self.processing
        p = processing.StableDiffusionProcessingTxt2Img(
            prompt="template {one|two}",
            negative_prompt="negative",
            batch_size=1,
            n_iter=1,
            seed=100,
            subseed=200,
            infinite_generation=True,
        )
        p._infinite_prompt_template = p.prompt
        p._infinite_negative_prompt_template = p.negative_prompt
        p._infinite_batch_size = 2
        p._infinite_base_seeds = [100, 101]
        p._infinite_base_subseeds = [200, 201]

        seen = []

        def process(processing):
            seen.append((processing.prompt, processing.batch_size, processing.seed))
            processing.all_prompts = [f"choice-{processing.seed + i}" for i in range(processing.batch_size)]
            processing.all_negative_prompts = [processing.negative_prompt] * processing.batch_size

        runner = mock.MagicMock()
        runner.process.side_effect = process
        p.scripts_value = runner

        with mock.patch.object(processing, "_extra_network_signature", side_effect=lambda prompt: prompt):
            p.iteration = 1
            processing._refresh_infinite_batch(p)
            first = list(p.all_prompts)
            processing._take_infinite_group(p)
            remaining = list(p.all_prompts)
            p.iteration = 2
            processing._refresh_infinite_batch(p)
            second = list(p.all_prompts)

        self.assertEqual(seen, [("template {one|two}", 2, 102), ("template {one|two}", 2, 104)])
        self.assertEqual(first, ["choice-102"])
        self.assertEqual(remaining, ["choice-103"])
        self.assertEqual(second, ["choice-104"])
        processing._take_infinite_group(p)
        self.assertEqual(p.all_prompts, ["choice-105"])
        self.assertEqual(p.all_seeds, [105])

    def test_aligned_signatures_keep_requested_batches(self):
        signatures = ["a", "a", "b", "b"]

        self.assertIsNone(group_extra_network_batches(signatures, batch_size=2))

    def test_infinite_groups_preserve_order_and_cap_batch_size(self):
        p = self.processing.StableDiffusionProcessingTxt2Img(
            prompt="test", batch_size=2, n_iter=1, seed=10, subseed=20, infinite_generation=True,
        )
        p.all_prompts = ["a0", "a1", "a2", "b3", "a4"]
        p.all_negative_prompts = [f"n{i}" for i in range(5)]
        p.all_seeds = list(range(10, 15))
        p.all_subseeds = list(range(20, 25))
        with mock.patch.object(self.processing, "_extra_network_signature", side_effect=lambda s: s[0]):
            self.processing._split_batches_for_extra_networks(p)
        groups = []
        while True:
            groups.append((list(p.all_prompts), list(p.all_seeds), list(p.all_negative_prompts)))
            if not p._infinite_pending_groups:
                break
            self.processing._take_infinite_group(p)
        self.assertEqual(groups, [
            (["a0", "a1"], [10, 11], ["n0", "n1"]),
            (["a2"], [12], ["n2"]),
            (["b3"], [13], ["n3"]),
            (["a4"], [14], ["n4"]),
        ])

    def test_infinite_refresh_advances_seeds_by_expanded_count(self):
        p = self.processing.StableDiffusionProcessingTxt2Img(
            prompt="template", negative_prompt="", batch_size=2, n_iter=1,
            seed=100, subseed=200, infinite_generation=True,
        )
        p._infinite_prompt_template = p.prompt
        p._infinite_negative_prompt_template = ""
        p._infinite_batch_size = 2
        p._infinite_base_seeds = [100, 101]
        p._infinite_base_subseeds = [200, 201]
        p._infinite_next_seed_offset = 2
        seen = []
        def expand(processing):
            seen.append(processing.seed)
            processing.all_prompts = [f"a{i}" for i in range(4)]
            processing.all_negative_prompts = [""] * 4
            processing.all_seeds = list(range(processing.seed, processing.seed + 4))
            processing.all_subseeds = list(range(processing.subseed, processing.subseed + 4))
        p.scripts_value = mock.MagicMock()
        p.scripts.process.side_effect = expand
        with mock.patch.object(self.processing, "_extra_network_signature", return_value="a"):
            self.processing._refresh_infinite_batch(p)
            self.assertEqual(p.all_seeds, [102, 103])
            self.processing._take_infinite_group(p)
            self.assertEqual(p.all_seeds, [104, 105])
            self.processing._refresh_infinite_batch(p)
        self.assertEqual(seen, [102, 106])

    def test_mixed_signatures_are_grouped_in_first_seen_order(self):
        signatures = ["a", "b", "a", "b", "a"]

        plan = group_extra_network_batches(signatures, batch_size=2)

        self.assertEqual(plan, [[0, 2], [4], [1, 3]])
        self.assertTrue(all(len({signatures[index] for index in batch}) == 1 for batch in plan))

    def test_output_bundles_and_infotexts_restore_original_order(self):
        images = ["image-0", "mask-0", "image-2", "image-1"]
        image_records = [
            (0, 0, 0, 2),
            (2, 1, 2, 3),
            (1, 2, 3, 4),
        ]
        infotext_records = [
            (0, 0, "info-0"),
            (2, 1, "info-2"),
            (1, 2, "info-1"),
        ]

        ordered_images, ordered_infotexts = restore_grouped_output_order(images, image_records, infotext_records)

        self.assertEqual(ordered_images, ["image-0", "mask-0", "image-1", "image-2"])
        self.assertEqual(ordered_infotexts, ["info-0", "info-1", "info-2"])

    def test_grouped_batch_values_follow_original_indices(self):
        values = [10, 11, 12, 13]

        selected = select_batch_values(values, iteration=1, batch_size=2, batch_indices=[0, 2])

        self.assertEqual(selected, [10, 12])

    def test_grouped_filename_values_follow_original_batch(self):
        processing = SimpleNamespace(
            _current_output_index=3,
            _extra_network_original_batch_size=2,
            all_seeds=[10, 11, 12, 13],
            batch_index=0,
            batch_size=1,
            iteration=3,
            n_iter=4,
        )
        self.assertEqual(effective_output_batch_size(processing), 2)
        self.assertEqual(effective_output_batch_index(processing), 1)
        self.assertEqual(effective_output_generation_index(processing), 3)

    def test_grouped_prompt_metadata_uses_original_index(self):
        all_prompts = ["zero", "one", "two", "three"]
        processing = SimpleNamespace(
            all_prompts=all_prompts,
            _current_batch_indices=[1, 3],
        )

        self.assertEqual(effective_prompt_index(processing, 1, ["one", "three"]), 3)
        self.assertEqual(effective_prompt_index(processing, 1, all_prompts), 1)


if __name__ == "__main__":
    unittest.main()
