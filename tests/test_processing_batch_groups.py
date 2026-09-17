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
            from modules import shared_init
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
            p.iteration = 2
            processing._refresh_infinite_batch(p)
            second = list(p.all_prompts)

        self.assertEqual(seen, [("template {one|two}", 2, 102), ("template {one|two}", 2, 104)])
        self.assertEqual(first, ["choice-102"])
        self.assertEqual(second, ["choice-104"])

    def test_aligned_signatures_keep_requested_batches(self):
        signatures = ["a", "a", "b", "b"]

        self.assertIsNone(group_extra_network_batches(signatures, batch_size=2))

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
