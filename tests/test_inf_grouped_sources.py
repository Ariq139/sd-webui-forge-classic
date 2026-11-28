import unittest
from types import SimpleNamespace

from modules.grouped_inputs import source_indices, group_sources


class GroupedSourceTests(unittest.TestCase):
    def test_group_retains_original_slot_order(self):
        self.assertEqual(source_indices([2, 0, 4], 3, 3), [2, 0, 1])
        p = SimpleNamespace(init_images=["a", "b", "c"])
        with group_sources(p, p.init_images, 3, [2, 0]):
            self.assertEqual(p.init_images, ["c", "a"])
        self.assertEqual(p.init_images, ["a", "b", "c"])

    def test_single_source_repeats_inside_initializer(self):
        p = SimpleNamespace(init_images=["a"])
        with group_sources(p, p.init_images, 3, [2, 0]):
            self.assertEqual(p.init_images, ["a"])

    def test_restores_after_failure(self):
        p = SimpleNamespace(init_images=["a", "b", "c"])
        with self.assertRaises(RuntimeError):
            with group_sources(p, p.init_images, 3, [1]):
                self.assertEqual(p.init_images, ["b"])
                raise RuntimeError("failed")
        self.assertEqual(p.init_images, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
