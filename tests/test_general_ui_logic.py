import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from modules import ui_result_utils as gallery
from modules_forge import native_requests as native


class GeneralUIRegressionTests(unittest.TestCase):
    def test_metadata_validation(self):
        for raw in (None, '', 'not json', '[]', 'null'):
            self.assertEqual(gallery.decode_metadata(raw), {})
        self.assertEqual(gallery.decode_metadata(json.dumps({'infotexts': ['x']})), {'infotexts': ['x']})
        self.assertEqual(gallery.first_image_index({'index_of_first_image': -1}), 0)
        self.assertEqual(gallery.safe_batch_size({'batch_size': 0}), 1)

    def test_selected_save_and_stale_selection(self):
        self.assertEqual(list(gallery.selected_range(4, 2, 1, True)), [2])
        for index in (-1, 0, 4, None, 'bad'):
            self.assertEqual(list(gallery.selected_range(4, index, 1, True)), list(range(4)))
        self.assertEqual(list(gallery.selected_range(4, 2, 1, False)), list(range(4)))
        self.assertEqual(list(gallery.selected_range(0, 0, 0, True)), [])

    def test_auxiliary_metadata_is_optional(self):
        data = {'infotexts': ['first', None, '', 'last']}
        for index in (-1, 1, 2, 4):
            self.assertIsNone(gallery.image_infotext(data, index))
        self.assertEqual(gallery.image_infotext(data, 3), 'last')
        self.assertIsNone(gallery.image_infotext({}, 0))

    def test_seed_fallback(self):
        data = {'Seed': '123', 'Variation seed': '456'}
        self.assertEqual(gallery.reusable_seed(data, True), 123)
        data['Variation seed strength'] = '0.5'
        self.assertEqual(gallery.reusable_seed(data, True), 456)
        self.assertEqual(gallery.reusable_seed(data, False), 123)
        self.assertIsNone(gallery.reusable_seed({}, False))
        self.assertIsNone(gallery.reusable_seed({'Seed': '-1'}, False))

    def test_gallery_insert_after_grid(self):
        old = {'index_of_first_image': 1, 'infotexts': ['grid', 'A', 'B'],
               'all_seeds': [10, 20], 'all_prompts': ['A', 'B']}
        new = {'index_of_first_image': 1, 'all_seeds': [30, 40], 'all_prompts': ['C', 'D']}
        result = gallery.splice_gallery_metadata(old, new, 4, 2, True, ['C', 'D'])
        self.assertEqual(result['infotexts'], ['grid', 'A', 'B', 'C', 'D', None])
        self.assertEqual(result['all_seeds'], [10, 20, 30, 40])
        self.assertEqual(result['all_prompts'], ['A', 'B', 'C', 'D'])
        self.assertEqual(result['index_of_first_image'], 1)
        self.assertEqual(old['all_seeds'], [10, 20])

    def test_gallery_replace_first(self):
        old = {'index_of_first_image': 1, 'infotexts': ['grid', 'A', 'B'],
               'all_seeds': [10, 20], 'width': 256}
        new = {'index_of_first_image': 0, 'all_seeds': [30], 'width': 512}
        result = gallery.splice_gallery_metadata(old, new, 3, 1, False, ['C'])
        self.assertEqual(result['infotexts'], ['grid', 'C', 'B'])
        self.assertEqual(result['all_seeds'], [30, 20])
        self.assertEqual(result['width'], 512)
        self.assertEqual(result['index_of_first_image'], 1)

    def test_repeated_gallery_edit_with_auxiliary_entry(self):
        data = {'infotexts': ['A', None, 'B'], 'all_seeds': [1, 2]}
        result = gallery.splice_gallery_metadata(data, {'all_seeds': [3]}, 3, 2, False, ['C'])
        self.assertEqual(result['infotexts'], ['A', None, 'C'])
        self.assertEqual(result['all_seeds'], [1, 3])

    def test_preview_directory_containment(self):
        with tempfile.TemporaryDirectory() as root:
            parent = Path(root) / 'models'
            parent.mkdir()
            outside = Path(root) / 'models_backup'
            outside.mkdir()
            self.assertTrue(gallery.path_is_child(parent, parent / 'preview.png'))
            self.assertFalse(gallery.path_is_child(parent, outside / 'preview.png'))
            self.assertFalse(gallery.path_is_child(parent, parent / '..' / 'models_backup' / 'preview.png'))
            try:
                (parent / 'link').symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                return
            self.assertFalse(gallery.path_is_child(parent, parent / 'link' / 'preview.png'))

    def test_dynamic_sampler_and_seed_control_indices(self):
        steps, seed, extra = object(), object(), object()
        sampler = type('ScriptSampler', (), {})()
        sampler.steps = steps
        seed_script = type('ScriptSeed', (), {})()
        seed_script.seed = seed
        runner = SimpleNamespace(alwayson_scripts=[seed_script, sampler], inputs=[None, extra, seed, steps])
        self.assertEqual(native.script_control(runner, [0, 17, 812, 30], 'ScriptSampler', 'steps'), 30)
        self.assertEqual(native.script_control(runner, [0, 17, 812, 30], 'ScriptSeed', 'seed'), 812)
        self.assertIsNone(native.script_control(runner, [0], 'ScriptSampler', 'steps'))
        runner.inputs = [None, steps, seed, extra]
        self.assertEqual(native.script_control(runner, [1, 48, 99, 123], 'ScriptSampler', 'steps'), 48)

    def test_native_count_validation(self):
        self.assertEqual(native.positive_integer(3.0, 'Count'), 3)
        self.assertEqual(native.positive_integer('8', 'Count'), 8)
        for value in (0, -1, 1.5, None, float('inf'), 'invalid'):
            with self.assertRaises(ValueError):
                native.positive_integer(value, 'Count')

    def test_video_resize_alignment(self):
        self.assertEqual(native.scaled_video_dimensions(640, 480, 1.5), (960, 736))
        self.assertEqual(native.scaled_video_dimensions(32, 32, 0.5), (64, 64))
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                native.scaled_video_dimensions(640, 480, value)

    def test_page_owner_scope_and_exception_cleanup(self):
        owner = native.PageJobOwner()
        called = []
        callback = lambda: called.append('interrupt')
        self.assertFalse(owner.interrupt('a', callback))
        with self.assertRaises(RuntimeError):
            with owner.active('a'):
                self.assertFalse(owner.interrupt('b', callback))
                self.assertTrue(owner.interrupt('a', callback))
                raise RuntimeError('failed generation')
        self.assertFalse(owner.interrupt('a', callback))
        with owner.active('b'):
            self.assertFalse(owner.interrupt('a', callback))
            self.assertTrue(owner.interrupt('b', callback))
        self.assertEqual(called, ['interrupt', 'interrupt'])


if __name__ == '__main__':
    unittest.main()
