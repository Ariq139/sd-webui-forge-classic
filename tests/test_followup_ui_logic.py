import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from modules.batch_gallery import BatchGallery
from modules.video_stream import first_frame, encode_frames
from modules_forge import native_validation as native


class BatchGalleryTests(unittest.TestCase):
    def result(self, prefix, seeds):
        return SimpleNamespace(images=[prefix + 'grid', *[prefix + str(seed) for seed in seeds]],
                               extra_images=[prefix + 'aux'], index_of_first_image=1,
                               infotexts=[prefix + 'grid metadata', *[str(seed) for seed in seeds]],
                               all_prompts=[prefix] * len(seeds), all_negative_prompts=[''] * len(seeds),
                               all_seeds=seeds, all_subseeds=[seed + 10 for seed in seeds], info='old')

    def test_multiple_grids_and_auxiliary_entries(self):
        gallery = BatchGallery(-1)
        gallery.add(self.result('A', [1, 2]))
        gallery.add(self.result('B', [3, 4]))
        result = gallery.finish()
        self.assertEqual(result.images, ['Agrid', 'A1', 'A2', 'Bgrid', 'B3', 'B4'])
        self.assertEqual(result.extra_images, ['Aaux', 'Baux'])
        self.assertEqual(result.gallery_kinds, ['grid', 'image', 'image', 'grid', 'image', 'image', 'auxiliary', 'auxiliary'])
        self.assertEqual(result.infotexts, ['Agrid metadata', '1', '2', 'Bgrid metadata', '3', '4', None, None])
        self.assertEqual(result.all_seeds, [1, 2, 3, 4])
        self.assertEqual(result.all_prompts, ['A', 'A', 'B', 'B'])
        self.assertEqual(result.all_subseeds, [11, 12, 13, 14])
        self.assertEqual(result.index_of_first_image, 1)

    def test_result_limit_and_disabled_gallery(self):
        for limit in (0, 1, 2, 4, 6):
            gallery = BatchGallery(limit)
            gallery.add(self.result('A', [1, 2]))
            gallery.add(self.result('B', [3, 4]))
            result = gallery.finish()
            self.assertEqual(len(result.images) + len(result.extra_images), limit)
            self.assertEqual(len(result.gallery_kinds), limit)
            self.assertEqual(len(result.infotexts), limit)
            self.assertEqual(len(result.all_seeds), result.gallery_kinds.count('image'))
        self.assertIsNone(BatchGallery(-1).finish())

    def test_interleaved_inpaint_previews_do_not_consume_seed_slots(self):
        result = self.result('A', [1, 2])
        mask = SimpleNamespace(_forge_auxiliary=True)
        result.images.insert(2, mask)
        result.infotexts.insert(2, 'mask metadata')
        gallery = BatchGallery(-1)
        gallery.add(result)
        result = gallery.finish()
        self.assertEqual(result.images, ['Agrid', 'A1', 'A2'])
        self.assertEqual(result.extra_images, [mask, 'Aaux'])
        self.assertEqual(result.all_seeds, [1, 2])
        self.assertEqual(result.infotexts, ['Agrid metadata', '1', '2', 'mask metadata', None])

    def test_missing_metadata_never_borrows_other_image_seed(self):
        result = self.result('A', [1, 2])
        result.all_seeds = [1]
        result.infotexts = ['grid', 'first']
        gallery = BatchGallery(-1)
        gallery.add(result)
        result = gallery.finish()
        self.assertEqual(result.all_seeds, [1, None])
        self.assertEqual(result.infotexts, ['grid', 'first', None, None])


class NativeValidationTests(unittest.TestCase):
    def test_ltx_modes_use_model_config(self):
        self.assertTrue(native.ltx_cross_timestep({'cross_attn_mod': True}))
        self.assertTrue(native.ltx_cross_timestep(SimpleNamespace(audio_cross_attn_mod=True)))
        self.assertFalse(native.ltx_cross_timestep({'cross_attn_mod': False}))
        self.assertFalse(native.ltx_cross_timestep({'cross_attn_mod': True}, 'LTX-2.0'))
        self.assertTrue(native.ltx_cross_timestep({}, 'LTX-2.3'))
        with self.assertRaises(ValueError):
            native.ltx_cross_timestep({}, 'unknown')

    def test_ltx_frame_geometry_and_numeric_inputs(self):
        self.assertEqual(native.ltx_dimensions(768, 512, 121, 24, 30), (768, 512, 121, 24, 30))
        for values in ((768, 512, 122, 24, 30), (768, 512, 9, 0, 30), (768, 512, 9, 24.5, 30),
                       (768, 512, 9, 24, 0), (767, 512, 9, 24, 30), (768, 512, 9, 24, float('nan'))):
            with self.assertRaises(ValueError):
                native.ltx_dimensions(*values)
        self.assertEqual(native.ltx_dimensions(768, 512, 17, 24, 30, 16)[2], 17)

    def test_audio_zero_is_not_silently_ignored(self):
        self.assertEqual(native.audio_override(0, 0, 'Audio STG'), 0)
        self.assertEqual(native.audio_override(1, 3, 'Audio CFG'), 1)
        with self.assertRaisesRegex(ValueError, 'Diffusers 0.39.0'):
            native.audio_override(0, 1, 'Audio STG')

    def test_ideogram_schedule_matches_pinned_recipe(self):
        self.assertEqual(native.ideogram_guidance(48, 7, True), {'guidance_scale': None, 'guidance_schedule': [7.] * 45 + [3.] * 3})
        self.assertEqual(native.ideogram_guidance(2, 7, True)['guidance_schedule'], [7., 3.])
        self.assertEqual(native.ideogram_guidance(1, 7, True)['guidance_schedule'], [7.])
        self.assertEqual(native.ideogram_guidance(20, 2, True)['guidance_schedule'], [2.] * 20)
        self.assertEqual(native.ideogram_guidance(20, 5, False), {'guidance_scale': 5., 'guidance_schedule': None})

    def test_seed_resolution(self):
        self.assertEqual(native.actual_seed(0), 0)
        self.assertEqual(native.actual_seed('9223372036854775807'), 2**63 - 1)
        for value in (-1, '-1', None, ''):
            self.assertTrue(0 <= native.actual_seed(value) < 2**31)
        for value in (-2, 1.5, 'bad', 2**63, float('inf')):
            with self.assertRaises(ValueError):
                native.actual_seed(value)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg tools required')
class VideoStreamTests(unittest.TestCase):
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '32x32', '-r', '30000/1001', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p']

    def frames(self, count):
        for i in range(count):
            yield np.full((32, 32, 3), i, dtype=np.uint8)

    def test_real_stream_export_and_fractional_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'clip.mp4'
            shape, frames = first_frame(self.frames(7))
            encode_frames(self.command, output, shape, frames)
            probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(output)])
            video = json.loads(probe)['streams'][0]
            self.assertEqual(video['nb_frames'], '7')
            self.assertEqual(video['avg_frame_rate'], '30000/1001')
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_audio_copy_preserves_stream_and_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'audio.m4a'
            output = Path(directory) / 'clip.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
                            '-i', 'anullsrc=r=48000:cl=stereo', '-t', '0.234', '-c:a', 'aac', str(audio)], check=True)
            command = self.command[:self.command.index("-") + 1] + ['-i', str(audio), '-map', '0:v', '-map', '1:a?', '-acodec', 'copy'] + self.command[self.command.index("-") + 1:]
            shape, frames = first_frame(self.frames(7))
            encode_frames(command, output, shape, frames)
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(output)]))
            video = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
            sound = next(stream for stream in probe['streams'] if stream['codec_type'] == 'audio')
            self.assertEqual(video['nb_frames'], '7')
            self.assertEqual(sound['codec_name'], 'aac')

    def test_cancel_and_shape_failure_do_not_replace_previous_result(self):
        def cancelled():
            yield from self.frames(2)
            raise RuntimeError('cancelled')
        def invalid():
            yield from self.frames(1)
            yield np.zeros((64, 32, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'clip.mp4'
            output.write_bytes(b'previous')
            for stream, exception in ((cancelled(), RuntimeError), (invalid(), ValueError)):
                shape, frames = first_frame(stream)
                with self.assertRaises(exception):
                    encode_frames(self.command, output, shape, frames)
                self.assertEqual(output.read_bytes(), b'previous')
                self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_empty_stream_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No video frames'):
            first_frame(iter(()))

    def test_encoder_failure_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'clip.mp4'
            shape, frames = first_frame(self.frames(1))
            with self.assertRaises((subprocess.CalledProcessError, BrokenPipeError)):
                encode_frames(self.command + ['-c:v', 'not_a_real_codec'], output, shape, frames)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
