import sys
import unittest
import weakref
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


class GenerationRetentionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules_forge" / "packages"))
        with mock.patch.object(sys, "argv", ["verify"]):
            from modules import shared_init
            shared_init.initialize()
            from modules import processing, shared
        cls.processing = processing
        cls.shared = shared

    def run_generation(self, retain=False, grouped=False, fail=False, interrupt=False, compatibility=False, check_lifetime=False, replace_samples=False, uneven_groups=False, fail_stage="post_sample", alias_decode=False):
        processing, shared = self.processing, self.shared
        forge = SimpleNamespace(vae=SimpleNamespace(latent_channels=4))
        forge.shallow_copy = lambda: forge
        model = SimpleNamespace(
            is_wan=False, model_config=SimpleNamespace(ztsnr=False),
            sd_checkpoint_info=SimpleNamespace(name_for_extra="test"), sd_model_hash="test",
            forge_objects=forge, forge_objects_original=forge, forge_objects_after_applying_lora=forge,
        )
        p = processing.StableDiffusionProcessingTxt2Img(
            prompt=["a", "b", "c" if uneven_groups else "a", "b"] if grouped else "landscape",
            batch_size=2, n_iter=2 if grouped else 12, seed=10, subseed=20,
            width=16, height=16, do_not_save_samples=True, do_not_save_grid=True,
        )
        runner = mock.MagicMock()
        runner.alwayson_scripts = [SimpleNamespace(requires_all_latents=retain)]
        runner.selectable_scripts = []
        p.scripts_value = runner
        batches, final = [], []
        latent_refs = []
        decoded_pointers = []

        def decode(model, samples, **kwargs):
            # Offset views must be treated as aliases, not just identical tensors.
            decoded = samples[:, 1:4] if alias_decode else torch.zeros(len(samples), 3, 16, 16)
            decoded_pointers.append(decoded.data_ptr())
            return decoded

        def assert_latents_released():
            if check_lifetime:
                self.assertTrue(all(ref() is None for ref in latent_refs), "previous batch latent tensor is still retained")

        def sample(**kwargs):
            assert_latents_released()
            samples = torch.stack([torch.full((4, 2, 2), float(seed)) for seed in p.seeds])
            latent_refs.append(weakref.ref(samples))
            return samples

        def post_sample(p, args):
            batches.append(len(p.latents_after_sampling))
            if replace_samples:
                args.samples = args.samples.clone()
                latent_refs.append(weakref.ref(args.samples))
            if fail and fail_stage == "post_sample":
                raise RuntimeError("post-sample failure")

        # Plain functions avoid mock call histories keeping latent tensors alive.
        runner.post_sample = post_sample
        if fail and fail_stage == "image":
            runner.postprocess_image.side_effect = RuntimeError("post-image failure")

        def postprocess(p, result):
            for latent in p.latents_after_sampling:
                if alias_decode:
                    torch.testing.assert_close(latent, torch.full_like(latent, latent[0, 0, 0].item()))
                final.append(float(latent[0, 0, 0]))

        runner.postprocess.side_effect = postprocess
        if interrupt:
            runner.postprocess_batch.side_effect = lambda *a, **k: setattr(shared.state, "stopping_generation", True)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(processing.sd_models.model_data, "sd_model", model))
            stack.enter_context(mock.patch.dict(shared.opts.data, {
                "forge_retain_all_latents": compatibility, "live_previews_enable": False,
                "sd_noise_schedule": "Default", "save_prompt_comments": False,
                "sd_vae_decode_method": "Full", "return_grid": False, "grid_save": False,
            }))
            stack.enter_context(mock.patch.object(shared.cmd_opts, "no_prompt_history", True))
            for name in ("interrupted", "stopping_generation", "skipped"):
                stack.enter_context(mock.patch.object(shared.state, name, False))
            stack.enter_context(mock.patch.object(shared.state, "nextjob"))
            stack.enter_context(mock.patch.object(processing, "apply_circular_forge"))
            stack.enter_context(mock.patch.object(processing.sd_models, "forge_model_reload"))
            stack.enter_context(mock.patch.object(processing.sd_unet, "apply_unet"))
            stack.enter_context(mock.patch.object(processing.devices, "torch_gc"))
            stack.enter_context(mock.patch.object(processing.rng, "ImageRNG"))
            stack.enter_context(mock.patch.object(processing.extra_networks, "activate"))
            stack.enter_context(mock.patch.object(processing.extra_networks, "deactivate"))
            stack.enter_context(mock.patch.object(processing, "_extra_network_signature", side_effect=lambda prompt: prompt))
            stack.enter_context(mock.patch.object(p, "init"))
            stack.enter_context(mock.patch.object(p, "setup_conds"))
            stack.enter_context(mock.patch.object(p, "parse_extra_network_prompts"))
            stack.enter_context(mock.patch.object(p, "sample", new=sample))
            stack.enter_context(mock.patch.object(processing, "decode_latent_batch", new=decode))
            stack.enter_context(mock.patch.object(processing, "create_infotext", side_effect=lambda p, prompts, seeds, subseeds, index=0, **k: f"{prompts[index]} seed={seeds[index]}"))
            if fail:
                with self.assertRaisesRegex(RuntimeError, "failure"):
                    processing.process_images_inner(p)
                self.assertEqual(p.latents_after_sampling, [])
                return p, None, batches, final
            result = processing.process_images_inner(p)
            assert_latents_released()
            if not alias_decode:
                self.assertEqual(decoded_pointers, [call.args[1].data_ptr() for call in runner.postprocess_batch.call_args_list])
        return p, result, batches, final

    def test_numbered_12_by_2_keeps_current_batch_only(self):
        p, result, batches, final = self.run_generation()
        self.assertEqual(len(result.images), 24)
        self.assertEqual(batches, [2] * 12)
        self.assertEqual(final, [])
        self.assertEqual(p.latents_after_sampling, [])

    def test_previous_batch_tensors_released_before_next_sample(self):
        for replace_samples in (False, True):
            with self.subTest(replace_samples=replace_samples):
                self.run_generation(check_lifetime=True, replace_samples=replace_samples)

    def test_grouped_opt_in_preserves_original_latent_and_gallery_order(self):
        p, result, batches, final = self.run_generation(retain=True, grouped=True)
        self.assertEqual(batches, [2, 4])
        self.assertEqual(final, [10, 11, 12, 13])
        self.assertEqual(result.infotexts, ["a seed=10", "b seed=11", "a seed=12", "b seed=13"])

    def test_callback_error_and_interruption_release_latents(self):
        self.run_generation(fail=True)
        p, result, batches, final = self.run_generation(interrupt=True)
        self.assertEqual(len(result.images), 2)
        self.assertEqual(p.latents_after_sampling, [])

    def test_compatibility_option_preserves_all_numbered_batches(self):
        p, result, batches, final = self.run_generation(compatibility=True)
        self.assertEqual(batches, list(range(2, 25, 2)))
        self.assertEqual(final, list(range(10, 34)))

    def test_grouped_settings_restored_after_callback_failures(self):
        for stage in ("post_sample", "image"):
            with self.subTest(stage=stage):
                p, _, _, _ = self.run_generation(grouped=True, uneven_groups=True, fail=True, fail_stage=stage)
                self.assertEqual((p.batch_size, p.n_iter), (2, 2))
                self.assertIsNone(p._current_batch_indices)
                self.assertIsNone(p._current_output_index)
                self.assertIsNone(p._extra_network_batch_plan)

    def test_grouped_settings_restored_after_success_and_interruption(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt):
                p, _, _, _ = self.run_generation(grouped=True, uneven_groups=True, interrupt=interrupt)
                self.assertEqual((p.batch_size, p.n_iter), (2, 2))
                self.assertIsNone(p._current_batch_indices)
                self.assertIsNone(p._current_output_index)
                self.assertIsNone(p._extra_network_batch_plan)

    def test_aliased_decode_does_not_mutate_retained_samples(self):
        for replace_samples in (False, True):
            with self.subTest(replace_samples=replace_samples):
                _, _, _, final = self.run_generation(retain=True, alias_decode=True, replace_samples=replace_samples)
                self.assertEqual(final, list(range(10, 34)))


if __name__ == "__main__":
    unittest.main()
