import threading
import unittest
import gc
import weakref
from unittest import mock
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from backend.multi_gpu.parallel import BatchSplitProfile, ParallelBatch, ParallelCFG, ParallelPredictions, SplitLinear
from backend.multi_gpu.replicas import ReplicaPair, UnsupportedReplica, clone_unet_patcher


class ParallelPredictionTests(unittest.TestCase):
    def test_automatic_placement_routes_to_the_selected_patcher(self):
        from backend.args import args
        from backend.nn.anima import Anima
        from backend.patcher.unet import UnetPatcher

        diffusion = Anima.__new__(Anima)
        torch.nn.Module.__init__(diffusion)
        diffusion.load_device = torch.device("cuda:0")
        diffusion.initial_device = diffusion.offload_device = torch.device("cpu")
        wrapped = SimpleNamespace(diffusion_model=diffusion)
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch("backend.patcher.unet.KModel", return_value=wrapped), mock.patch("backend.multi_gpu.patcher.ShardedModelPatcher") as sharded:
            for mode in ("replicas", "single"):
                with self.subTest(mode=mode):
                    diffusion._forge_auto_multi_gpu_mode = mode
                    self.assertIsInstance(UnetPatcher.from_model(diffusion, None, None), UnetPatcher)
                    sharded.assert_not_called()
            diffusion._forge_auto_multi_gpu_mode = "sharding"
            self.assertIs(UnetPatcher.from_model(diffusion, None, None), sharded.return_value)
            sharded.assert_called_once()

    def test_adaptive_split_requires_samples_and_memory(self):
        pair = BatchSplitProfile(2)
        self.assertEqual(pair.choose(lambda split: self.fail("a two-image batch has no alternative split")), 1)
        self.assertEqual(pair.selected, 1)
        profile = BatchSplitProfile(3)
        for _ in range(3):
            self.assertEqual(profile.choose(lambda split: True), 2)
            profile.observe(2, 20, 10)
        self.assertEqual(profile.choose(lambda split: False), 2)
        self.assertEqual(profile.selected, 2)
        profile = BatchSplitProfile(3)
        for _ in range(3):
            profile.observe(2, 20, 10)
        for _ in range(3):
            self.assertEqual(profile.choose(lambda split: True), 1)
            profile.observe(1, 10, 15)
        self.assertEqual(profile.choose(lambda split: True), 1)
        profile = BatchSplitProfile(4)
        for _ in range(3):
            profile.observe(2, 10, 20)
        self.assertEqual(profile.choose(lambda split: True), 3)
        for _ in range(3):
            profile.observe(3, 19.5, 10)
        self.assertEqual(profile.choose(lambda split: True), 2)

    def test_replica_copy_checks_host_memory_before_deepcopy(self):
        from backend import memory_management as mm
        from backend.patcher.unet import UnetPatcher

        class Model(torch.nn.Linear):
            computation_dtype = torch.float32

        unet = UnetPatcher(Model(2, 2), torch.device("cuda:0"), torch.device("cpu"))
        with mock.patch.object(mm, "get_free_memory", return_value=0), mock.patch("backend.multi_gpu.replicas.copy.deepcopy") as deepcopy:
            with self.assertRaisesRegex(UnsupportedReplica, "system memory"):
                clone_unet_patcher(unet, torch.device("cuda:1"))
            deepcopy.assert_not_called()

    def test_parallel_batch_rejects_unchecked_or_active_callbacks(self):
        from backend.sampling import sampling_function as sampling_module
        from modules import script_callbacks
        safe = SimpleNamespace(callback=SimpleNamespace(_forge_parallel_batch_safe=lambda: True))
        active = SimpleNamespace(callback=SimpleNamespace(_forge_parallel_batch_safe=lambda: False))
        with mock.patch.dict(script_callbacks.callback_map, {"callbacks_cfg_denoiser": [safe]}):
            self.assertTrue(sampling_module._parallel_batch_callbacks_safe())
        with mock.patch.dict(script_callbacks.callback_map, {"callbacks_cfg_denoiser": [safe, active]}):
            self.assertFalse(sampling_module._parallel_batch_callbacks_safe())
        with mock.patch.dict(script_callbacks.callback_map, {"callbacks_cfg_denoiser": [SimpleNamespace(callback=lambda: None)]}):
            self.assertFalse(sampling_module._parallel_batch_callbacks_safe())

    def test_cfg1_memory_estimate_uses_single_batch_only_when_explicit(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.sampling import sampling_function as sampling_module
        unet = SimpleNamespace(memory_required=mock.Mock(return_value=0), extra_preserved_memory_during_sampling=0,
                               extra_model_patchers_during_sampling=[], controlnet_linked_list=None, has_online_lora=lambda: False,
                               model=SimpleNamespace(predictor=SimpleNamespace(percent_to_sigma=lambda value: value)), list_controlnets=lambda: [])
        x = torch.ones(3, 1, 2, 2)
        with mock.patch.object(args, "parallel_cfg", None), mock.patch.dict(sampling_module.shared.opts.data, {"batch_cond_uncond": True}), mock.patch.object(mm, "load_models_gpu"), mock.patch.object(sampling_module, "_parallel_cfg_cache", None):
            for cfg, allowed, expected in ((1.0, True, 3), (1.0, False, 6), (2.0, True, 6), (None, True, 6)):
                with self.subTest(cfg=cfg, allowed=allowed):
                    sampling_module.sampling_prepare(unet, x, cfg_scale=cfg, cfg1_single_condition=allowed)
                    self.assertEqual(unet.memory_required.call_args.args[0][0], expected)

    def test_parallel_batch_does_not_clone_untested_model_types(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.sampling import sampling_function as sampling_module
        unet = SimpleNamespace(memory_required=mock.Mock(return_value=0), extra_preserved_memory_during_sampling=0,
                               extra_model_patchers_during_sampling=[], controlnet_linked_list=None, has_online_lora=lambda: False,
                               model=SimpleNamespace(predictor=SimpleNamespace(percent_to_sigma=lambda value: value)), list_controlnets=lambda: [])
        with mock.patch.object(args, "parallel_cfg", None), mock.patch.object(args, "parallel_batch", 1), mock.patch.object(mm, "load_models_gpu"), mock.patch.object(sampling_module, "_parallel_cfg_cache", None), mock.patch("backend.multi_gpu.replicas.clone_unet_patcher") as clone:
            sampling_module.sampling_prepare(unet, torch.ones(3, 1, 2, 2), cfg_scale=1.0, cfg1_single_condition=True)
            self.assertFalse(unet._experimental_cfg_batch_mode)
            clone.assert_not_called()

    def test_replica_memory_probe_failure_falls_back_to_primary(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.sampling import sampling_function as sampling_module
        unet = SimpleNamespace(load_device=torch.device("cuda:0"), model_size=lambda: 1,
                               memory_required=mock.Mock(return_value=0), extra_preserved_memory_during_sampling=0,
                               extra_model_patchers_during_sampling=[], controlnet_linked_list=None, extra_concat_condition=None,
                               object_patches={}, has_online_lora=lambda: False,
                               model=SimpleNamespace(predictor=SimpleNamespace(percent_to_sigma=lambda value: value)), list_controlnets=lambda: [])
        with mock.patch.object(args, "parallel_cfg", None), mock.patch.object(args, "parallel_batch", 1), mock.patch.object(torch.cuda, "device_count", return_value=2), mock.patch.object(sampling_module, "_parallel_batch_model_supported", return_value=True), mock.patch.object(sampling_module, "_parallel_batch_callbacks_safe", return_value=True), mock.patch.object(sampling_module, "_parallel_cfg_cache", None), mock.patch.object(mm, "get_free_memory", side_effect=RuntimeError("probe failed")), mock.patch.object(mm, "load_models_gpu") as load, mock.patch("backend.multi_gpu.replicas.clone_unet_patcher") as clone:
            sampling_module.sampling_prepare(unet, torch.ones(3, 1, 2, 2), cfg_scale=1.0, cfg1_single_condition=True)
            clone.assert_not_called()
            load.assert_called_once()

    def test_large_model_replica_uses_actual_secondary_memory(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.sampling import sampling_function as sampling_module
        gib = 1024 ** 3
        for free, should_clone in ((8 * gib, False), (12 * gib, True), (24 * gib, True)):
            with self.subTest(free=free):
                unet = SimpleNamespace(load_device=torch.device("cuda:0"), model_size=lambda: 7 * gib,
                                       memory_required=mock.Mock(return_value=0), extra_preserved_memory_during_sampling=0,
                                       extra_model_patchers_during_sampling=[], controlnet_linked_list=None, extra_concat_condition=None,
                                       object_patches={}, has_online_lora=lambda: False,
                                       model=SimpleNamespace(predictor=SimpleNamespace(percent_to_sigma=lambda value: value)), list_controlnets=lambda: [])
                replica = object()
                with mock.patch.object(args, "parallel_cfg", None), mock.patch.object(args, "parallel_batch", 1), mock.patch.object(torch.cuda, "device_count", return_value=2), mock.patch.object(sampling_module, "_parallel_batch_model_supported", return_value=True), mock.patch.object(sampling_module, "_parallel_batch_callbacks_safe", return_value=True), mock.patch.object(sampling_module, "_parallel_cfg_cache", None), mock.patch.object(mm, "get_free_memory", return_value=free), mock.patch.object(mm, "minimum_inference_memory", return_value=2 * gib), mock.patch.object(mm, "load_models_gpu"), mock.patch("backend.multi_gpu.replicas.clone_unet_patcher", return_value=replica) as clone:
                    sampling_module.sampling_prepare(unet, torch.ones(3, 1, 2, 2), cfg_scale=1.0, cfg1_single_condition=True)
                    self.assertEqual(clone.called, should_clone)
                    self.assertIs(getattr(unet, "_experimental_cfg_replica", None), replica if should_clone else None)

    def test_failed_source_offload_keeps_tracking_and_falls_back(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.sampling import sampling_function as sampling_module

        model = SimpleNamespace(predictor=SimpleNamespace(percent_to_sigma=lambda value: value))
        unet = SimpleNamespace(load_device=torch.device("cuda:0"), model_size=lambda: 1,
                               memory_required=lambda shape: 0, extra_preserved_memory_during_sampling=0,
                               extra_model_patchers_during_sampling=[], controlnet_linked_list=None,
                               extra_concat_condition=None, object_patches={}, has_online_lora=lambda: False,
                               model=model, list_controlnets=lambda: [])
        loaded = SimpleNamespace(model=SimpleNamespace(model=model), model_unload=mock.Mock(return_value=False))
        unrelated = SimpleNamespace(model=SimpleNamespace(model=object()), model_unload=mock.Mock())
        with mock.patch.object(args, "parallel_cfg", 1), mock.patch.object(mm, "current_loaded_models", [loaded, unrelated]), mock.patch.object(sampling_module, "_parallel_cfg_cache", None), mock.patch.object(torch.cuda, "device_count", return_value=2), mock.patch.object(mm, "get_free_memory", return_value=10**12), mock.patch.object(mm, "load_models_gpu") as load, mock.patch("backend.multi_gpu.replicas.clone_unet_patcher") as clone:
            sampling_module.sampling_prepare(unet, torch.ones(1, 1, 2, 2), cfg_scale=2.0)
            self.assertEqual(mm.current_loaded_models, [loaded, unrelated])
            loaded.model_unload.assert_called_once()
            unrelated.model_unload.assert_not_called()
            clone.assert_not_called()
            load.assert_called_once()

    def test_automatic_modes_do_not_clone_when_replicas_are_not_selected(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.sampling import sampling_function as sampling_module

        for mode in ("sharding", "single"):
            with self.subTest(mode=mode):
                model = SimpleNamespace(diffusion_model=SimpleNamespace(_forge_auto_multi_gpu_mode=mode), predictor=SimpleNamespace(percent_to_sigma=lambda value: value))
                unet = SimpleNamespace(load_device=torch.device("cuda:0"), model_size=lambda: 1,
                                       memory_required=lambda shape: 0, extra_preserved_memory_during_sampling=0,
                                       extra_model_patchers_during_sampling=[], controlnet_linked_list=None,
                                       extra_concat_condition=None, object_patches={}, has_online_lora=lambda: False,
                                       model=model, list_controlnets=lambda: [])
                with mock.patch.object(args, "parallel_cfg", 1), mock.patch.object(args, "parallel_batch", 1), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(sampling_module, "_parallel_cfg_cache", None), mock.patch.object(sampling_module, "_parallel_batch_model_supported", return_value=True), mock.patch.object(sampling_module, "_parallel_batch_callbacks_safe", return_value=True), mock.patch.object(mm, "load_models_gpu") as load, mock.patch("backend.multi_gpu.replicas.clone_unet_patcher") as clone:
                    sampling_module.sampling_prepare(unet, torch.ones(2, 1, 2, 2), cfg_scale=1.0, cfg1_single_condition=True)
                    clone.assert_not_called()
                    load.assert_called_once()

    def test_automatic_replica_copy_failure_runs_primary_model(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.sampling import sampling_function as sampling_module

        model = SimpleNamespace(diffusion_model=SimpleNamespace(_forge_auto_multi_gpu_mode="replicas"), predictor=SimpleNamespace(percent_to_sigma=lambda value: value))
        unet = SimpleNamespace(load_device=torch.device("cuda:0"), model_size=lambda: 1,
                               memory_required=lambda shape: 0, extra_preserved_memory_during_sampling=0,
                               extra_model_patchers_during_sampling=[], controlnet_linked_list=None,
                               extra_concat_condition=None, object_patches={}, has_online_lora=lambda: False,
                               model=model, list_controlnets=lambda: [])
        with mock.patch.object(args, "parallel_cfg", 1), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(sampling_module, "_parallel_cfg_cache", None), mock.patch.object(torch.cuda, "device_count", return_value=2), mock.patch.object(mm, "get_free_memory", return_value=10**12), mock.patch.object(mm, "load_models_gpu") as load, mock.patch("backend.multi_gpu.replicas.clone_unet_patcher", side_effect=UnsupportedReplica("injected failure")) as clone:
            sampling_module.sampling_prepare(unet, torch.ones(1, 1, 2, 2), cfg_scale=2.0)
            clone.assert_called_once()
            self.assertIsNone(getattr(unet, "_experimental_cfg_replica", None))
            self.assertIs(load.call_args.kwargs["models"][0], unet)

    def test_auto_sharded_fallback_copies_and_refreshes_patcher_state(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.patcher.unet import UnetPatcher
        from backend.sampling import sampling_function as sampling_module

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion_model = torch.nn.Linear(2, 2)
                self.diffusion_model._forge_auto_multi_gpu_mode = "replicas"
                self.computation_dtype = torch.float32

        model = ToyModel()
        unet = UnetPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
        unet.add_patches({"diffusion_model.weight": (torch.ones_like(model.diffusion_model.weight),)}, strength_patch=0.25)
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(torch.cuda, "device_count", return_value=2), mock.patch.object(mm, "current_loaded_models", []):
            fallback = sampling_module._auto_sharded_fallback(unet)
            self.assertIs(fallback.model, model)
            self.assertEqual(fallback.patches_uuid, unet.patches_uuid)
            self.assertEqual(fallback.patches.keys(), unet.patches.keys())
            self.assertIs(sampling_module._auto_sharded_fallback(unet), fallback)
            unet.add_patches({"diffusion_model.weight": (torch.ones_like(model.diffusion_model.weight),)}, strength_patch=0.1)
            updated = sampling_module._auto_sharded_fallback(unet)
            self.assertIsNot(updated, fallback)
            self.assertEqual(updated.patches_uuid, unet.patches_uuid)
            self.assertEqual(len(updated.patches["diffusion_model.weight"]), 2)
            unet.add_patches({"diffusion_model.bias": (torch.ones_like(model.diffusion_model.bias),)}, strength_patch=0.1, online_mode=True)
            online = sampling_module._auto_sharded_fallback(unet)
            self.assertIsNot(online, updated)
            self.assertEqual(online.patches_uuid, unet.patches_uuid)
            self.assertEqual(online.weight_wrapper_patches.keys(), unet.weight_wrapper_patches.keys())
            original_revision = unet.patches_uuid
            unet.weight_wrapper_patches["diffusion_model.bias"].append(object())
            self.assertEqual(unet.patches_uuid, original_revision)
            self.assertIsNot(sampling_module._auto_sharded_fallback(unet), online)

    def test_failed_sharded_fallback_restores_primary(self):
        from backend import memory_management as mm
        from backend.sampling import sampling_function as sampling_module

        model = SimpleNamespace(_anima_shard=None)
        unet = SimpleNamespace(model=model)
        fallback = SimpleNamespace(detach=mock.Mock())
        model._forge_auto_sharded_fallback = fallback
        with mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(mm.logger, "exception"), mock.patch.object(mm, "load_models_gpu", side_effect=[torch.OutOfMemoryError("injected"), None]) as load:
            self.assertFalse(sampling_module._load_auto_sharded_fallback(unet, fallback, 123))
            fallback.detach.assert_called_once()
            self.assertIsNone(model._forge_auto_sharded_fallback)
            self.assertEqual(load.call_args_list[0].args[0], [fallback])
            self.assertEqual(load.call_args_list[1].args[0], [unet])

    def test_late_replica_memory_failure_attempts_sharding(self):
        from backend import memory_management as mm
        from backend.multi_gpu.parallel import ParallelCFG
        from backend.sampling import sampling_function as sampling_module

        replica = SimpleNamespace(patches_uuid="revision", load_device=torch.device("cuda:1"), model_size=lambda: 1)
        unet = SimpleNamespace(_experimental_cfg_replica=replica, patches_uuid="revision", model=SimpleNamespace(model_lowvram=False), loaded_size=lambda: 1, model_size=lambda: 1, memory_required=lambda shape: 1, extra_preserved_memory_during_sampling=0, object_patches={}, controlnet_linked_list=None, extra_model_patchers_during_sampling=[], extra_concat_condition=None)
        fallback = object()
        with mock.patch.object(ParallelCFG, "eligible", return_value=True), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(mm, "get_free_memory", return_value=0), mock.patch.object(sampling_module, "_auto_sharded_fallback", return_value=fallback), mock.patch.object(sampling_module, "_load_auto_sharded_fallback") as load:
            self.assertIsNone(sampling_module._load_parallel_cfg(unet, [], [], {}, torch.ones(1, 1, 2, 2)))
            self.assertIsNone(unet._experimental_cfg_replica)
            self.assertIs(load.call_args.args[1], fallback)

    def test_cfg1_memory_plan_rejects_complex_or_mutable_conditioning(self):
        from backend.args import dynamic_args
        from modules import prompt_parser, script_callbacks, sd_samplers_common
        unet = SimpleNamespace(controlnet_linked_list=None, extra_concat_condition=None, object_patches={}, extra_model_patchers_during_sampling=[], model_options={"transformer_options": {}})
        sampler = SimpleNamespace(func=SimpleNamespace(__module__="k_diffusion.sampling"), model_wrap=SimpleNamespace(inner_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet))))
        p = SimpleNamespace(cfg_scale=1.0, is_hr_pass=False, refiner_checkpoint=None, refiner_cfg=None, image_cfg_scale=None)
        cond = prompt_parser.MulticondLearnedConditioning((2,), [[SimpleNamespace(weight=1.0)], [SimpleNamespace(weight=1.0)]])
        x = torch.ones(2, 1, 2, 2)
        check = sd_samplers_common.cfg1_single_condition_eligible
        with mock.patch.object(dynamic_args, "context_handler", None), mock.patch.dict(script_callbacks.callback_map, {"callbacks_cfg_denoiser": []}):
            self.assertTrue(check(sampler, p, cond, None, x))
            self.assertFalse(check(sampler, p, cond, object(), x))
            with mock.patch.object(p, "cfg_scale", 2.0):
                self.assertFalse(check(sampler, p, cond, None, x))
            with mock.patch.object(p, "is_hr_pass", True), mock.patch.object(p, "hr_cfg", 2.0, create=True):
                self.assertFalse(check(sampler, p, cond, None, x))
            with mock.patch.object(p, "refiner_cfg", 2.0):
                self.assertFalse(check(sampler, p, cond, None, x))
            with mock.patch.object(dynamic_args, "ref_latents", [torch.ones(1)]):
                self.assertFalse(check(sampler, p, cond, None, x))
            with mock.patch.dict(unet.model_options, {"conditioning_modifiers": [object()]}):
                self.assertFalse(check(sampler, p, cond, None, x))
            with mock.patch.dict(script_callbacks.callback_map, {"callbacks_cfg_denoiser": [object()]}):
                self.assertFalse(check(sampler, p, cond, None, x))
            safe_callback = SimpleNamespace(callback=SimpleNamespace(_forge_preserves_cfg1_memory_plan=True))
            with mock.patch.dict(script_callbacks.callback_map, {"callbacks_cfg_denoiser": [safe_callback]}):
                self.assertTrue(check(sampler, p, cond, None, x))
            with mock.patch.dict(script_callbacks.callback_map, {"callbacks_cfg_denoiser": [safe_callback, object()]}):
                self.assertFalse(check(sampler, p, cond, None, x))
            cond.batch[0].append(SimpleNamespace(weight=1.0))
            self.assertFalse(check(sampler, p, cond, None, x))

    def test_sampler_setup_failures_release_prepared_replica(self):
        from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers_timesteps
        from backend.sampling import sampling_function as sampling_module

        unet = SimpleNamespace()
        inner = SimpleNamespace(forge_objects=SimpleNamespace(unet=unet), alphas_cumprod=torch.ones(1))
        p = SimpleNamespace(cfg_scale=2, steps=1)
        x = torch.ones(1, 1, 2, 2)
        for module, sampler_class, schedule in ((sd_samplers_kdiffusion, sd_samplers_kdiffusion.KDiffusionSampler, "get_sigmas"),
                                                 (sd_samplers_timesteps, sd_samplers_timesteps.CompVisSampler, "get_timesteps")):
            for img2img in (False, True):
                with self.subTest(sampler=sampler_class.__name__, img2img=img2img):
                    sampler = sampler_class.__new__(sampler_class)
                    sampler.model_wrap = SimpleNamespace(inner_model=inner)
                    with mock.patch.object(module, "sampling_prepare"), mock.patch.object(sampler, schedule, side_effect=torch.OutOfMemoryError("schedule failed")), mock.patch.object(sd_samplers_common, "setup_img2img_steps", return_value=(1, 1)), mock.patch.object(sampling_module, "_release_parallel_cfg") as release:
                        with self.assertRaisesRegex(torch.OutOfMemoryError, "schedule failed"):
                            if img2img:
                                sampler.sample_img2img(p, x, x, None, None, steps=1)
                            else:
                                sampler.sample(p, x, None, None, steps=1)
                        release.assert_called_once_with(unet)

    def test_memory_estimation_failure_releases_reused_resident_replica(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.sampling import sampling_function as sampling_module

        primary = torch.nn.Linear(1, 1)
        secondary = torch.device("cuda:1")
        replica = SimpleNamespace(patches_uuid="revision", load_device=secondary, current_device=secondary, loaded_size=lambda: 1)
        loaded = SimpleNamespace(model=replica, model_unload=mock.Mock())
        unet = SimpleNamespace(model=primary, patches_uuid="revision", load_device=torch.device("cuda:0"), model_size=lambda: 1,
                               memory_required=mock.Mock(side_effect=torch.OutOfMemoryError("estimate failed")), object_patches={},
                               controlnet_linked_list=None, extra_model_patchers_during_sampling=[], extra_concat_condition=None)
        cached = (weakref.ref(primary), replica)
        with mock.patch.object(args, "parallel_cfg", 1), mock.patch.object(mm, "current_loaded_models", [loaded]), mock.patch.object(mm, "get_free_memory", return_value=10**12), mock.patch.object(torch.cuda, "device_count", return_value=2), mock.patch.object(sampling_module, "_parallel_cfg_cache", cached):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "estimate failed"):
                sampling_module.sampling_prepare(unet, torch.ones(1, 1, 2, 2), cfg_scale=2)
            self.assertIsNone(unet._experimental_cfg_replica)
            self.assertIsNone(sampling_module._parallel_cfg_cache)
            self.assertEqual(mm.current_loaded_models, [])
            loaded.model_unload.assert_called_once()

    def test_resident_replica_cache_releases_and_retries_failed_unload(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.sampling import sampling_function as sampling_module

        primary = torch.nn.Linear(1, 1)
        replica = SimpleNamespace(patches_uuid="revision")
        adapter = SimpleNamespace(close=mock.Mock())
        unet = SimpleNamespace(model=primary, patches_uuid="revision", _experimental_parallel_cfg=adapter, _experimental_cfg_replica=replica)
        loaded = SimpleNamespace(model=replica, model_unload=mock.Mock(side_effect=[RuntimeError("injected"), None]))
        with mock.patch.object(args, "parallel_cfg_keep_resident", True), mock.patch.object(mm, "current_loaded_models", [loaded]), mock.patch.object(sampling_module, "_parallel_cfg_cache", None), mock.patch.object(mm.logger, "exception"):
            sampling_module._release_parallel_cfg(unet, preserve_cpu=True)
            self.assertIsNone(unet._experimental_cfg_replica)
            self.assertIs(sampling_module._parallel_cfg_cache[1], replica)
            self.assertEqual(mm.current_loaded_models, [loaded])
            self.assertFalse(sampling_module._discard_parallel_cfg_cache())
            self.assertIs(sampling_module._parallel_cfg_cache[1], replica)
            self.assertTrue(sampling_module._discard_parallel_cfg_cache())
            self.assertIsNone(sampling_module._parallel_cfg_cache)
            self.assertEqual(mm.current_loaded_models, [])
            adapter.close.assert_called_once()

    def test_failed_replica_load_rolls_back_and_retries_failed_cleanup(self):
        from backend import memory_management as mm
        from backend.sampling.sampling_function import _load_parallel_cfg, _release_parallel_cfg
        replica = SimpleNamespace(patches_uuid="revision", load_device=torch.device("cuda:1"), model_size=lambda: 1, detach=mock.Mock(side_effect=[RuntimeError("cleanup failure"), None]))
        unet = SimpleNamespace(_experimental_cfg_replica=replica, patches_uuid="revision", model=SimpleNamespace(model_lowvram=False), loaded_size=lambda: 1, model_size=lambda: 1, memory_required=lambda shape: 0, object_patches={}, controlnet_linked_list=None, extra_model_patchers_during_sampling=[], extra_concat_condition=None)
        x = torch.ones(1, 1, 2, 2)
        with mock.patch.object(ParallelCFG, "eligible", return_value=True), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(mm, "get_free_memory", return_value=10**12), mock.patch.object(mm, "load_models_gpu", side_effect=torch.OutOfMemoryError("injected")), mock.patch.object(mm.logger, "exception"):
            self.assertIsNone(_load_parallel_cfg(unet, [{}], [{}], {}, x))
            self.assertIs(unet._experimental_cfg_replica, replica)
            self.assertTrue(unet._experimental_cfg_needs_rollback)
            _release_parallel_cfg(unet)
            self.assertIsNone(unet._experimental_cfg_replica)
            self.assertEqual(replica.detach.call_count, 2)

    def test_cpu_cache_does_not_keep_source_model_alive(self):
        from backend.sampling import sampling_function as sampling_module
        source = torch.nn.Linear(1, 1)
        reference = weakref.ref(source, sampling_module._clear_parallel_cfg_cache)
        with mock.patch.object(sampling_module, "_parallel_cfg_cache", (reference, object())):
            del source
            gc.collect()
            self.assertIsNone(reference())
            self.assertIsNone(sampling_module._parallel_cfg_cache)

    def test_in_place_model_reload_invalidates_only_matching_cache(self):
        from backend.sampling import sampling_function as sampling_module
        source = torch.nn.Linear(1, 1)
        other = torch.nn.Linear(1, 1)
        cache = (weakref.ref(source), object())
        with mock.patch.object(sampling_module, "_parallel_cfg_cache", cache):
            sampling_module._invalidate_parallel_cfg_cache(other)
            self.assertIs(sampling_module._parallel_cfg_cache, cache)
            sampling_module._invalidate_parallel_cfg_cache(source)
            self.assertIsNone(sampling_module._parallel_cfg_cache)

    def test_interrupted_sampler_releases_replica(self):
        from modules import sd_samplers_common
        unet = object()
        sampler = SimpleNamespace(
            model_wrap_cfg=SimpleNamespace(clear_conditioning_cache=mock.Mock()),
            config=SimpleNamespace(total_steps=lambda steps: steps),
            model_wrap=SimpleNamespace(inner_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet))),
        )
        def interrupt():
            raise sd_samplers_common.InterruptedException()
        with mock.patch("backend.sampling.sampling_function._release_parallel_cfg") as release:
            sd_samplers_common.Sampler.launch_sampling(sampler, 1, interrupt)
        release.assert_called_once_with(unet, preserve_cpu=True)

    def test_branches_overlap_and_preserve_order(self):
        barrier = threading.Barrier(2, timeout=5)

        def predict(value):
            self.assertTrue(torch.is_inference_mode_enabled())
            barrier.wait()
            return value * 2

        with ParallelPredictions(predict, predict) as executor:
            self.assertEqual(executor.predict((2,), (3,)), (4, 6))

    def test_failed_branch_waits_for_other_branch(self):
        started = threading.Event()
        finished = threading.Event()

        def fail():
            started.wait(timeout=5)
            raise RuntimeError("prediction failed")

        def complete():
            started.set()
            finished.set()

        with ParallelPredictions(fail, complete) as executor:
            with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                executor.predict((), ())
        self.assertTrue(finished.is_set())


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "two CUDA devices required")
class SplitLinearTests(unittest.TestCase):
    def test_parallel_cfg1_batch_preserves_even_and_odd_image_order(self):
        from backend.args import dynamic_args
        from backend.sampling.condition import compile_conditions
        from backend.sampling.sampling_function import sampling_function_inner

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.25))
            def memory_required(self, shape):
                return 0
            def apply_model(self, x, timestep, c_crossattn, **kwargs):
                return x * self.scale + c_crossattn[:, :1, :1, None]

        with mock.patch.object(dynamic_args, "context_handler", None), ReplicaPair(Model(), ("cuda:0", "cuda:1")) as pair:
            primary, secondary = pair.models
            adapter = ParallelBatch(primary, secondary, pair.devices)
            try:
                for count in (2, 3):
                    with self.subTest(batch=count):
                        cond = compile_conditions(torch.arange(count, dtype=torch.float32).reshape(count, 1, 1))
                        cond[0]["batch_indices"] = list(range(count))
                        x = torch.arange(count, dtype=torch.float32, device="cuda:0").reshape(count, 1, 1, 1).expand(count, 1, 2, 2)
                        timestep = torch.ones(count, device="cuda:0")
                        self.assertTrue(adapter.eligible(cond, None, {}, x))
                        with torch.inference_mode():
                            actual = sampling_function_inner(primary, x, timestep, None, cond, 1.0, return_full=True, parallel_cfg=adapter)
                            expected = sampling_function_inner(primary, x, timestep, None, cond, 1.0, return_full=True)
                        for observed, reference in zip(actual, expected):
                            torch.testing.assert_close(observed, reference)
                        # A shared condition is repeated by the serial sampler;
                        # it must not be sliced into an empty secondary branch.
                        shared = compile_conditions(torch.full((1, 1, 1), 3.0))
                        self.assertTrue(adapter.eligible(shared, None, {}, x))
                        with torch.inference_mode(), mock.patch.object(adapter, "predict", wraps=adapter.predict) as full_predict:
                            actual_direct = sampling_function_inner(primary, x, timestep, None, shared, 1.0, parallel_cfg=adapter)
                            expected_direct = sampling_function_inner(primary, x, timestep, None, shared, 1.0)
                            full_predict.assert_not_called()
                            actual_full = sampling_function_inner(primary, x, timestep, None, shared, 1.0, return_full=True, parallel_cfg=adapter)
                            expected_full = sampling_function_inner(primary, x, timestep, None, shared, 1.0, return_full=True)
                            full_predict.assert_called_once()
                        torch.testing.assert_close(actual_direct, expected_direct)
                        for observed, reference in zip(actual_full, expected_full):
                            torch.testing.assert_close(observed, reference)
                        self.assertFalse(adapter.eligible(cond, cond, {}, x))
                        self.assertFalse(adapter.eligible(cond, None, {"sampler_pre_cfg_function": [object()]}, x))
            finally:
                adapter.close()

    def test_adaptive_batch_retries_balanced_split_after_oom(self):
        from backend.sampling.condition import compile_conditions
        from backend.sampling.sampling_function import sampling_function_inner

        class Model(torch.nn.Module):
            def memory_required(self, shape):
                return 0
            def apply_model(self, x, timestep, c_crossattn, **kwargs):
                return x + c_crossattn[:, :1, :1, None]

        with ReplicaPair(Model(), ("cuda:0", "cuda:1")) as pair:
            adapter = ParallelBatch(*pair.models, pair.devices, adaptive=True)
            cond = compile_conditions(torch.arange(3, dtype=torch.float32).reshape(3, 1, 1))
            x = torch.zeros(3, 1, 2, 2, device="cuda:0")
            timestep = torch.ones(3, device="cuda:0")
            executor = adapter._executor.predict
            calls = []
            def fail_first(left, right):
                calls.append(left[1].shape[0])
                if len(calls) == 1:
                    raise torch.OutOfMemoryError("injected")
                return executor(left, right)
            try:
                with mock.patch.object(BatchSplitProfile, "choose", return_value=1), mock.patch.object(adapter._executor, "predict", side_effect=fail_first), torch.inference_mode():
                    actual = sampling_function_inner(pair.models[0], x, timestep, None, cond, 1.0, parallel_cfg=adapter)
                    expected = sampling_function_inner(pair.models[0], x, timestep, None, cond, 1.0)
                self.assertEqual(calls, [1, 2])
                self.assertEqual(adapter._profile.selected, 2)
                torch.testing.assert_close(actual, expected)
            finally:
                adapter.close()

    def test_parallel_batch_reuses_and_invalidates_conditioning(self):
        from backend.sampling.condition import compile_conditions

        class Model(torch.nn.Module):
            def memory_required(self, shape):
                return 0
            def apply_model(self, x, timestep, c_crossattn, **kwargs):
                return x + c_crossattn[:, :1, :1, None]

        with ReplicaPair(Model(), ("cuda:0", "cuda:1")) as pair:
            adapter = ParallelBatch(*pair.models, pair.devices)
            source = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1)
            cond = compile_conditions(source)
            x = torch.zeros(3, 1, 2, 2, device="cuda:0")
            timestep = torch.ones(3, device="cuda:0")
            try:
                with torch.inference_mode():
                    first, _ = adapter.predict(cond, None, x, timestep, {})
                    cached = next(iter(adapter._secondary_cache.values()))
                    repeated, _ = adapter.predict(cond, None, x, timestep, {})
                self.assertIs(cached, next(iter(adapter._secondary_cache.values())))
                torch.testing.assert_close(first, repeated)
                source.add_(10)
                with torch.inference_mode():
                    updated, _ = adapter.predict(cond, None, x, timestep, {})
                self.assertIsNot(cached, next(iter(adapter._secondary_cache.values())))
                torch.testing.assert_close(updated, first + 10)
            finally:
                adapter.close()
            self.assertFalse(adapter._secondary_cache)

    def test_adaptive_batch_split_preserves_odd_batch_results(self):
        from backend.sampling.condition import compile_conditions
        from backend.sampling.sampling_function import sampling_function_inner

        class Model(torch.nn.Module):
            def memory_required(self, shape):
                return 0
            def apply_model(self, x, timestep, c_crossattn, **kwargs):
                return x + c_crossattn[:, :1, :1, None]

        with ReplicaPair(Model(), ("cuda:0", "cuda:1")) as pair:
            adapter = ParallelBatch(*pair.models, pair.devices, adaptive=True)
            cond = compile_conditions(torch.arange(3, dtype=torch.float32).reshape(3, 1, 1))
            x = torch.zeros(3, 1, 2, 2, device="cuda:0")
            timestep = torch.ones(3, device="cuda:0")
            try:
                with mock.patch.object(BatchSplitProfile, "choose", return_value=1), torch.inference_mode():
                    actual = sampling_function_inner(pair.models[0], x, timestep, None, cond, 1.0, parallel_cfg=adapter)
                    expected = sampling_function_inner(pair.models[0], x, timestep, None, cond, 1.0)
                torch.testing.assert_close(actual, expected)
            finally:
                adapter.close()

    def test_resident_replica_can_be_evicted_by_memory_manager(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.patcher.unet import UnetPatcher
        from backend.sampling import sampling_function as sampling_module

        class Model(torch.nn.Linear):
            computation_dtype = torch.float32
            def get_dtype(self):
                return torch.float32

        primary = Model(2, 2)
        replica = UnetPatcher(Model(2, 2), torch.device("cuda:1"), torch.device("cpu"))
        unet = SimpleNamespace(model=primary, patches_uuid=replica.patches_uuid, _experimental_parallel_cfg=SimpleNamespace(close=mock.Mock()), _experimental_cfg_replica=replica)
        with mock.patch.object(args, "parallel_cfg_keep_resident", True), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(sampling_module, "_parallel_cfg_cache", None):
            try:
                mm.load_models_gpu([replica], force_full_load=True)
                sampling_module._release_parallel_cfg(unet, preserve_cpu=True)
                self.assertGreater(replica.loaded_size(), 0)
                mm.free_memory(float("inf"), torch.device("cuda:1"))
                self.assertEqual(replica.loaded_size(), 0)
                self.assertTrue(sampling_module._discard_parallel_cfg_cache())
            finally:
                sampling_module._discard_parallel_cfg_cache()

    def test_replica_clones_quantized_tensor_storage_independently(self):
        from backend.patcher.unet import UnetPatcher
        from backend.quant_ops import QuantizedTensor

        class Model(torch.nn.Module):
            computation_dtype = torch.float32

            def __init__(self):
                super().__init__()
                weight = QuantizedTensor.from_float(torch.randn(16, 16), "TensorWiseINT8Layout")
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

        original = UnetPatcher(Model(), torch.device("cuda:0"), torch.device("cpu"))
        replica = clone_unet_patcher(original, torch.device("cuda:1"))
        self.assertEqual(replica.model.weight._layout_cls, original.model.weight._layout_cls)
        self.assertNotEqual(replica.model.weight._qdata.data_ptr(), original.model.weight._qdata.data_ptr())
        self.assertNotEqual(replica.model.weight._params.scale.data_ptr(), original.model.weight._params.scale.data_ptr())

    def test_replica_clones_fp8_weights_without_changing_storage(self):
        from backend.patcher.unet import UnetPatcher
        from backend.quant_ops import QuantizedTensor

        for layout in ("TensorCoreFP8E4M3Layout", "TensorCoreFP8E5M2Layout"):
            with self.subTest(layout=layout):
                class Model(torch.nn.Module):
                    computation_dtype = torch.float32

                    def __init__(self):
                        super().__init__()
                        self.weight = torch.nn.Parameter(QuantizedTensor.from_float(torch.randn(16, 16), layout), requires_grad=False)

                original = UnetPatcher(Model(), torch.device("cuda:0"), torch.device("cpu"))
                replica = clone_unet_patcher(original, torch.device("cuda:1"))
                self.assertEqual(replica.model.weight._qdata.dtype, original.model.weight._qdata.dtype)
                self.assertNotEqual(replica.model.weight._qdata.data_ptr(), original.model.weight._qdata.data_ptr())
                torch.testing.assert_close(replica.model.weight.dequantize(), original.model.weight.dequantize())

    def test_sampling_lifecycle_loads_and_unloads_independent_lora_replica(self):
        from backend import memory_management as mm
        from backend.args import args, dynamic_args
        from backend.operations import using_forge_operations
        from backend.patcher.unet import UnetPatcher
        from backend.sampling.condition import compile_conditions
        from backend.sampling.sampling_function import _load_parallel_cfg, sampling_cleanup, sampling_function_inner, sampling_prepare
        from backend.sampling import sampling_function as sampling_module

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
                    self.diffusion_model = torch.nn.Linear(2, 2)
                torch.nn.init.uniform_(self.diffusion_model.weight, -0.1, 0.1)
                torch.nn.init.zeros_(self.diffusion_model.bias)
                self.computation_dtype = torch.float32
                self.predictor = SimpleNamespace(percent_to_sigma=lambda value: value)

            def get_dtype(self):
                return torch.float32

            def memory_required(self, input_shape):
                return 0

            def apply_model(self, x, timestep, c_crossattn, **kwargs):
                return self.diffusion_model(x) + c_crossattn[:, :1, :1, None]

        x = torch.ones(1, 1, 2, 2, device="cuda:0")
        t = torch.ones(1, device="cuda:0")
        cond = compile_conditions(torch.ones(1, 1, 1))
        uncond = compile_conditions(torch.zeros(1, 1, 1))
        for online in (False, True):
            with self.subTest(online=online), mock.patch.object(args, "parallel_cfg", 1), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(dynamic_args, "ref_latents", []), mock.patch.object(dynamic_args, "context_handler", None):
                unet = UnetPatcher(Model(), torch.device("cuda:0"), torch.device("cpu"))
                unet.add_patches({"diffusion_model.weight": (torch.ones(2, 2),)}, strength_patch=0.02, online_mode=online)
                # A CFG 1 request leaves the primary model resident without a
                # replica; the next request must still enable parallel CFG.
                sampling_prepare(unet, x, cfg_scale=1.0)
                self.assertIsNone(getattr(unet, "_experimental_cfg_replica", None))
                sampling_cleanup(unet)
                sampling_prepare(unet, x)
                try:
                    replica = unet._experimental_cfg_replica
                    self.assertIsNotNone(replica)
                    adapter = _load_parallel_cfg(unet, cond, uncond, {}, x)
                    self.assertIsNotNone(adapter)
                    with torch.inference_mode():
                        actual = sampling_function_inner(unet.model, x, t, uncond, cond, 2.0, return_full=True, parallel_cfg=adapter)
                        expected = sampling_function_inner(unet.model, x, t, uncond, cond, 2.0, return_full=True)
                    for observed, reference in zip(actual, expected):
                        torch.testing.assert_close(observed, reference)
                    self.assertIsNot(unet.model.diffusion_model.weight, replica.model.diffusion_model.weight)
                finally:
                    sampling_cleanup(unet)
                self.assertIsNone(unet._experimental_cfg_replica)
                self.assertIsNone(unet._experimental_parallel_cfg)
                self.assertIs(sampling_module._parallel_cfg_cache[1], replica)
                with mock.patch.object(args, "parallel_cfg_keep_resident", True), mock.patch.object(args, "parallel_cfg", None), mock.patch.object(args, "parallel_batch", 1):
                    sampling_prepare(unet, x, cfg_scale=1.0)
                    self.assertIs(sampling_module._parallel_cfg_cache[1], replica)
                    sampling_cleanup(unet)
                sampling_prepare(unet, x)
                try:
                    self.assertIs(unet._experimental_cfg_replica, replica)
                    self.assertIsNotNone(_load_parallel_cfg(unet, cond, uncond, {}, x))
                finally:
                    sampling_cleanup(unet)
                unet.add_patches({"diffusion_model.weight": (torch.ones(2, 2),)}, strength_patch=0.01, online_mode=online)
                sampling_prepare(unet, x)
                self.assertIsNotNone(unet._experimental_cfg_replica)
                self.assertIsNot(unet._experimental_cfg_replica, replica)
                self.assertIsNone(sampling_module._parallel_cfg_cache)
                try:
                    sampling_cleanup(unet)
                finally:
                    mm.free_memory(float("inf"), torch.device("cuda:0"))

    def test_cfg_one_does_not_prepare_a_replica(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.patcher.unet import UnetPatcher
        from backend.sampling.sampling_function import sampling_prepare

        class Model(torch.nn.Linear):
            computation_dtype = torch.float32
            predictor = SimpleNamespace(percent_to_sigma=lambda value: value)
            def get_dtype(self):
                return torch.float32
            def memory_required(self, input_shape):
                return 0

        unet = UnetPatcher(Model(2, 2), torch.device("cuda:0"), torch.device("cpu"))
        with mock.patch.object(args, "parallel_cfg", 1), mock.patch.object(mm, "current_loaded_models", []), mock.patch("backend.multi_gpu.replicas.clone_unet_patcher") as clone:
            try:
                sampling_prepare(unet, torch.ones(1, 1, 2, 2, device="cuda:0"), cfg_scale=1.0)
                clone.assert_not_called()
            finally:
                mm.free_memory(float("inf"), torch.device("cuda:0"))

    def test_single_image_followed_by_parallel_batch_creates_replica(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.patcher.unet import UnetPatcher
        from backend.sampling import sampling_function as sampling_module

        class Model(torch.nn.Linear):
            computation_dtype = torch.float32
            predictor = SimpleNamespace(percent_to_sigma=lambda value: value)
            def get_dtype(self):
                return torch.float32
            def memory_required(self, input_shape):
                return 0

        unet = UnetPatcher(Model(2, 2), torch.device("cuda:0"), torch.device("cpu"))
        with mock.patch.object(args, "parallel_cfg", 1), mock.patch.object(args, "parallel_batch", 1), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(sampling_module, "_parallel_cfg_cache", None), mock.patch.object(sampling_module, "_parallel_batch_model_supported", return_value=True), mock.patch.object(sampling_module, "_parallel_batch_callbacks_safe", return_value=True):
            try:
                sampling_module.sampling_prepare(unet, torch.ones(1, 1, 2, 2, device="cuda:0"), cfg_scale=1.0, cfg1_single_condition=True)
                sampling_module.sampling_cleanup(unet)
                self.assertGreater(unet.loaded_size(), 0)
                sampling_module.sampling_prepare(unet, torch.ones(2, 1, 2, 2, device="cuda:0"), cfg_scale=1.0, cfg1_single_condition=True)
                self.assertIsNotNone(unet._experimental_cfg_replica)
                self.assertIsNot(unet._experimental_cfg_replica.model, unet.model)
            finally:
                sampling_module.sampling_cleanup(unet)
                mm.free_memory(float("inf"), torch.device("cuda:0"))

    def test_primary_load_failure_releases_pending_cpu_replica(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.patcher.unet import UnetPatcher
        from backend.sampling.sampling_function import sampling_prepare

        class Model(torch.nn.Linear):
            computation_dtype = torch.float32
            predictor = SimpleNamespace(percent_to_sigma=lambda value: value)
            def get_dtype(self):
                return torch.float32
            def memory_required(self, input_shape):
                return 0

        unet = UnetPatcher(Model(2, 2), torch.device("cuda:0"), torch.device("cpu"))
        with mock.patch.object(args, "parallel_cfg", 1), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(mm, "load_models_gpu", side_effect=torch.OutOfMemoryError("primary load")):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "primary load"):
                sampling_prepare(unet, torch.ones(1, 1, 2, 2, device="cuda:0"), cfg_scale=2)
        self.assertIsNone(unet._experimental_cfg_replica)

    def test_replica_clones_gguf_and_online_lora_independently(self):
        from gguf import GGMLQuantizationType
        from backend.operations_gguf import ParameterGGUF
        from backend.patcher.unet import UnetPatcher

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = ParameterGGUF(torch.zeros(1, 18, dtype=torch.uint8), tensor_type=GGMLQuantizationType.Q4_0, tensor_shape=torch.Size([1, 32]))
                self.computation_dtype = torch.float16

        original = UnetPatcher(Model(), torch.device("cuda:0"), torch.device("cpu"))
        original.add_patches({"weight": (torch.ones(1, 32),)}, online_mode=True)
        replica = clone_unet_patcher(original, torch.device("cuda:1"))
        self.assertIsNot(replica.model, original.model)
        self.assertIsInstance(replica.model.weight, ParameterGGUF)
        self.assertEqual(replica.model.weight.tensor_type, original.model.weight.tensor_type)
        self.assertEqual(replica.model.weight.shape, original.model.weight.shape)
        self.assertNotEqual(replica.model.weight.data.data_ptr(), original.model.weight.data.data_ptr())
        self.assertIsNot(replica.weight_wrapper_patches["weight"][0], original.weight_wrapper_patches["weight"][0])
        self.assertEqual(replica.patches_uuid, original.patches_uuid)

    def test_replica_clones_offline_lora_state_independently(self):
        from backend.patcher.unet import UnetPatcher

        class Model(torch.nn.Linear):
            computation_dtype = torch.float32

        original = UnetPatcher(Model(4, 4), torch.device("cuda:0"), torch.device("cpu"))
        original.add_patches({"weight": (torch.ones(4, 4),)}, online_mode=False)
        replica = clone_unet_patcher(original, torch.device("cuda:1"))
        self.assertEqual(replica.patches_uuid, original.patches_uuid)
        self.assertIsNot(replica.patches["weight"], original.patches["weight"])
        self.assertIsNot(replica.patches["weight"][0][1][0], original.patches["weight"][0][1][0])

    def test_sampler_dispatch_matches_normal_cfg_and_skips_cfg_one(self):
        from backend.args import dynamic_args
        from backend.sampling.condition import compile_conditions
        from backend.sampling.sampling_function import sampling_function_inner

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.25))

            def memory_required(self, shape):
                return 0

            def apply_model(self, x, timestep, c_crossattn, **kwargs):
                return x * self.scale + c_crossattn[:, :1, :1, None]

        source = Model()
        cond = compile_conditions(torch.full((1, 1, 1), 3.0))
        uncond = compile_conditions(torch.full((1, 1, 1), 1.0))
        x = torch.ones(1, 1, 2, 2, device="cuda:0")
        timestep = torch.ones(1, device="cuda:0")
        with mock.patch.object(dynamic_args, "can_reference", return_value=False), mock.patch.object(dynamic_args, "context_handler", None):
            with ReplicaPair(source, ("cuda:0", "cuda:1")) as pair:
                primary, secondary = pair.models
                adapter = ParallelCFG(primary, secondary, pair.devices)
                try:
                    with torch.inference_mode(), mock.patch.object(adapter, "predict", wraps=adapter.predict) as predict:
                        options = {"transformer_options": {"sampling_sigmas": timestep}}
                        actual = sampling_function_inner(primary, x, timestep, uncond, cond, 2.0, options, return_full=True, parallel_cfg=adapter)
                        self.assertEqual(predict.call_count, 1)
                        skipped = sampling_function_inner(primary, x, timestep, uncond, cond, 1.0, parallel_cfg=adapter)
                        self.assertEqual(predict.call_count, 1)
                        expected = sampling_function_inner(primary, x, timestep, uncond, cond, 2.0, return_full=True)
                        baseline_one = sampling_function_inner(primary, x, timestep, uncond, cond, 1.0)
                    for observed, reference in zip(actual, expected):
                        torch.testing.assert_close(observed, reference)
                    torch.testing.assert_close(skipped, baseline_one)
                finally:
                    adapter.close()

    def test_sampler_rejects_extensions_and_complex_conditioning(self):
        from backend.args import dynamic_args
        from backend.sampling.condition import compile_conditions
        x = torch.ones(1, 1, 2, 2, device="cuda:0")
        cond = compile_conditions(torch.ones(1, 1, 1))
        uncond = compile_conditions(torch.zeros(1, 1, 1))
        with mock.patch.object(dynamic_args, "can_reference", return_value=False), mock.patch.object(dynamic_args, "context_handler", None):
            self.assertTrue(ParallelCFG.eligible(cond, uncond, {}, x))
            self.assertTrue(ParallelCFG.eligible([{**cond[0], "batch_indices": [0]}], [{**uncond[0], "batch_indices": [0]}], {}, x))
            self.assertFalse(ParallelCFG.eligible([{**cond[0], "batch_indices": []}], uncond, {}, x))
            self.assertTrue(ParallelCFG.eligible(cond, uncond, {"transformer_options": {"sampling_sigmas": torch.ones(2)}}, x))
            self.assertFalse(ParallelCFG.eligible(cond, None, {}, x))
            self.assertFalse(ParallelCFG.eligible(cond, uncond, {"model_function_wrapper": lambda *args: None}, x))
            self.assertFalse(ParallelCFG.eligible([{**cond[0], "mask": torch.ones(1, 2, 2)}], uncond, {}, x))
            self.assertFalse(ParallelCFG.eligible(cond + cond, uncond, {}, x))

    def test_parallel_condition_cache_invalidates_on_change(self):
        from backend.sampling.condition import compile_conditions

        adapter = ParallelCFG.__new__(ParallelCFG)
        adapter._conditional_cache = {}
        adapter._conditional_sources = ()
        tensor = torch.ones(1, 1, 1)
        cond = compile_conditions(tensor)
        cache = adapter._condition_cache(cond, "conditional")
        condition = cond[0]["model_conds"]["c_crossattn"]
        first = condition.process_cond(1, torch.device("cuda:1"), cache=cache).cond
        self.assertEqual(first.item(), 1)
        cache["sentinel"] = 1
        self.assertIs(adapter._condition_cache(cond, "conditional"), cache)
        self.assertIn("sentinel", cache)
        tensor.add_(1)
        adapter._condition_cache(cond, "conditional")
        self.assertFalse(cache)
        updated = condition.process_cond(1, torch.device("cuda:1"), cache=cache).cond
        self.assertEqual(updated.item(), 2)
        cache["sentinel"] = 1
        adapter._condition_cache(compile_conditions(tensor.clone()), "conditional")
        self.assertFalse(cache)
        cache["sentinel"] = 1
        changed_key = compile_conditions(tensor.clone())
        adapter._condition_cache(changed_key, "conditional")
        cache["sentinel"] = 1
        changed_key[0]["model_conds"]["c_crossattn"].cache_key = ("changed",)
        adapter._condition_cache(changed_key, "conditional")
        self.assertFalse(cache)
        with torch.inference_mode():
            inference_cond = compile_conditions(torch.ones(1, 1, 1))
        cache["sentinel"] = 1
        self.assertIsNone(adapter._condition_cache(inference_cond, "conditional"))
        self.assertFalse(cache)

    def test_replicas_are_independent_and_match_source(self):
        torch.manual_seed(5)
        source = torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.SiLU(), torch.nn.Linear(12, 4)).eval()
        x = torch.randn(3, 8)
        with torch.inference_mode():
            expected = source(x)
            with ReplicaPair(source, ("cuda:0", "cuda:1")) as pair:
                first, second = pair.models
                self.assertIsNot(first, second)
                self.assertNotEqual(next(first.parameters()).data_ptr(), next(second.parameters()).data_ptr())
                with ParallelPredictions(first, second) as executor:
                    a, b = executor.predict((x.to("cuda:0"),), (x.to("cuda:1"),))
                torch.testing.assert_close(a.cpu(), expected)
                torch.testing.assert_close(b.cpu(), expected)
            self.assertFalse(pair.models)

    def test_replica_loader_rejects_hooks_and_reserves_memory(self):
        model = torch.nn.Linear(4, 4)
        handle = model.register_forward_hook(lambda *unused: None)
        try:
            with self.assertRaisesRegex(UnsupportedReplica, "hooks"):
                ReplicaPair(model, ("cuda:0", "cuda:1"))
        finally:
            handle.remove()
        with self.assertRaisesRegex(UnsupportedReplica, "Insufficient free memory"):
            ReplicaPair(model, ("cuda:0", "cuda:1"), reserve_bytes=10**15)
    def test_column_and_row_match_single_device_for_uneven_shapes(self):
        torch.manual_seed(42)
        weight = torch.randn(7, 9, device="cuda:0")
        bias = torch.randn(7, device="cuda:0")
        x = torch.randn(2, 3, 9, device="cuda:0")
        with torch.inference_mode():
            expected = F.linear(x, weight, bias)
            for partition in ("column", "row"):
                layer = SplitLinear(weight, bias, ("cuda:0", "cuda:1"), partition)
                torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-5)

    def test_bias_is_optional(self):
        weight = torch.randn(6, 8, device="cuda:0")
        x = torch.randn(2, 8, device="cuda:0")
        with torch.inference_mode():
            layer = SplitLinear(weight, None, ("cuda:0", "cuda:1"), "row")
            torch.testing.assert_close(layer(x), F.linear(x, weight), rtol=1e-5, atol=1e-5)
