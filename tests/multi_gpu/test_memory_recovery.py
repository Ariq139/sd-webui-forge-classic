import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules_forge" / "packages"))

from backend import memory_management as mm
from backend.args import dynamic_args
from backend.diffusion_engine.krea import Krea2
from backend.modules.k_model import KModel
from backend.nn.unet import apply_control
from backend import operations, operations_gguf, operations_mixed_precision


class MemoryRecoveryTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "AcceleratorError"), "accelerator errors unavailable")
    def test_non_oom_accelerator_errors_are_not_retried_as_oom(self):
        with mock.patch.object(mm, "discard_cuda_async_error") as discard:
            self.assertFalse(mm.is_oom(torch.AcceleratorError("device-side assertion")))
            self.assertTrue(mm.is_oom(torch.AcceleratorError("CUDA out of memory")))
        discard.assert_called_once()

    def test_clearing_async_error_is_safe_without_cuda(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False), mock.patch.object(torch.cuda, "synchronize") as synchronize:
            mm.discard_cuda_async_error()
        synchronize.assert_not_called()

    def test_krea_conditioning_oom_retries_after_unloading_both_devices(self):
        primary, secondary = torch.device("cuda:0"), torch.device("cuda:1")
        clip = SimpleNamespace(load_device=primary)
        unet = SimpleNamespace(load_device=primary, secondary_device=secondary)
        engine = Krea2.__new__(Krea2)
        engine.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=clip), unet=unet)
        initial, reference = object(), object()
        engine.ini_latent = initial
        previous_refs = dynamic_args.ref_latents
        self.addCleanup(setattr, dynamic_args, "ref_latents", previous_refs)
        dynamic_args.ref_latents = [reference]
        attempts = []

        def conditioning(prompt):
            attempts.append((engine.ini_latent, list(dynamic_args.ref_latents)))
            if len(attempts) == 1:
                engine.ini_latent = None
                dynamic_args.ref_latents.clear()
                raise torch.OutOfMemoryError("simulated secondary OOM")
            return "ok"

        with mock.patch.object(engine, "_get_learned_conditioning", side_effect=conditioning), mock.patch.object(mm, "unload_models_except") as unload, mock.patch.object(mm, "soft_empty_cache") as empty:
            self.assertEqual(engine.get_learned_conditioning(object()), "ok")
        self.assertEqual(attempts, [(initial, [reference]), (initial, [reference])])
        self.assertEqual({call.kwargs["device"] for call in unload.call_args_list}, {primary, secondary})
        self.assertTrue(all(call.args == ([clip],) for call in unload.call_args_list))
        empty.assert_called_once_with(force=True)

    def test_krea_non_oom_is_not_retried(self):
        engine = Krea2.__new__(Krea2)
        engine.ini_latent = None
        with mock.patch.object(engine, "_get_learned_conditioning", side_effect=ValueError("invalid input")) as conditioning, mock.patch.object(mm, "unload_models_except") as unload:
            with self.assertRaisesRegex(ValueError, "invalid input"):
                engine.get_learned_conditioning(object())
        conditioning.assert_called_once()
        unload.assert_not_called()

    def test_krea_second_oom_is_propagated_without_further_retry(self):
        primary = torch.device("cuda:0")
        clip = SimpleNamespace(load_device=primary)
        engine = Krea2.__new__(Krea2)
        engine.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=clip), unet=SimpleNamespace(load_device=primary))
        engine.ini_latent = None
        old_refs = dynamic_args.ref_latents
        self.addCleanup(setattr, dynamic_args, "ref_latents", old_refs)
        with mock.patch.object(engine, "_get_learned_conditioning", side_effect=torch.OutOfMemoryError("still full")) as conditioning, mock.patch.object(mm, "unload_models_except") as unload, mock.patch.object(mm, "soft_empty_cache"):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "still full"):
                engine.get_learned_conditioning(object())
        self.assertEqual(conditioning.call_count, 2)
        self.assertEqual(unload.call_args_list, [mock.call([clip], device=primary), mock.call([], device=primary)])

    def test_krea_restores_latents_after_retry_also_fails(self):
        primary = torch.device("cuda:0")
        clip = SimpleNamespace(load_device=primary)
        engine = Krea2.__new__(Krea2)
        engine.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=clip), unet=SimpleNamespace(load_device=primary))
        initial, reference = object(), object()
        engine.ini_latent = initial
        old_refs = dynamic_args.ref_latents
        self.addCleanup(setattr, dynamic_args, "ref_latents", old_refs)
        dynamic_args.ref_latents = [reference]

        def fail(prompt):
            engine.ini_latent = None
            dynamic_args.ref_latents.clear()
            raise torch.OutOfMemoryError("still full")

        with mock.patch.object(engine, "_get_learned_conditioning", side_effect=fail) as conditioning, mock.patch.object(mm, "unload_models_except"), mock.patch.object(mm, "soft_empty_cache"):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "still full"):
                engine.get_learned_conditioning(object())
        self.assertEqual(conditioning.call_count, 2)
        self.assertIs(engine.ini_latent, initial)
        self.assertEqual(dynamic_args.ref_latents, [reference])

    def test_krea_final_oom_attempts_both_devices_even_if_cleanup_fails(self):
        primary, secondary = torch.device("cuda:0"), torch.device("cuda:1")
        clip = SimpleNamespace(load_device=primary)
        engine = Krea2.__new__(Krea2)
        engine.ini_latent = None
        engine.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=clip), unet=SimpleNamespace(load_device=primary, secondary_device=secondary))
        old_refs = dynamic_args.ref_latents
        self.addCleanup(setattr, dynamic_args, "ref_latents", old_refs)
        released = []

        def unload(models, *, device):
            if not models:
                released.append(device)
                if device == primary:
                    raise RuntimeError("cleanup failed")

        with mock.patch.object(engine, "_get_learned_conditioning", side_effect=torch.OutOfMemoryError("original OOM")), mock.patch.object(mm, "unload_models_except", side_effect=unload), mock.patch.object(mm, "soft_empty_cache") as empty, mock.patch.object(mm.logger, "exception"):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "original OOM"):
                engine.get_learned_conditioning(object())
        self.assertEqual(set(released), {primary, secondary})
        self.assertEqual(empty.call_count, 2)

    def test_krea_retry_is_aborted_if_initial_cleanup_fails(self):
        primary, secondary = torch.device("cuda:0"), torch.device("cuda:1")
        clip = SimpleNamespace(load_device=primary)
        engine = Krea2.__new__(Krea2)
        engine.ini_latent = None
        engine.forge_objects = SimpleNamespace(clip=SimpleNamespace(patcher=clip), unet=SimpleNamespace(load_device=primary, secondary_device=secondary))
        old_refs = dynamic_args.ref_latents
        self.addCleanup(setattr, dynamic_args, "ref_latents", old_refs)
        attempted = []

        def unload(models, *, device):
            attempted.append(device)
            if device == primary:
                raise RuntimeError("cleanup failure")

        with mock.patch.object(engine, "_get_learned_conditioning", side_effect=torch.OutOfMemoryError("initial OOM")) as conditioning, mock.patch.object(mm, "unload_models_except", side_effect=unload), mock.patch.object(mm, "soft_empty_cache") as empty, mock.patch.object(mm.logger, "exception"):
            with self.assertRaisesRegex(RuntimeError, "retry aborted"):
                engine.get_learned_conditioning(object())
        self.assertEqual(set(attempted), {primary, secondary})
        self.assertEqual(conditioning.call_count, 1)
        empty.assert_called_once_with(force=True)

    def test_cuda_memory_uses_matching_nvml_uuid_without_double_counting(self):
        device = torch.device("cuda:1")
        nvml = SimpleNamespace(nvmlDeviceGetHandleByUUID=mock.Mock(return_value="handle1"), nvmlDeviceGetMemoryInfo=mock.Mock(return_value=SimpleNamespace(free=650)))
        stats = {"active_bytes.all.current": 200, "reserved_bytes.all.current": 300}
        with mock.patch.object(mm, "PYNVML_IS_AVAILABLE", True), mock.patch.object(mm, "pynvml", nvml, create=True), mock.patch.object(mm, "is_intel_xpu", return_value=False), mock.patch.object(torch.cuda, "memory_stats", return_value=stats), mock.patch.object(torch.cuda, "mem_get_info", return_value=(700, 1000)), mock.patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(uuid="secondary-uuid")):
            self.assertEqual(mm.get_free_memory(device), 750)
        nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-secondary-uuid")

    def test_cuda_memory_falls_back_if_nvml_device_lookup_fails(self):
        device = torch.device("cuda:1")
        nvml = SimpleNamespace(nvmlDeviceGetHandleByUUID=mock.Mock(side_effect=RuntimeError("unavailable")))
        stats = {"active_bytes.all.current": 200, "reserved_bytes.all.current": 300}
        with mock.patch.object(mm, "PYNVML_IS_AVAILABLE", True), mock.patch.object(mm, "pynvml", nvml, create=True), mock.patch.object(mm, "is_intel_xpu", return_value=False), mock.patch.object(torch.cuda, "memory_stats", return_value=stats), mock.patch.object(torch.cuda, "mem_get_info", return_value=(700, 1000)), mock.patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(uuid="secondary-uuid")):
            self.assertEqual(mm.get_free_memory(device), 800)

    def test_offload_stream_waits_on_the_selected_device_and_stream(self):
        device = torch.device("cuda:1")
        stream_a, stream_b, active = mock.Mock(), mock.Mock(), object()
        with mock.patch.object(mm, "STREAMS", {}), mock.patch.object(mm, "stream_counters", {}), mock.patch.object(mm, "NUM_STREAMS", 2), mock.patch.object(torch.compiler, "is_compiling", return_value=False), mock.patch.object(torch.cuda, "Stream", side_effect=[stream_a, stream_b]), mock.patch.object(torch.cuda, "current_stream", return_value=active) as current:
            self.assertIs(mm.get_offload_stream(device), stream_a)
            self.assertIs(mm.get_offload_stream(device), stream_b)
            self.assertIs(mm.get_offload_stream(device), stream_a)
        self.assertEqual(current.call_args_list, [mock.call(device), mock.call(device), mock.call(device)])
        self.assertEqual(stream_a.wait_stream.call_args_list, [mock.call(active), mock.call(active)])
        stream_b.wait_stream.assert_called_once_with(active)

    def test_oom_callback_cleanup_does_not_mask_original_error(self):
        class Diffusion:
            _anima_shard_devices = (torch.device("cuda:0"), torch.device("cuda:1"), 1)
            def __call__(self, *args, **kwargs):
                raise torch.OutOfMemoryError("original OOM")

        predictor = SimpleNamespace(calculate_input=lambda sigma, x: x, timestep=lambda t: t)
        model = SimpleNamespace(diffusion_model=Diffusion(), predictor=predictor, computation_dtype=torch.float32)
        x = torch.ones(1)
        with mock.patch("backend.multi_gpu.patches.restore_patch_devices", side_effect=RuntimeError("cleanup failure")) as restore, mock.patch.object(mm, "free_memory") as free, mock.patch.object(mm, "soft_empty_cache") as empty, mock.patch.object(mm.logger, "exception"):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "original OOM"):
                KModel.apply_model(model, x, x, c_crossattn=x)
        self.assertEqual(restore.call_args.args[1], torch.device("cpu"))
        self.assertEqual(free.call_args_list, [mock.call(float("inf"), torch.device("cuda:0")), mock.call(float("inf"), torch.device("cuda:1"))])
        empty.assert_called_once_with(force=True)

    def test_sharded_unet_control_does_not_swallow_oom(self):
        class FailingAdd:
            device = torch.device("cpu")
            def __iadd__(self, value):
                raise torch.OutOfMemoryError("control OOM")

        with self.assertRaisesRegex(torch.OutOfMemoryError, "control OOM"):
            apply_control(FailingAdd(), {"input": [torch.ones(1)]}, "input", move_to_device=True)

    def test_oom_during_input_preparation_releases_both_devices(self):
        diffusion = SimpleNamespace(_anima_shard_devices=(torch.device("cuda:0"), torch.device("cuda:1"), 1))
        predictor = SimpleNamespace(calculate_input=mock.Mock(side_effect=torch.OutOfMemoryError("input OOM")))
        model = SimpleNamespace(diffusion_model=diffusion, predictor=predictor, computation_dtype=torch.float32)
        x = torch.ones(1)
        with mock.patch.object(mm, "free_memory") as free, mock.patch.object(mm, "soft_empty_cache"):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "input OOM"):
                KModel.apply_model(model, x, x, c_crossattn=x)
        self.assertEqual([call.args[1] for call in free.call_args_list], [torch.device("cuda:0"), torch.device("cuda:1")])

    def test_oom_drops_moss_reference_cache_before_reclaiming_memory(self):
        from backend.args import dynamic_args
        cached = torch.ones(1)
        diffusion = SimpleNamespace(_anima_shard_devices=(torch.device("cuda:0"), torch.device("cuda:1"), 1))
        predictor = SimpleNamespace(calculate_input=mock.Mock(side_effect=torch.OutOfMemoryError("input OOM")))
        model = SimpleNamespace(diffusion_model=diffusion, predictor=predictor, computation_dtype=torch.float32)
        def check_cache(*args, **kwargs):
            self.assertIsNone(dynamic_args.moss_ref_kv)
        with mock.patch.object(dynamic_args, "moss_ref_kv", cached), mock.patch.object(mm, "free_memory", side_effect=check_cache) as free, mock.patch.object(mm, "soft_empty_cache"):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "input OOM"):
                KModel.apply_model(model, cached, cached, c_crossattn=cached)
            self.assertEqual(free.call_count, 2)

    def test_batched_reference_oom_does_not_restore_cached_gpu_tensors(self):
        from backend.args import dynamic_args
        from backend.sampling.sampling_function import conditioning_batch
        refs, cached = [torch.ones(2, 1)], torch.ones(2, 1)
        with mock.patch.object(dynamic_args, "ref_latents", refs), mock.patch.object(dynamic_args, "moss_ref_kv", cached):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "batch OOM"):
                with conditioning_batch(None, [0], 2):
                    dynamic_args.moss_ref_kv = torch.ones(1)
                    raise torch.OutOfMemoryError("batch OOM")
            self.assertIs(dynamic_args.ref_latents, refs)
            self.assertIsNone(dynamic_args.moss_ref_kv)

    def test_batched_reference_restores_cache_after_other_errors(self):
        from backend.args import dynamic_args
        from backend.sampling.sampling_function import conditioning_batch
        refs, cached = [torch.ones(2, 1)], torch.ones(2, 1)
        with mock.patch.object(dynamic_args, "ref_latents", refs), mock.patch.object(dynamic_args, "moss_ref_kv", cached):
            with self.assertRaisesRegex(ValueError, "other failure"):
                with conditioning_batch(None, [0], 2):
                    raise ValueError("other failure")
            self.assertIs(dynamic_args.ref_latents, refs)
            self.assertIs(dynamic_args.moss_ref_kv, cached)

    def test_oom_restoring_callback_releases_both_devices(self):
        class Diffusion:
            _anima_shard_devices = (torch.device("cuda:0"), torch.device("cuda:1"), 1)
            def __call__(self, x, *args, **kwargs):
                return x

        predictor = SimpleNamespace(calculate_input=lambda sigma, x: x, timestep=lambda t: t, calculate_denoised=lambda sigma, output, x: output)
        model = SimpleNamespace(diffusion_model=Diffusion(), predictor=predictor, computation_dtype=torch.float32)
        x = torch.ones(1)
        with mock.patch("backend.multi_gpu.patches.restore_patch_devices", side_effect=torch.OutOfMemoryError("restore OOM")), mock.patch.object(mm, "free_memory") as free, mock.patch.object(mm, "soft_empty_cache"):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "restore OOM"):
                KModel.apply_model(model, x, x, c_crossattn=x)
        self.assertEqual([call.args[1] for call in free.call_args_list], [torch.device("cuda:0"), torch.device("cuda:1")])

    def test_gguf_kernel_oom_does_not_fall_back_to_dequantization(self):
        weight = SimpleNamespace(gguf_cls=object(), tensor_type=SimpleNamespace(name="Q4_0"), data=torch.ones(2), real_shape=(2,))
        input_tensor = SimpleNamespace(device=torch.device("cuda:1"), dtype=torch.float16)
        module = SimpleNamespace(may_support_linear_qtype_name=lambda name: True, may_support_embedding_qtype_name=lambda name: True, linear=mock.Mock(side_effect=torch.OutOfMemoryError("kernel OOM")), embedding=mock.Mock(side_effect=torch.OutOfMemoryError("kernel OOM")))
        with mock.patch.object(operations_gguf, "_llamacpp_cuda_module", return_value=module):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "kernel OOM"):
                operations_gguf.try_llamacpp_cuda_linear(weight, input_tensor, None)
            with self.assertRaisesRegex(torch.OutOfMemoryError, "kernel OOM"):
                operations_gguf.try_llamacpp_cuda_embedding(weight, input_tensor, torch.float16)

    def test_quantized_parameter_oom_does_not_retry_with_clone(self):
        module = torch.nn.Linear(2, 2, bias=False)
        with mock.patch.object(module, "register_parameter", side_effect=torch.OutOfMemoryError("registration OOM")) as register:
            with self.assertRaisesRegex(torch.OutOfMemoryError, "registration OOM"):
                operations_mixed_precision._quantized_apply(module, lambda tensor: tensor, recurse=False)
        self.assertEqual(register.call_count, 1)

    def test_fp8_kernel_oom_is_not_silently_retried(self):
        module = operations.ForgeOperationsFP8.Linear(2, 2)
        with mock.patch.object(operations, "fp8_linear", side_effect=torch.OutOfMemoryError("FP8 OOM")):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "FP8 OOM"):
                module(torch.ones(1, 2))

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "FP8 dtype unavailable")
    def test_fast_fp8_requires_support_on_both_devices(self):
        primary, secondary = torch.device("cuda:0"), torch.device("cuda:1")
        with mock.patch.object(operations.args, "fast_fp8", True), mock.patch.object(mm, "get_torch_device", return_value=primary), mock.patch.object(mm, "supports_fp8_compute", side_effect=lambda device: device == primary), mock.patch.object(dynamic_args, "ops", None):
            with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float8_e4m3fn, quant_devices=(secondary,)):
                self.assertIs(torch.nn.Linear, operations.ForgeOperations.Linear)
