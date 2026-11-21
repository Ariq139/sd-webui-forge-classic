import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from backend import memory_management as mm
from backend.args import args
from backend.multi_gpu.patcher import ShardedAnimaPatcher
from backend.multi_gpu.planner import InsufficientDeviceMemory
from backend.modules.k_model import KModel
from backend.nn.anima import Anima
from backend.operations import using_forge_operations


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            self.diffusion_model = Anima(in_channels=2, out_channels=2, patch_spatial=2, patch_temporal=1, model_channels=48, crossattn_emb_channels=48, adaln_lora_dim=8, num_blocks=2, num_heads=4)
        for parameter in self.parameters():
            torch.nn.init.uniform_(parameter, -0.02, 0.02)
        self.computation_dtype = torch.float32


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "two CUDA devices required")
class AnimaShardingTests(unittest.TestCase):
    def test_loader_forward_and_eviction_on_secondary(self):
        torch.manual_seed(42)
        model = Model().eval()
        reference = copy.deepcopy(model.diffusion_model).to("cuda:0")
        x = torch.randn(1, 2, 1, 4, 4, device="cuda:0")
        timestep = torch.ones(1, device="cuda:0")
        context = torch.randn(1, 3, 48, device="cuda:0")
        with torch.inference_mode():
            expected = reference(x, timestep, context)
        del reference
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            mm.load_models_gpu([patcher])
            self.assertEqual(next(model.diffusion_model.blocks[0].parameters()).device, torch.device("cuda:0"))
            self.assertEqual(next(model.diffusion_model.blocks[1].parameters()).device, torch.device("cuda:1"))
            with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                actual = model.diffusion_model(x, timestep, context)
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            gpu_events = {event.device_index for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA}
            self.assertTrue({0, 1}.issubset(gpu_events), gpu_events)
            mm.free_memory(float("inf"), torch.device("cuda:1"))
            self.assertFalse(mm.current_loaded_models)
            self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
            self.assertFalse(hasattr(model.diffusion_model, "_anima_shard_devices"))

    def test_secondary_weight_patch_is_restored_after_eviction(self):
        model = Model().eval()
        name = "diffusion_model.blocks.1.mlp.layer1.weight"
        original = dict(model.named_parameters())[name].detach().clone()
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            self.assertEqual(patcher.add_patches({name: (torch.ones_like(original),)}, strength_patch=0.1), [name])
            mm.load_models_gpu([patcher])
            weight = dict(model.named_parameters())[name]
            self.assertEqual(weight.device, torch.device("cuda:1"))
            torch.testing.assert_close(weight.cpu(), original + 0.1, atol=1e-4, rtol=1e-3)
            mm.free_memory(float("inf"), torch.device("cuda:1"))
            torch.testing.assert_close(dict(model.named_parameters())[name], original)

    def test_failed_load_restores_cpu_residency(self):
        model = Model().eval()
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            with mock.patch.object(model.diffusion_model.blocks[1], "to", side_effect=RuntimeError("simulated load error")):
                with self.assertRaisesRegex(RuntimeError, "simulated load error"):
                    mm.load_models_gpu([patcher])
            self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
            self.assertIsNone(model._anima_shard)
            self.assertFalse(mm.current_loaded_models)

    def test_failed_extension_move_attempts_offload(self):
        model = Model().eval()
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            with mock.patch.object(patcher, "model_patches_to", side_effect=[RuntimeError("callback move failed"), None]) as move:
                with self.assertRaisesRegex(RuntimeError, "callback move failed"):
                    mm.load_models_gpu([patcher])
            self.assertEqual(move.call_args_list, [mock.call(torch.device("cuda:0")), mock.call(torch.device("cpu"))])
            self.assertFalse(mm.current_loaded_models)

    def test_failed_rollback_remains_registered_for_later_eviction(self):
        model = Model().eval()
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(mm.logger, "exception"):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            with mock.patch.object(model.diffusion_model.blocks[1], "to", side_effect=RuntimeError("load failure")), mock.patch.object(patcher, "unpatch_model", side_effect=RuntimeError("rollback failure")):
                with self.assertRaisesRegex(RuntimeError, "load failure"):
                    mm.load_models_gpu([patcher])
            self.assertEqual(len(mm.current_loaded_models), 1)
            self.assertEqual(patcher.resident_devices(), {torch.device("cuda:0"), torch.device("cuda:1")})
            self.assertIsNone(model._anima_residency)
            self.assertFalse(hasattr(model.diffusion_model, "_anima_shard_devices"))
            mm.free_memory(float("inf"), torch.device("cuda:1"))
            self.assertFalse(mm.current_loaded_models)
            self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))

    def test_post_load_oom_releases_shards(self):
        model = Model().eval()
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            with mock.patch.object(mm, "bake_gguf_model", side_effect=torch.OutOfMemoryError("post-load OOM")):
                with self.assertRaisesRegex(torch.OutOfMemoryError, "post-load OOM"):
                    mm.load_models_gpu([patcher])
            self.assertFalse(mm.current_loaded_models)
            self.assertIsNone(model._anima_shard)
            self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))

    def test_post_load_oom_with_failed_cleanup_remains_tracked(self):
        model = Model().eval()
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(mm.logger, "exception"):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            with mock.patch.object(mm, "bake_gguf_model", side_effect=torch.OutOfMemoryError("post-load OOM")), mock.patch.object(patcher, "detach", side_effect=RuntimeError("cleanup failed")):
                with self.assertRaisesRegex(torch.OutOfMemoryError, "post-load OOM"):
                    mm.load_models_gpu([patcher])
            self.assertEqual(len(mm.current_loaded_models), 1)
            self.assertIsNotNone(model._anima_shard)
            mm.free_memory(float("inf"), torch.device("cuda:1"))
            self.assertFalse(mm.current_loaded_models)
            self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))

    def test_changed_patches_replan_after_old_shards_are_unloaded(self):
        model = Model().eval()
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            mm.load_models_gpu([patcher])
            key = "diffusion_model.blocks.1.mlp.layer1.weight"
            patcher.add_patches({key: (torch.ones_like(dict(model.named_parameters())[key].cpu()),)}, strength_patch=0.1)
            seen = []
            plan = patcher.device_memory_requirements

            def record_plan(reserve):
                seen.append((model._anima_shard, reserve))
                return plan(reserve)

            with mock.patch.object(patcher, "device_memory_requirements", side_effect=record_plan):
                mm.load_models_gpu([patcher], memory_required=2 * 1024**3)
            self.assertEqual(len(seen), 1)
            self.assertIsNone(seen[0][0])
            self.assertGreaterEqual(seen[0][1], 2 * 1024**3)
            self.assertIsNotNone(model._anima_shard)
            mm.free_memory(float("inf"), torch.device("cuda:1"))

    def test_force_cast_is_installed_and_restored(self):
        model = Model().eval()
        module = model.diffusion_model.blocks[1].mlp.layer1
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            patcher.force_cast_weights = True
            mm.load_models_gpu([patcher])
            self.assertTrue(module.forge_force_cast_weights)
            self.assertTrue(module.parameters_manual_cast)
            scratch = max(parameter.numel() * model.computation_dtype.itemsize * 2 for parameter in model.parameters())
            self.assertEqual(patcher.device_memory_requirements(0), {torch.device("cuda:0"): scratch, torch.device("cuda:1"): scratch})
            mm.free_memory(float("inf"), torch.device("cuda:1"))
            self.assertFalse(module.parameters_manual_cast)

    def test_manual_cast_reserves_transient_weight_memory(self):
        model = Model().eval()
        module = model.diffusion_model.blocks[1].mlp.layer1
        module.parameters_manual_cast = True
        with mock.patch.object(args, "multi_gpu_anima", 1):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            expected = module.weight.numel() * model.computation_dtype.itemsize * 2
            self.assertGreaterEqual(patcher._runtime_scratch(), expected)

    def test_force_cast_change_on_resident_clone_reloads_weights(self):
        model = Model().eval()
        module = model.diffusion_model.blocks[1].mlp.layer1
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            mm.load_models_gpu([patcher])
            clone = patcher.clone()
            clone.force_cast_weights = True
            mm.load_models_gpu([clone])
            self.assertTrue(module.forge_force_cast_weights)
            self.assertTrue(module.parameters_manual_cast)
            self.assertTrue(model._multi_gpu_force_cast_weights)
            mm.free_memory(float("inf"), torch.device("cuda:1"))

    def test_unsupported_object_patch_is_rejected_while_resident(self):
        model = Model().eval()
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            mm.load_models_gpu([patcher])
            patcher.add_object_patch("diffusion_model.blocks.0", object())
            with self.assertRaisesRegex(RuntimeError, "object patches"):
                mm.load_models_gpu([patcher])
            self.assertIsNotNone(model._anima_shard)
            patcher.object_patches.clear()
            mm.free_memory(float("inf"), torch.device("cuda:1"))

    def test_rejected_clone_keeps_resident_model_registered(self):
        model = Model().eval()
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            mm.load_models_gpu([patcher])
            clone = patcher.clone()
            clone.add_object_patch("diffusion_model.blocks.0", object())
            with self.assertRaisesRegex(RuntimeError, "object patches"):
                mm.load_models_gpu([clone])
            self.assertEqual(len(mm.current_loaded_models), 1)
            self.assertIs(mm.current_loaded_models[0].model, clone)
            self.assertIsNotNone(model._anima_shard)
            clone.object_patches.clear()
            mm.free_memory(float("inf"), torch.device("cuda:1"))
            self.assertFalse(mm.current_loaded_models)
            self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))

    def test_misplaced_shard_is_restored_on_next_load(self):
        model = Model().eval()
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
            mm.load_models_gpu([patcher])
            model.diffusion_model.blocks[1].to("cpu")
            with mock.patch.object(patcher, "device_memory_requirements", wraps=patcher.device_memory_requirements) as plan:
                mm.load_models_gpu([patcher])
            self.assertTrue(plan.called)
            model._anima_residency.validate(model)
            mm.free_memory(float("inf"), torch.device("cuda:1"))

    def test_resident_shards_are_replanned_when_headroom_shrinks(self):
        model = Model().eval()
        primary, secondary = torch.device("cuda:0"), torch.device("cuda:1")
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, primary, torch.device("cpu"))
            mm.load_models_gpu([patcher])
            original_free = mm.get_free_memory

            def free_memory(device, *args, **kwargs):
                if device == secondary and model._anima_shard is not None:
                    return 0
                return original_free(device, *args, **kwargs)

            with mock.patch.object(mm, "get_free_memory", side_effect=free_memory), mock.patch.object(patcher, "detach", wraps=patcher.detach) as detach:
                mm.load_models_gpu([patcher])
            self.assertGreaterEqual(detach.call_count, 1)
            self.assertIsNotNone(model._anima_shard)
            mm.free_memory(float("inf"), secondary)

    def test_failed_partition_reclaims_both_devices_and_retries(self):
        model = Model().eval()
        primary, secondary = torch.device("cuda:0"), torch.device("cuda:1")
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, primary, torch.device("cpu"))
            requirements = {primary: 1, secondary: 1}
            with mock.patch.object(patcher, "device_memory_requirements", side_effect=[InsufficientDeviceMemory("no fit"), requirements]) as plan, mock.patch.object(mm, "free_memory") as free, mock.patch.object(mm, "get_free_memory", return_value=10**12), mock.patch.object(mm.LoadedModel, "model_load"):
                mm.load_models_gpu([patcher])
            self.assertEqual(plan.call_count, 2)
            reclaimed = [call.args[1] for call in free.call_args_list if call.args[0] == float("inf")]
            self.assertEqual(reclaimed, [primary, secondary])

    def test_combined_requirements_fail_before_loading_when_device_is_full(self):
        model = Model().eval()
        primary, secondary = torch.device("cuda:0"), torch.device("cuda:1")
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, primary, torch.device("cpu"))
            with mock.patch.object(patcher, "device_memory_requirements", return_value={primary: 1, secondary: 10**11}), mock.patch.object(mm, "free_memory"), mock.patch.object(mm, "get_free_memory", return_value=10**10), mock.patch.object(mm.LoadedModel, "model_load") as load:
                with self.assertRaisesRegex(torch.OutOfMemoryError, "cuda:1"):
                    mm.load_models_gpu([patcher])
            load.assert_not_called()

    def test_failed_partition_reports_oom_after_reclaiming_both_devices(self):
        model = Model().eval()
        primary, secondary = torch.device("cuda:0"), torch.device("cuda:1")
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, primary, torch.device("cpu"))
            with mock.patch.object(patcher, "device_memory_requirements", side_effect=InsufficientDeviceMemory("no fit")) as plan, mock.patch.object(mm, "free_memory") as free:
                with self.assertRaisesRegex(torch.OutOfMemoryError, "placement cannot fit") as failure:
                    mm.load_models_gpu([patcher])
            self.assertIsInstance(failure.exception.__cause__, InsufficientDeviceMemory)
            self.assertEqual(plan.call_count, 2)
            self.assertEqual([call.args[1] for call in free.call_args_list], [primary, secondary])

    def test_failed_partition_attempts_both_devices_when_reclaim_fails(self):
        model = Model().eval()
        primary, secondary = torch.device("cuda:0"), torch.device("cuda:1")
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, primary, torch.device("cpu"))
            with mock.patch.object(patcher, "device_memory_requirements", side_effect=InsufficientDeviceMemory("no fit")) as plan, mock.patch.object(mm, "free_memory", side_effect=[RuntimeError("cleanup failed"), []]) as free, mock.patch.object(mm.logger, "exception"):
                with self.assertRaisesRegex(RuntimeError, "retry aborted"):
                    mm.load_models_gpu([patcher])
            self.assertEqual(plan.call_count, 1)
            self.assertEqual([call.args[1] for call in free.call_args_list], [primary, secondary])

    def test_callback_cleanup_failure_still_offloads_and_keeps_model_tracked(self):
        model = Model().eval()
        primary, secondary = torch.device("cuda:0"), torch.device("cuda:1")
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(model, primary, torch.device("cpu"))
            mm.load_models_gpu([patcher])
            with mock.patch.object(patcher, "model_patches_to", side_effect=RuntimeError("callback cleanup failed")):
                with self.assertRaisesRegex(RuntimeError, "callback cleanup failed"):
                    mm.free_memory(float("inf"), secondary)
            self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
            self.assertEqual(len(mm.current_loaded_models), 1)
            self.assertIsNotNone(model._anima_shard)
            mm.free_memory(float("inf"), secondary)
            self.assertFalse(mm.current_loaded_models)

    def test_denoiser_oom_evicts_the_entire_sharded_model(self):
        root = Model().eval()
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(mm, "soft_empty_cache"):
            patcher = ShardedAnimaPatcher(root, torch.device("cuda:0"), torch.device("cpu"))
            mm.load_models_gpu([patcher])
            predictor = SimpleNamespace(calculate_input=lambda sigma, x: x, timestep=lambda t: t)
            model = SimpleNamespace(diffusion_model=root.diffusion_model, predictor=predictor, computation_dtype=torch.float32)
            x = torch.ones(1, device="cuda:0")
            with mock.patch.object(root.diffusion_model, "forward", side_effect=torch.OutOfMemoryError("simulated OOM")):
                with self.assertRaisesRegex(torch.OutOfMemoryError, "simulated OOM"):
                    KModel.apply_model(model, x, x, c_crossattn=x)
            self.assertFalse(mm.current_loaded_models)
            self.assertTrue(all(parameter.device.type == "cpu" for parameter in root.parameters()))
            self.assertIsNone(root._anima_shard)


if __name__ == "__main__":
    unittest.main()
