import unittest
from contextlib import ExitStack
from unittest import mock

import torch

from backend import memory_management as mm
from backend.patcher.base import ModelPatcher


def patcher(device="cpu"):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    return ModelPatcher(model, torch.device(device), torch.device("cpu"))


class ModelResidencyTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(mm, "current_loaded_models", []))
        self.stack.enter_context(mock.patch.object(mm, "vram_state", mm.VRAMState.NORMAL_VRAM))
        self.stack.enter_context(mock.patch.object(mm, "signal_empty_cache", False))

    def test_same_resident_model_skips_patcher_and_keeps_one_registration(self):
        p = patcher()
        mm.load_models_gpu([p])
        loaded = mm.current_loaded_models[0]
        finalizer = loaded.model_finalizer
        with mock.patch.object(p, "partially_load", wraps=p.partially_load) as load:
            mm.load_models_gpu([p])
        load.assert_not_called()
        self.assertEqual(mm.current_loaded_models, [loaded])
        self.assertIs(loaded.model_finalizer, finalizer)

    def test_partially_resident_model_can_load_remaining_weights(self):
        p = patcher()
        p.model.extend([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
        for module in p.model:
            module.parameters_manual_cast = False
            module.weight_function = []
            module.bias_function = []
        loaded = mm.LoadedModel(p)
        loaded.model_load()
        with mock.patch.object(mm, "pin_memory", return_value=False), mock.patch.object(mm, "NUM_STREAMS", 0):
            p.partially_unload(torch.device("cpu"), 1)
        self.assertGreater(p.loaded_size(), 0)
        self.assertLess(p.loaded_size(), p.model_size())
        with mock.patch.object(p, "load", wraps=p.load) as load:
            loaded.model_load(1000)
        load.assert_called_once()
        self.assertEqual(p.loaded_size(), p.model_size())

    def test_gguf_resident_weights_are_not_rebaked(self):
        from backend.operations_gguf import ParameterGGUF
        from modules_forge.packages.gguf import GGMLQuantizationType

        model = torch.nn.Sequential(torch.nn.Linear(32, 1, bias=False))
        model[0].weight = ParameterGGUF(torch.zeros((1, 18), dtype=torch.uint8), tensor_type=GGMLQuantizationType.Q4_0, tensor_shape=torch.Size([1, 32]))
        p = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        loaded = mm.LoadedModel(p)
        loaded.model_load()
        weight = model[0].weight
        expected = weight.gguf_cls.dequantize_pytorch(weight).clone()
        with mock.patch.object(mm, "bake_gguf_model", wraps=mm.bake_gguf_model) as bake:
            loaded.model_load()
        bake.assert_not_called()
        self.assertIs(model[0].weight, weight)
        torch.testing.assert_close(weight.gguf_cls.dequantize_pytorch(weight), expected)

    def test_cleanup_tolerates_pending_model_registration(self):
        p = patcher()
        loaded = mm.LoadedModel(p)
        mm.current_loaded_models.append(loaded)
        mm.cleanup_models()
        self.assertEqual(mm.current_loaded_models, [loaded])

    def test_vram_modes_and_force_full_load_preserve_loading_budgets(self):
        p = patcher("cuda")
        loaded = mm.LoadedModel(p)
        mm.current_loaded_models[:] = [loaded]
        self.stack.enter_context(mock.patch.object(mm, "get_free_memory", return_value=10**12))
        self.stack.enter_context(mock.patch.object(mm, "free_memory"))
        self.stack.enter_context(mock.patch.object(mm, "lowvram_available", True))
        for mode in (mm.VRAMState.NORMAL_VRAM, mm.VRAMState.LOW_VRAM, mm.VRAMState.NO_VRAM, mm.VRAMState.HIGH_VRAM):
            with mock.patch.object(mm, "vram_state", mode), mock.patch.object(loaded, "model_load") as load:
                mm.load_models_gpu([p])
                budget = load.call_args.args[0]
                if mode is mm.VRAMState.NO_VRAM:
                    self.assertEqual(budget, 0.1)
                elif mode is mm.VRAMState.HIGH_VRAM:
                    self.assertEqual(budget, 0)
                else:
                    self.assertGreater(budget, 0)
                mm.load_models_gpu([p], force_full_load=True)
                self.assertTrue(load.call_args.kwargs["force_full_load"])
                if mode is not mm.VRAMState.NO_VRAM:
                    self.assertEqual(load.call_args.args[0], 0)

    def test_force_patch_and_uuid_change_bypass_fast_path(self):
        p = patcher()
        mm.load_models_gpu([p])
        with mock.patch.object(p, "partially_load", wraps=p.partially_load) as load:
            mm.load_models_gpu([p], force_patch_weights=True)
            self.assertEqual(load.call_count, 1)
            mm.load_models_gpu([p], force_full_load=True)
            self.assertEqual(load.call_count, 2)
            p.add_patches({"0.weight": (torch.ones_like(p.model[0].weight),)})
            mm.load_models_gpu([p])
            self.assertEqual(load.call_count, 3)

    def test_clone_replaces_registration_and_applies_object_patch(self):
        p = patcher()
        p.model.tag = "base"
        mm.load_models_gpu([p])
        clone = p.clone()
        clone.object_patches["tag"] = "clone"
        mm.load_models_gpu([clone])
        self.assertEqual(len(mm.current_loaded_models), 1)
        self.assertIs(mm.current_loaded_models[0].model, clone)
        self.assertEqual(p.model.tag, "clone")
        mm.load_models_gpu([p])
        self.assertEqual(p.model.tag, "base")

    def test_dtype_hooks_and_negative_budget_bypass_fast_path(self):
        p = patcher()
        p.model.get_dtype = lambda: p.model[0].weight.dtype
        loaded = mm.LoadedModel(p)
        loaded.model_load()
        with mock.patch.object(p, "partially_load") as load:
            p.model.double()
            loaded.model_load()
            self.assertEqual(load.call_count, 1)
            p.model_options["model_function_wrapper"] = object()
            loaded.model_load()
            self.assertEqual(load.call_count, 2)
            p.model_options = {"transformer_options": {}}
            loaded.model_load(-1)
            self.assertEqual(load.call_count, 3)

    def test_target_eviction_is_deferred_until_after_non_targets(self):
        target, other = patcher("cuda"), patcher("cuda")
        target_loaded, other_loaded = mm.LoadedModel(target), mm.LoadedModel(other)
        mm.current_loaded_models[:] = [target_loaded, other_loaded]
        events = []
        free = [0]

        def unload_other(deficit):
            events.append(("other", deficit))
            free[0] += 70
            return True

        def unload_target(deficit):
            events.append(("target", deficit))
            free[0] += deficit
            return False

        for obj in (target_loaded, other_loaded):
            self.stack.enter_context(mock.patch.object(obj, "model_offloaded_memory", return_value=0))
            self.stack.enter_context(mock.patch.object(obj, "model_memory", return_value=100))
        self.stack.enter_context(mock.patch.object(target_loaded, "model_memory_required", return_value=0))
        self.stack.enter_context(mock.patch.object(target_loaded, "model_load"))
        self.stack.enter_context(mock.patch.object(target_loaded, "model_unload", side_effect=unload_target))
        self.stack.enter_context(mock.patch.object(other_loaded, "model_unload", side_effect=unload_other))
        self.stack.enter_context(mock.patch.object(mm, "minimum_inference_memory", return_value=100))
        self.stack.enter_context(mock.patch.object(mm, "extra_reserved_memory", return_value=0))
        self.stack.enter_context(mock.patch.object(mm, "DISABLE_SMART_MEMORY", False))
        self.stack.enter_context(mock.patch.object(mm, "soft_empty_cache"))
        self.stack.enter_context(mock.patch.object(torch.cuda, "synchronize"))
        self.stack.enter_context(mock.patch.object(mm, "get_free_memory", side_effect=lambda device, torch_free_too=False: (free[0], 0) if torch_free_too else free[0]))
        mm.load_models_gpu([target])
        self.assertEqual(events, [("other", 100), ("target", 30)])
        self.assertEqual(mm.current_loaded_models, [target_loaded])

    def test_resident_target_is_not_offloaded_when_other_eviction_suffices(self):
        # Inspect both memory passes while running real target bookkeeping.
        p = patcher("cuda")
        loaded = mm.LoadedModel(p)
        mm.current_loaded_models[:] = [loaded]
        self.stack.enter_context(mock.patch.object(loaded, "model_load"))
        self.stack.enter_context(mock.patch.object(mm, "get_free_memory", return_value=10**12))
        with mock.patch.object(mm, "free_memory") as free:
            mm.load_models_gpu([p])
        self.assertEqual(free.call_count, 1)
        self.assertEqual(free.call_args.kwargs["keep_loaded"], [loaded])


class LoadSynchronizationTests(unittest.TestCase):
    def test_wait_occurs_once_after_all_module_transfers(self):
        p = patcher()
        events = []
        handles = []
        for module in p.model:
            original = module.to
            handles.append(mock.patch.object(module, "to", side_effect=lambda *a, original=original, **k: (events.append("copy"), original(*a, **k))[1]))
        with ExitStack() as stack:
            for handle in handles:
                stack.enter_context(handle)
            stack.enter_context(mock.patch.object(mm, "is_device_cuda", return_value=True))
            stream = stack.enter_context(mock.patch.object(torch.cuda, "current_stream"))
            stream.return_value.synchronize.side_effect = lambda: events.append("wait")
            p.load(torch.device("cpu"), full_load=True)
        self.assertEqual(events, ["copy", "copy", "wait"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_real_cuda_loading_and_lora_repatch_on_nondefault_stream(self):
        p = patcher("cuda")
        with torch.no_grad():
            p.model[0].weight.fill_(0.25)  # Exact in Forge's default FP16 LoRA computation.
        original = p.model[0].weight.detach().clone()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            p.patch_model(torch.device("cuda"))
            p.add_patches({"0.weight": (torch.ones_like(original),)})
            p.partially_load(torch.device("cuda"), extra_memory=10**9)
        torch.testing.assert_close(p.model[0].weight.cpu(), original + 1)
        p.detach()


if __name__ == "__main__":
    unittest.main()
