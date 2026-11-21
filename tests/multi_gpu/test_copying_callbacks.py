import unittest
from types import SimpleNamespace

import torch

from backend.modules.k_model import KModel
from backend.multi_gpu.patches import device_aware_options, restore_patch_devices
from backend.patcher.base import ModelPatcher


class CopyingCallback:
    def __init__(self, device=None, calls=0, fail=False):
        self.device = device
        self.calls = calls
        self.fail = fail

    def to(self, device):
        return type(self)(torch.device(device), self.calls, self.fail)

    def __call__(self, tensor, *unused):
        if tensor.device != self.device:
            raise RuntimeError("callback on wrong device")
        self.calls += 1
        if self.fail:
            raise RuntimeError("callback failed")
        return tensor


class CallbackMovementTests(unittest.TestCase):
    def test_shared_copying_callback_stays_shared_during_model_moves(self):
        original = CopyingCallback()
        options = {"transformer_options": {"patches": {"one": [original]}, "patches_replace": {"dit": {("single_block", 0): original}}}, "model_function_wrapper": original}
        ModelPatcher.model_patches_to(SimpleNamespace(model_options=options), torch.device("cpu"))
        moved = options["transformer_options"]["patches"]["one"][0]
        self.assertIsNot(moved, original)
        self.assertIs(moved, options["transformer_options"]["patches_replace"]["dit"][("single_block", 0)])
        self.assertIs(moved, options["model_function_wrapper"])
        self.assertEqual(moved.device, torch.device("cpu"))

    def test_in_place_callback_with_no_return_is_preserved(self):
        class InPlaceCallback:
            def __init__(self):
                self.moves = 0
            def to(self, device):
                self.moves += 1
            def __call__(self, value):
                return value
        original = InPlaceCallback()
        options = {"transformer_options": {"patches": {"one": [original, original]}}, "model_function_wrapper": original}
        ModelPatcher.model_patches_to(SimpleNamespace(model_options=options), torch.device("cpu"))
        self.assertEqual(original.moves, 1)
        self.assertIs(options["transformer_options"]["patches"]["one"][0], original)
        self.assertIs(options["model_function_wrapper"], original)

    def test_modifier_callbacks_follow_model_moves_and_keep_aliases(self):
        original = CopyingCallback()
        shared = [original]
        transformer_options = {"patches": {"one": [original]}, "block_modifiers": shared, "block_inner_modifiers": shared, "group_norm_wrapper": original}
        options = {"transformer_options": transformer_options}
        ModelPatcher.model_patches_to(SimpleNamespace(model_options=options), torch.device("cpu"))
        moved = transformer_options["patches"]["one"][0]
        self.assertIsNot(moved, original)
        self.assertIs(transformer_options["block_modifiers"][0], moved)
        self.assertIs(transformer_options["block_inner_modifiers"][0], moved)
        self.assertIs(transformer_options["group_norm_wrapper"], moved)
        self.assertEqual(moved.device, torch.device("cpu"))

    def test_modifier_callbacks_are_not_sent_dtype_as_device(self):
        original = CopyingCallback()
        options = {"transformer_options": {"block_modifiers": [original], "group_norm_wrapper": original}}
        ModelPatcher.model_patches_to(SimpleNamespace(model_options=options), torch.float16)
        self.assertIs(options["transformer_options"]["block_modifiers"][0], original)
        self.assertIs(options["transformer_options"]["group_norm_wrapper"], original)

    def test_dtype_move_propagates_shared_wrapper_replacement_to_modifiers(self):
        class Wrapper:
            def to(self, target):
                return Wrapper()
            def __call__(self, value):
                return value
        original = Wrapper()
        options = {"transformer_options": {"block_modifiers": [original], "group_norm_wrapper": original}, "model_function_wrapper": original}
        ModelPatcher.model_patches_to(SimpleNamespace(model_options=options), torch.float16)
        moved = options["model_function_wrapper"]
        self.assertIsNot(moved, original)
        self.assertIs(options["transformer_options"]["block_modifiers"][0], moved)
        self.assertIs(options["transformer_options"]["group_norm_wrapper"], moved)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "two CUDA devices required")
class CopyingCallbackTests(unittest.TestCase):
    def test_replacement_propagates_to_aliases_and_next_pass(self):
        original = CopyingCallback()
        owner = {"patches": {"first": [original]}, "patches_replace": {"dit": {("single_block", 0): original}}}
        conditional = {"patches": {"first": [original]}}
        runtime = {"patches": {"first": owner["patches"]["first"] + conditional["patches"]["first"]}, "patches_replace": owner["patches_replace"]}
        wrapped = device_aware_options(runtime)
        first, second = wrapped["patches"]["first"]
        self.assertIs(first, second)
        first(torch.ones(1, device="cuda:0"))
        second(torch.ones(1, device="cuda:1"))
        restore_patch_devices(wrapped, torch.device("cuda:0"), (runtime, owner, conditional))
        final = owner["patches"]["first"][0]
        self.assertIsNot(final, original)
        self.assertIs(final, conditional["patches"]["first"][0])
        self.assertIs(final, owner["patches_replace"]["dit"][("single_block", 0)])
        self.assertEqual(final.device, torch.device("cuda:0"))
        self.assertEqual(final.calls, 2)
        second_pass = device_aware_options(owner)
        second_pass["patches"]["first"][0](torch.ones(1, device="cuda:1"))
        restore_patch_devices(second_pass, torch.device("cuda:0"), (owner,))
        self.assertEqual(owner["patches"]["first"][0].calls, 3)
        self.assertIs(owner["patches"]["first"][0], owner["patches_replace"]["dit"][("single_block", 0)])

    def test_model_exception_restores_latest_callback(self):
        original = CopyingCallback(fail=True)
        owner = {"patches": {"sample": [original]}}
        options = {"patches": owner["patches"], "_multi_gpu_patch_sources": [owner]}

        class Diffusion:
            _anima_shard_devices = (torch.device("cuda:0"), torch.device("cuda:1"), 1)
            def __call__(self, x, t, *, transformer_options, **kwargs):
                return transformer_options["patches"]["sample"][0](x.to("cuda:1"))

        predictor = SimpleNamespace(calculate_input=lambda sigma, x: x, timestep=lambda t: t)
        model = SimpleNamespace(diffusion_model=Diffusion(), predictor=predictor, computation_dtype=torch.float32)
        x = torch.ones(1, device="cuda:0")
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            KModel.apply_model(model, x, x, c_crossattn=x, transformer_options=options)
        restored = owner["patches"]["sample"][0]
        self.assertIsNot(restored, original)
        self.assertEqual(restored.device, torch.device("cuda:0"))
        self.assertEqual(restored.calls, 1)
