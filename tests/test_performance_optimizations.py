import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from backend import operations
from backend.args import args
from backend.nn.flux import EmbedND
from backend.nn.krea import _imgids
from backend.operations_lora import activation_space_lora_patches
from backend.operations_mixed_precision import mixed_precision_ops
from backend.patcher.base import OnlineLoRAPatch
from backend.quant_ops import ck
from backend.sampling.condition import ConditionCrossAttn
from backend.sampling.sampling_function import (
    calc_cond_uncond_batch,
    compute_cond_indices,
    compute_cond_mark,
    get_area_and_mult,
    sampling_function_inner,
    should_skip_unconditional,
    single_condition_fast_path_eligible,
    _cached_cond_indices,
)
from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter


def online_lora(up, down, *, strength=1.0, alpha=None, offset=None, function=None):
    adapter = LoRAAdapter(set(), (up, down, alpha, None, None, None))
    return OnlineLoRAPatch("test", "weight", [strength, adapter, 1.0, offset, function])


class PerformanceOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.flag = mock.patch.object(args, "activation_space_lora", True, create=True)
        self.flag.start()
        self.addCleanup(self.flag.stop)

    def test_activation_lora_matches_dense_linear_merge(self):
        up = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        down = torch.tensor([[2.0, 1.0, -1.0], [-1.0, 0.25, 2.0]])
        strength, alpha = 0.5, 2.0
        patch = online_lora(up, down, strength=strength, alpha=alpha)
        patches = activation_space_lora_patches([patch], [], 3, 2)
        self.assertIsNotNone(patches)

        weight = torch.tensor([[0.2, 0.3, 0.5], [-0.1, 0.4, 0.8]])
        inputs = torch.tensor([[1.0, 2.0, -1.0], [0.25, -0.5, 3.0]])
        base = torch.nn.functional.linear(inputs, weight)
        result = operations.apply_activation_space_lora(base, inputs, patches)
        merged_weight = weight + strength * (alpha / down.shape[0]) * (up @ down)
        expected = torch.nn.functional.linear(inputs, merged_weight)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_unit_strength_lora_weight_merge_keeps_expected_delta(self):
        up = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        down = torch.tensor([[2.0, 1.0, -1.0], [-1.0, 0.25, 2.0]])
        adapter = LoRAAdapter(set(), (up, down, 2.0, None, None, None))
        weight = torch.zeros((2, 3))

        result = adapter.calculate_weight(
            weight, "weight", 1.0, 1.0, None, lambda value: value, intermediate_dtype=torch.float32
        )

        torch.testing.assert_close(result, up @ down)

    def test_activation_lora_runs_through_forge_linear(self):
        with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            layer = torch.nn.Linear(3, 2, bias=False)
        layer.weight.data.copy_(torch.tensor([[0.2, 0.3, 0.5], [-0.1, 0.4, 0.8]]))
        up = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        down = torch.tensor([[2.0, 1.0, -1.0], [-1.0, 0.25, 2.0]])
        layer.weight_function = [online_lora(up, down, strength=0.5, alpha=2.0)]
        inputs = torch.tensor([[1.0, 2.0, -1.0], [0.25, -0.5, 3.0]])

        result = layer(inputs)
        merged = layer.weight + 0.5 * (up @ down)
        torch.testing.assert_close(result, torch.nn.functional.linear(inputs, merged), rtol=1e-5, atol=1e-6)

    def test_activation_lora_runs_through_mixed_precision_linear(self):
        ops = mixed_precision_ops(compute_dtype=torch.float32)
        layer = ops.Linear(3, 2, bias=False, device="cpu")
        layer.weight = torch.nn.Parameter(torch.tensor([[0.2, 0.3, 0.5], [-0.1, 0.4, 0.8]]))
        up = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        down = torch.tensor([[2.0, 1.0, -1.0], [-1.0, 0.25, 2.0]])
        layer.weight_function = [online_lora(up, down, strength=0.5, alpha=2.0)]
        inputs = torch.tensor([[1.0, 2.0, -1.0], [0.25, -0.5, 3.0]])

        result = layer(inputs)
        merged = layer.weight + 0.5 * (up @ down)
        torch.testing.assert_close(result, torch.nn.functional.linear(inputs, merged), rtol=1e-5, atol=1e-6)

    def test_unsupported_online_lora_uses_legacy_path(self):
        up = torch.ones((2, 2))
        down = torch.ones((2, 3))
        for patch in (
            online_lora(up, down, offset=(0, 0, 1)),
            online_lora(up, down, function=lambda value: value),
        ):
            self.assertIsNone(activation_space_lora_patches([patch], [], 3, 2))
        self.assertIsNone(activation_space_lora_patches([online_lora(up, down)], [object()], 3, 2))

    def test_singleton_krea_position_ids_broadcast_without_copy(self):
        positions = _imgids(3, 0, 2, 4, torch.device("cpu"))
        self.assertEqual(tuple(positions.shape), (3, 8, 3))
        self.assertEqual(positions.stride(0), 0)
        self.assertTrue(torch.equal(positions[0], positions[2]))
        self.assertTrue(torch.equal(positions[0, :, 0], torch.zeros(8)))
        self.assertTrue(torch.equal(positions[0, :, 1], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])))

    def test_singleton_krea_rope_matches_per_item_positions(self):
        image_ids = _imgids(1, 0, 2, 2, torch.device("cpu"))
        text_ids = torch.zeros((1, 3, 3))
        positions = torch.cat((text_ids, image_ids), dim=1)
        embed = EmbedND(dim=6, theta=10000, axes_dim=[2, 2, 2])
        shared_freqs = embed(positions)
        repeated_freqs = embed(positions.expand(3, -1, -1))
        q = torch.randn((3, 7, 2, 6))
        k = torch.randn_like(q)
        with torch.inference_mode():
            shared_q, shared_k = ck.rms_rope_(
                q.clone(), k.clone(), shared_freqs.transpose(1, 2), torch.ones(6), torch.ones(6), 1e-6
            )
            repeated_q, repeated_k = ck.rms_rope_(
                q.clone(), k.clone(), repeated_freqs.transpose(1, 2), torch.ones(6), torch.ones(6), 1e-6
            )
        torch.testing.assert_close(shared_q, repeated_q)
        torch.testing.assert_close(shared_k, repeated_k)

    def test_sampling_mask_broadcast_preserves_values_and_avoids_channel_copy(self):
        x = torch.zeros((3, 4, 2, 2))
        mask = torch.tensor([[[0.0, 0.25], [0.5, 1.0]]])
        conds = {"mask": mask, "mask_strength": 0.5, "strength": 0.25, "model_conds": {}}

        result = get_area_and_mult(conds, x, torch.tensor([1.0]))
        expected = mask.expand(3, -1, -1).unsqueeze(1).expand(-1, 4, -1, -1) * 0.125
        self.assertEqual(tuple(result.mult.shape), (3, 4, 2, 2))
        torch.testing.assert_close(result.mult, expected)

        unit = get_area_and_mult(
            {"mask": mask, "model_conds": {}}, x, torch.tensor([1.0])
        ).mult
        self.assertEqual(unit.stride(0), 0)
        self.assertEqual(unit.stride(1), 0)

    def test_resident_manual_cast_skips_offload_stream_dependencies(self):
        weight = torch.nn.Parameter(torch.randn((2, 3)))
        bias = torch.nn.Parameter(torch.randn((2,)))
        layer = SimpleNamespace(weight=weight, bias=bias, weight_function=[], bias_function=[])

        with (
            mock.patch.object(operations.memory_management, "device_supports_non_blocking", return_value=False),
            mock.patch.object(operations.memory_management, "cast_to", side_effect=lambda tensor, **kwargs: tensor),
            mock.patch.object(operations.stream, "should_use_stream", return_value=True) as should_use_stream,
            mock.patch.object(operations.memory_management, "get_offload_stream") as get_offload_stream,
            mock.patch.object(operations.memory_management, "sync_stream") as sync_stream,
        ):
            cast_weight, cast_bias, transfer_signal = operations.weights_manual_cast(layer, torch.randn((1, 3)))

        should_use_stream.assert_not_called()
        get_offload_stream.assert_not_called()
        sync_stream.assert_not_called()
        self.assertIs(cast_weight, weight)
        self.assertIs(cast_bias, bias)
        self.assertIsNone(transfer_signal[0])
        self.assertIs(transfer_signal[1], weight)
        self.assertIs(transfer_signal[2], bias)

    def test_resident_stream_worker_uses_null_context(self):
        weight = torch.empty((2, 3))
        bias = torch.empty((2,))

        worker = operations.main_stream_worker(weight, bias, (None, weight, bias))

        self.assertIsInstance(worker, nullcontext)
        with (
            mock.patch.object(operations.memory_management, "current_stream") as current_stream,
            mock.patch.object(operations, "record_region") as record_region,
        ):
            with worker:
                pass

        current_stream.assert_not_called()
        record_region.assert_not_called()

    def test_offload_stream_worker_waits_after_operation(self):
        order = []
        device = torch.device("cuda:1")
        weight = SimpleNamespace(device=device)
        stream = SimpleNamespace(wait_stream=lambda current: order.append(("wait", current)))
        current_stream = object()

        with mock.patch.object(operations.memory_management, "current_stream", return_value=current_stream):
            with operations.main_stream_worker(weight, None, (stream, weight, None)):
                order.append(("operation", None))

        self.assertEqual(order, [("operation", None), ("wait", current_stream)])

    def test_offloaded_manual_cast_keeps_transfer_stream_dependencies(self):
        source_device = torch.device("cpu")
        target_device = torch.device("cuda", 0)
        weight = SimpleNamespace(device=source_device)
        bias = SimpleNamespace(device=source_device)
        layer = SimpleNamespace(weight=weight, bias=bias, weight_function=[], bias_function=[])
        offload_stream = object()
        input_tensor = SimpleNamespace(device=target_device, dtype=torch.float32)

        with (
            mock.patch.object(operations.memory_management, "device_supports_non_blocking", return_value=False),
            mock.patch.object(operations.memory_management, "cast_to", side_effect=lambda tensor, **kwargs: tensor),
            mock.patch.object(operations.stream, "should_use_stream", return_value=True) as should_use_stream,
            mock.patch.object(operations.stream, "stream_context", return_value=lambda _: mock.MagicMock()),
            mock.patch.object(operations.memory_management, "get_offload_stream", return_value=offload_stream) as get_offload_stream,
            mock.patch.object(operations.memory_management, "sync_stream") as sync_stream,
        ):
            _, _, transfer_signal = operations.weights_manual_cast(layer, input_tensor)

        should_use_stream.assert_called_once_with()
        get_offload_stream.assert_called_once_with(target_device)
        sync_stream.assert_called_once_with(target_device, offload_stream)
        self.assertIs(transfer_signal[0], offload_stream)

    def test_single_condition_cfg1_fast_path_matches_general_sampler_path(self):
        class Model:
            def __init__(self):
                self.diffusion_model = SimpleNamespace()

            def apply_model(self, x, timestep, **conditioning):
                self.options = conditioning["transformer_options"]
                return x + conditioning["c_crossattn"].mean()

            def memory_required(self, shape):
                return 1

        model = Model()
        x = torch.randn((1, 2, 2, 2))
        timestep = torch.tensor([1.0])
        condition = {"model_conds": {"c_crossattn": ConditionCrossAttn(torch.randn((1, 3, 4)), cache_key=("prompt",))}}
        cache = {}

        fast = sampling_function_inner(model, x, timestep, None, [condition], 1.0, {}, seed=0, return_full=False, condition_cache=cache)
        expected, _ = calc_cond_uncond_batch(model, [condition], None, x, timestep, {}, condition_cache=cache, materialize_uncond=False)

        torch.testing.assert_close(fast, expected)
        self.assertEqual(model.options["cond_or_uncond"], [0])
        self.assertEqual(model.options["cond_indices"], [0])
        self.assertEqual(model.options["uncond_indices"], [])
        self.assertTrue(torch.equal(model.options["cond_mark"], torch.zeros_like(timestep)))

    def test_single_condition_fast_path_rejects_complex_conditioning(self):
        model = SimpleNamespace(diffusion_model=SimpleNamespace())
        x = torch.zeros((1, 2, 4, 4))
        condition = {"model_conds": {"c_crossattn": ConditionCrossAttn(torch.zeros((1, 3, 4)), cache_key=("prompt",))}}
        self.assertTrue(single_condition_fast_path_eligible(model, [condition], x, {}, {}))

        for extra in (
            {"mask": torch.ones((1, 4, 4))},
            {"area": (2, 4, 0, 0)},
            {"batch_indices": [0]},
            {"control": object()},
            {"strength": 0.5},
        ):
            self.assertFalse(single_condition_fast_path_eligible(model, [dict(condition, **extra)], x, {}, {}))

        self.assertFalse(single_condition_fast_path_eligible(model, [condition, condition], x, {}, {}))
        self.assertFalse(single_condition_fast_path_eligible(model, [condition], x, {"model_function_wrapper": object()}, {}))
        self.assertFalse(single_condition_fast_path_eligible(model, [condition], x, {}, None))

    def test_condition_mark_is_created_on_sigma_device_and_dtype(self):
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        sigmas = torch.tensor([1.0, 0.5], device=device, dtype=torch.float16)

        marks = compute_cond_mark([0, 1], sigmas)

        self.assertEqual(marks.device, sigmas.device)
        self.assertEqual(marks.dtype, sigmas.dtype)
        torch.testing.assert_close(marks, torch.tensor([0, 0, 1, 1], device=device, dtype=torch.float16))

    def test_condition_indices_reuse_fixed_compositions_without_sharing_lists(self):
        _cached_cond_indices.cache_clear()
        sigmas = torch.ones((2,))
        first_cond, first_uncond = compute_cond_indices([0, 1], sigmas)
        first_cond.append(99)

        second_cond, second_uncond = compute_cond_indices([0, 1], sigmas)

        self.assertEqual(second_cond, [0, 1])
        self.assertEqual(second_uncond, [2, 3])
        self.assertEqual(_cached_cond_indices.cache_info().hits, 1)

    def test_disabled_ngms_does_not_read_cuda_sigma_scalar(self):
        class UnreadableSigma:
            def __getitem__(self, _):
                raise AssertionError("disabled NGMS must not read sigma")

        sigma = UnreadableSigma()
        self.assertFalse(should_skip_unconditional(1, sigma, 0.0))
        self.assertFalse(should_skip_unconditional(2, sigma, 1.0))

    def test_enabled_ngms_preserves_step_and_sigma_decisions(self):
        sigma = torch.tensor([0.5])
        self.assertTrue(should_skip_unconditional(1, sigma, 1.0))
        self.assertFalse(should_skip_unconditional(2, sigma, 1.0))
        self.assertTrue(should_skip_unconditional(2, sigma, 1.0, every_step=True))
        self.assertFalse(should_skip_unconditional(1, torch.tensor([0.0]), 1.0))


if __name__ == "__main__":
    unittest.main()
