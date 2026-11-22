import unittest
from contextlib import nullcontext
from unittest import mock

import torch

from backend.multi_gpu.planner import Budget, contiguous_split
from backend.multi_gpu.residency import Residency
from backend.multi_gpu.precision import replica_compute_dtype, replica_fits_memory, replica_memory_requirement, sharded_compute_dtype


class MultiGpuFoundationTests(unittest.TestCase):
    def test_replica_memory_preflight_scales_to_larger_cards(self):
        gib = 1024 ** 3
        state = {"weight": torch.empty((7 * gib,), dtype=torch.uint8, device="meta"), "metadata": "ignored"}
        required = replica_memory_requirement(state, int(1.2 * gib))
        self.assertGreater(required, 8 * gib)
        self.assertLess(required, 12 * gib)
        self.assertLess(required, 24 * gib)
        self.assertFalse(replica_fits_memory(state, 8 * gib, int(1.2 * gib)))
        self.assertTrue(replica_fits_memory(state, 12 * gib, int(1.2 * gib)))
        self.assertTrue(replica_fits_memory(state, 24 * gib, int(1.2 * gib)))

    def test_replica_precision_respects_device_capabilities(self):
        devices = (torch.device("cuda:0"), torch.device("cuda:1"))
        supported = [torch.float16, torch.bfloat16, torch.float32]
        with mock.patch.object(torch.cuda, "device", side_effect=lambda _: nullcontext()), mock.patch.object(torch.cuda, "is_bf16_supported", side_effect=[True, False]):
            self.assertEqual(replica_compute_dtype(torch.bfloat16, supported, devices), torch.float16)
        with mock.patch.object(torch.cuda, "device", side_effect=lambda _: nullcontext()), mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            self.assertEqual(replica_compute_dtype(torch.bfloat16, supported, devices), torch.bfloat16)
        with mock.patch.object(torch.cuda, "device", side_effect=lambda _: nullcontext()), mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            self.assertEqual(replica_compute_dtype(torch.bfloat16, [torch.bfloat16, torch.float32], devices), torch.float32)
        self.assertEqual(replica_compute_dtype(torch.float32, supported, devices), torch.float32)
        self.assertEqual(replica_compute_dtype(torch.float16, supported, devices), torch.float16)

    def test_sharded_precision_preserves_fp32_and_model_capabilities(self):
        self.assertEqual(sharded_compute_dtype(torch.float32, [torch.float16, torch.float32]), torch.float32)
        self.assertEqual(sharded_compute_dtype(torch.bfloat16, [torch.bfloat16, torch.float32]), torch.float32)
        self.assertEqual(sharded_compute_dtype(torch.bfloat16, [torch.float16, torch.bfloat16]), torch.float16)
    def test_split_respects_reservations_and_fixed_weights(self):
        self.assertEqual(contiguous_split([2, 3, 4], Budget(9, 2), Budget(9, 1), first_fixed=1, second_fixed=2), 2)
        with self.assertRaisesRegex(ValueError, "No contiguous split"):
            contiguous_split([4, 4], Budget(5, 2), Budget(6, 1))

    def test_primary_preference_respects_both_memory_budgets(self):
        self.assertEqual(contiguous_split([2, 2, 2, 2], Budget(7, 0), Budget(7, 0), policy="prefer_primary"), 3)
        self.assertEqual(contiguous_split([2, 2, 2, 2], Budget(5, 0), Budget(7, 0), policy="prefer_primary"), 2)
        with self.assertRaisesRegex(ValueError, "Unknown multi-GPU split policy"):
            contiguous_split([2, 2], Budget(10, 0), Budget(10, 0), policy="unknown")

    def test_residency_detects_missing_and_wrong_device(self):
        model = torch.nn.Linear(2, 2)
        Residency({"weight": "cpu", "bias": "cpu"}).validate(model)
        with self.assertRaisesRegex(ValueError, "unexpected=.*bias"):
            Residency({"weight": "cpu"}).validate(model)
        with self.assertRaisesRegex(ValueError, "misplaced=.*weight"):
            Residency({"weight": "cuda:0", "bias": "cpu"}).validate(model)

    def test_record_is_copied_and_requires_indexed_cuda(self):
        devices = {"weight": "cpu"}
        residency = Residency(devices)
        devices["weight"] = "cuda:0"
        self.assertEqual(residency.tensors["weight"], torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "explicit indices"):
            Residency({"weight": "cuda"})

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "two CUDA devices required")
    def test_live_residency_on_two_devices(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        model[0].to("cuda:0")
        model[1].to("cuda:1")
        residency = Residency({"0.weight": "cuda:0", "0.bias": "cuda:0", "1.weight": "cuda:1", "1.bias": "cuda:1"})
        residency.validate(model)
        value = model[1](model[0](torch.ones(1, 2, device="cuda:0")).to("cuda:1"))
        self.assertEqual(value.device, torch.device("cuda:1"))


if __name__ == "__main__":
    unittest.main()
