import unittest

import torch

from backend.multi_gpu.planner import Budget, contiguous_split
from backend.multi_gpu.residency import Residency
from backend.multi_gpu.precision import sharded_compute_dtype


class MultiGpuFoundationTests(unittest.TestCase):
    def test_sharded_precision_preserves_fp32_and_model_capabilities(self):
        self.assertEqual(sharded_compute_dtype(torch.float32, [torch.float16, torch.float32]), torch.float32)
        self.assertEqual(sharded_compute_dtype(torch.bfloat16, [torch.bfloat16, torch.float32]), torch.float32)
        self.assertEqual(sharded_compute_dtype(torch.bfloat16, [torch.float16, torch.bfloat16]), torch.float16)
    def test_split_respects_reservations_and_fixed_weights(self):
        self.assertEqual(contiguous_split([2, 3, 4], Budget(9, 2), Budget(9, 1), first_fixed=1, second_fixed=2), 2)
        with self.assertRaisesRegex(ValueError, "No contiguous split"):
            contiguous_split([4, 4], Budget(5, 2), Budget(6, 1))

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
