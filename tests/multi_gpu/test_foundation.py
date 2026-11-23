import unittest
from contextlib import nullcontext
from unittest import mock

import torch

from backend.multi_gpu.planner import Budget, contiguous_split
from backend.multi_gpu.residency import Residency
from backend.multi_gpu.precision import automatic_multi_gpu_mode, replica_compute_dtype, replica_fits_memory, replica_memory_requirement, sharded_compute_dtype


class MultiGpuFoundationTests(unittest.TestCase):
    def test_multi_gpu_option_enables_automatic_paths(self):
        from backend.args import configure_multi_gpu, parser

        automatic = configure_multi_gpu(parser.parse_args(["--multi-gpu"]))
        self.assertEqual(automatic.multi_gpu_anima, 1)
        self.assertTrue(automatic.multi_gpu_auto)
        self.assertEqual(automatic.parallel_cfg, 1)
        self.assertEqual(automatic.parallel_batch, 1)
        self.assertTrue(automatic.parallel_batch_adaptive)
        self.assertTrue(automatic.parallel_cfg_keep_resident)
        selected = configure_multi_gpu(parser.parse_args(["--multi-gpu", "3"]))
        self.assertEqual((selected.multi_gpu_anima, selected.parallel_cfg, selected.parallel_batch), (3, 3, 3))
        sharded = configure_multi_gpu(parser.parse_args(["--multi-gpu", "--multi-gpu-sharding"]))
        self.assertFalse(sharded.multi_gpu_auto)
        self.assertIsNone(sharded.parallel_cfg)
        self.assertIsNone(sharded.parallel_batch)
        off = configure_multi_gpu(parser.parse_args([]))
        self.assertIsNone(off.multi_gpu_anima)
        self.assertFalse(off.multi_gpu_auto)

    def test_automatic_mode_uses_each_device_and_host_memory(self):
        gib = 1024 ** 3
        state = {"weight": torch.empty((7 * gib,), dtype=torch.uint8, device="meta")}
        def choose(gpus, host=32 * gib, replicas=True, sharding=True):
            return automatic_multi_gpu_mode(state, gpus, host, int(1.2 * gib), replicas_supported=replicas, sharding_supported=sharding)
        self.assertEqual(choose([8 * gib, 8 * gib]), "sharding")
        self.assertEqual(choose([12 * gib, 24 * gib]), "replicas")
        self.assertEqual(choose([24 * gib, 8 * gib]), "sharding")
        self.assertEqual(choose([12 * gib, 12 * gib], host=7 * gib), "sharding")
        self.assertEqual(choose([24 * gib, 24 * gib], replicas=False), "sharding")
        self.assertEqual(choose([8 * gib, 8 * gib], sharding=False), "single")

    def test_automatic_mode_uses_checkpoint_size_not_model_family(self):
        gib = 1024 ** 3
        for family, checkpoint_size, expected in (
            ("large Anima", 7 * gib, "sharding"),
            ("small Krea", 2 * gib, "replicas"),
            ("small Anima", 2 * gib, "replicas"),
            ("large Krea", 7 * gib, "sharding"),
        ):
            with self.subTest(model=family):
                state = {"weight": torch.empty((checkpoint_size,), dtype=torch.uint8, device="meta")}
                self.assertEqual(
                    automatic_multi_gpu_mode(state, [8 * gib, 8 * gib], 32 * gib, int(1.2 * gib),
                                             replicas_supported=True, sharding_supported=True),
                    expected,
                )

    def test_automatic_mode_checks_memory_boundaries_and_unknown_sizes(self):
        gib = 1024 ** 3
        reserve = gib
        for model_size in (gib, 3 * gib, 7 * gib, 15 * gib):
            state = {"weight": torch.empty(model_size, dtype=torch.uint8, device="meta")}
            required = model_size + reserve
            host_required = model_size + 512 * 1024 ** 2
            with self.subTest(size=model_size):
                for gpus, host, expected in (
                    ([required, required], host_required, "replicas"),
                    ([required - 1, required], host_required, "sharding"),
                    ([required, required - 1], host_required, "sharding"),
                    ([required, required], host_required - 1, "sharding"),
                ):
                    self.assertEqual(automatic_multi_gpu_mode(state, gpus, host, reserve, replicas_supported=True, sharding_supported=True), expected)
        for state, gpus, host, reserve in (
            ({}, [8 * gib, 8 * gib], 32 * gib, gib),
            ({"weight": torch.ones(1)}, [8 * gib], 32 * gib, gib),
            ({"weight": torch.ones(1)}, [8 * gib, -1], 32 * gib, gib),
            ({"weight": torch.ones(1)}, [8 * gib, 8 * gib], -1, gib),
            ({"weight": torch.ones(1)}, [8 * gib, 8 * gib], 32 * gib, -1),
        ):
            self.assertEqual(automatic_multi_gpu_mode(state, gpus, host, reserve, replicas_supported=True, sharding_supported=True), "single")
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
