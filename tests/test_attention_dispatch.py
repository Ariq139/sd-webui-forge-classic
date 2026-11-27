import unittest
from unittest import mock

import torch

from backend.attention import _cuda_compute_capability, _should_use_pytorch_for_ck_int8_attention


class FakeCudaTensor:
    def __init__(self, shape, dtype=torch.float16, device="cuda:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)
        self.ndim = len(shape)
        self.is_cuda = self.device.type == "cuda"


class TestCKAttentionDispatch(unittest.TestCase):
    def setUp(self):
        _cuda_compute_capability.cache_clear()

    def tearDown(self):
        _cuda_compute_capability.cache_clear()

    @mock.patch("torch.backends.cuda.flash_sdp_enabled", return_value=True)
    @mock.patch("torch.cuda.get_device_capability", return_value=(8, 6))
    def test_uses_pytorch_for_small_unmasked_cuda_half_attention(self, _capability, _flash_enabled):
        q = FakeCudaTensor((1, 512, 1024))
        k = FakeCudaTensor((1, 77, 1024))
        v = FakeCudaTensor((1, 77, 1024))

        self.assertTrue(_should_use_pytorch_for_ck_int8_attention(q, k, v, 16, None, False, {}))

    @mock.patch("torch.backends.cuda.flash_sdp_enabled", return_value=True)
    @mock.patch("torch.cuda.get_device_capability", return_value=(8, 6))
    def test_keeps_comfy_kitchen_for_large_attention(self, _capability, _flash_enabled):
        q = FakeCudaTensor((1, 2048, 1024))
        k = FakeCudaTensor((1, 2048, 1024))
        v = FakeCudaTensor((1, 2048, 1024))

        self.assertFalse(_should_use_pytorch_for_ck_int8_attention(q, k, v, 16, None, False, {}))

    @mock.patch("torch.backends.cuda.flash_sdp_enabled", return_value=True)
    @mock.patch("torch.cuda.get_device_capability", return_value=(8, 6))
    def test_keeps_comfy_kitchen_when_score_memory_cap_is_exceeded(self, _capability, _flash_enabled):
        q = FakeCudaTensor((1, 16384, 1024))
        k = FakeCudaTensor((1, 256, 1024))
        v = FakeCudaTensor((1, 256, 1024))

        self.assertFalse(_should_use_pytorch_for_ck_int8_attention(q, k, v, 16, None, False, {}))

    @mock.patch("torch.backends.cuda.flash_sdp_enabled", return_value=True)
    @mock.patch("torch.cuda.get_device_capability", return_value=(7, 5))
    def test_keeps_comfy_kitchen_when_flash_attention_is_unavailable(self, _capability, _flash_enabled):
        q = FakeCudaTensor((1, 512, 1024))
        k = FakeCudaTensor((1, 77, 1024))
        v = FakeCudaTensor((1, 77, 1024))

        self.assertFalse(_should_use_pytorch_for_ck_int8_attention(q, k, v, 16, None, False, {}))

    @mock.patch("torch.backends.cuda.flash_sdp_enabled", return_value=False)
    @mock.patch("torch.cuda.get_device_capability", return_value=(8, 6))
    def test_keeps_comfy_kitchen_when_flash_sdp_is_disabled(self, _capability, _flash_disabled):
        q = FakeCudaTensor((1, 512, 1024))
        k = FakeCudaTensor((1, 77, 1024))
        v = FakeCudaTensor((1, 77, 1024))

        self.assertFalse(_should_use_pytorch_for_ck_int8_attention(q, k, v, 16, None, False, {}))

    @mock.patch("torch.backends.cuda.flash_sdp_enabled", return_value=True)
    @mock.patch("torch.cuda.get_device_capability", return_value=(8, 6))
    def test_supports_pre_reshaped_inputs(self, _capability, _flash_enabled):
        q = FakeCudaTensor((1, 16, 512, 64))
        k = FakeCudaTensor((1, 16, 77, 64))
        v = FakeCudaTensor((1, 16, 77, 64))

        self.assertTrue(_should_use_pytorch_for_ck_int8_attention(q, k, v, 16, None, True, {}))

    @mock.patch("torch.backends.cuda.flash_sdp_enabled", return_value=True)
    @mock.patch("torch.cuda.get_device_capability", return_value=(8, 6))
    def test_rejects_head_dimensions_outside_flash_attention_limits(self, _capability, _flash_enabled):
        q = FakeCudaTensor((1, 512, 8192))
        k = FakeCudaTensor((1, 77, 8192))
        v = FakeCudaTensor((1, 77, 8192))

        self.assertFalse(_should_use_pytorch_for_ck_int8_attention(q, k, v, 16, None, False, {}))

    @mock.patch("torch.cuda.get_device_capability", return_value=(8, 6))
    def test_caches_device_capability_lookup(self, get_capability):
        device = torch.device("cuda:0")

        self.assertEqual(_cuda_compute_capability(device), (8, 6))
        self.assertEqual(_cuda_compute_capability(device), (8, 6))

        get_capability.assert_called_once_with(device)

    @mock.patch("torch.backends.cuda.flash_sdp_enabled", return_value=True)
    @mock.patch("torch.cuda.get_device_capability", return_value=(8, 6))
    def test_rejects_masked_gqa_unsupported_dtype_and_non_cuda_inputs(self, _capability, _flash_enabled):
        q = FakeCudaTensor((1, 512, 1024))
        k = FakeCudaTensor((1, 77, 1024))
        v = FakeCudaTensor((1, 77, 1024))

        self.assertFalse(_should_use_pytorch_for_ck_int8_attention(q, k, v, 16, torch.ones(1), False, {}))
        self.assertFalse(_should_use_pytorch_for_ck_int8_attention(q, k, v, 16, None, False, {"enable_gqa": True}))
        float32_q = FakeCudaTensor((1, 512, 1024), torch.float32)
        self.assertFalse(_should_use_pytorch_for_ck_int8_attention(float32_q, k, v, 16, None, False, {}))

        cpu_q = FakeCudaTensor((1, 512, 1024), device="cpu")
        cpu_k = FakeCudaTensor((1, 77, 1024), device="cpu")
        cpu_v = FakeCudaTensor((1, 77, 1024), device="cpu")
        self.assertFalse(_should_use_pytorch_for_ck_int8_attention(cpu_q, cpu_k, cpu_v, 16, None, False, {}))


if __name__ == "__main__":
    unittest.main()
