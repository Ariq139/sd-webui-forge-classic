import unittest
from contextlib import nullcontext
from unittest import mock

import torch
from gguf import GGMLQuantizationType

from backend import operations, operations_gguf
from backend.operations_gguf import ParameterGGUF


class FakeCudaTensor:
    device = torch.device("cuda:0")
    dtype = torch.float16
    shape = (1, 4)

    def stride(self):
        return (4, 1)


def make_gguf_weight(*, dtype=torch.uint8, baked=False):
    weight = ParameterGGUF(
        torch.arange(8, dtype=dtype),
        tensor_type=GGMLQuantizationType.Q4_0,
        tensor_shape=(2, 4),
    )
    weight.baked = baked
    return weight


class GGUFLinearFastPathTests(unittest.TestCase):
    def setUp(self):
        operations_gguf._CUDA_LINEAR_QTYPE_SUPPORT.clear()
        operations_gguf._CUDA_LINEAR_FAILURES.clear()

    def test_kernel_only_accepts_original_contiguous_packed_bytes(self):
        supported_module = mock.Mock()
        supported_module.may_support_linear_qtype_name.return_value = True
        fake_input = FakeCudaTensor()

        with mock.patch.object(operations_gguf, "_llamacpp_cuda_module", return_value=supported_module):
            self.assertTrue(operations_gguf.can_use_llamacpp_cuda_linear(make_gguf_weight(), fake_input))
            self.assertFalse(operations_gguf.can_use_llamacpp_cuda_linear(make_gguf_weight(baked=True), fake_input))
            self.assertFalse(operations_gguf.can_use_llamacpp_cuda_linear(make_gguf_weight(dtype=torch.float16), fake_input))

        supported_module.may_support_linear_qtype_name.assert_called_once_with("Q4_0")

    def test_known_unsupported_shape_is_not_retried(self):
        module = mock.Mock()
        module.may_support_linear_qtype_name.return_value = True
        module.linear.side_effect = RuntimeError("unsupported input shape")
        weight = make_gguf_weight()
        fake_input = FakeCudaTensor()

        with mock.patch.object(operations_gguf, "_llamacpp_cuda_module", return_value=module):
            self.assertIsNone(operations_gguf.try_llamacpp_cuda_linear(weight, fake_input, None))
            self.assertIsNone(operations_gguf.try_llamacpp_cuda_linear(weight, fake_input, None))

        module.linear.assert_called_once()

    def test_out_of_memory_from_kernel_is_propagated(self):
        module = mock.Mock()
        module.may_support_linear_qtype_name.return_value = True
        module.linear.side_effect = RuntimeError("out of memory")
        with (
            mock.patch.object(operations_gguf, "_llamacpp_cuda_module", return_value=module),
            mock.patch.object(operations_gguf.memory_management, "is_oom", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                operations_gguf.try_llamacpp_cuda_linear(make_gguf_weight(), FakeCudaTensor(), None)

    def test_qtype_support_is_checked_and_cached_before_weight_preparation(self):
        module = mock.Mock()
        module.may_support_linear_qtype_name.return_value = False
        with mock.patch.object(operations_gguf, "_llamacpp_cuda_module", return_value=module):
            self.assertFalse(operations_gguf.can_use_llamacpp_cuda_linear(make_gguf_weight(), FakeCudaTensor()))
            self.assertFalse(operations_gguf.can_use_llamacpp_cuda_linear(make_gguf_weight(), FakeCudaTensor()))
        module.may_support_linear_qtype_name.assert_called_once_with("Q4_0")

    def test_packed_weight_cast_preserves_bytes_and_uses_temporary_storage(self):
        layer = operations.ForgeOperationsGGUF.Linear()
        layer.weight = make_gguf_weight()
        layer.bias = None
        layer.weight_function = []
        layer.bias_function = []
        inputs = torch.ones((1, 4), dtype=torch.float16)

        copied, _, _ = operations.weights_manual_cast(
            layer, inputs, weight_fn=lambda value: value, skip_weight_dtype=True, skip_bias_dtype=True
        )

        self.assertEqual(copied.dtype, torch.uint8)
        self.assertIsNot(copied, layer.weight)
        self.assertNotEqual(copied.data.data_ptr(), layer.weight.data.data_ptr())
        torch.testing.assert_close(copied.data, layer.weight.data)

    def test_fast_path_fallback_reuses_packed_copy_without_second_cast(self):
        layer = operations.ForgeOperationsGGUF.Linear()
        layer.weight = make_gguf_weight()
        layer.bias = None
        layer.weight_function = []
        layer.bias_function = []
        inputs = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
        prepared = make_gguf_weight()
        dense_weight = torch.tensor([[0.25, 0.5, -0.5, 0.75], [1.0, 0.0, 0.25, -1.0]])

        with (
            mock.patch.object(operations, "can_use_llamacpp_cuda_linear", return_value=True),
            mock.patch.object(operations, "weights_manual_cast", return_value=(prepared, None, None)) as cast,
            mock.patch.object(operations, "try_llamacpp_cuda_linear", return_value=None),
            mock.patch.object(operations, "dequantize_tensor", return_value=dense_weight) as dequantize,
            mock.patch.object(operations.args, "enable_triton_backend", False, create=True),
        ):
            output = layer(inputs)

        cast.assert_called_once()
        self.assertTrue(cast.call_args.kwargs["skip_weight_dtype"])
        dequantize.assert_called_once_with(prepared)
        torch.testing.assert_close(output, torch.nn.functional.linear(inputs, dense_weight))


class ProfilerRegionTests(unittest.TestCase):
    def tearDown(self):
        from backend.performance import set_profiler_enabled

        set_profiler_enabled(False)

    def test_region_is_noop_when_profiling_is_disabled(self):
        with mock.patch("torch.profiler.record_function") as record:
            with operations_gguf.record_region("forge::test"):
                pass
        record.assert_not_called()

    def test_region_is_recorded_when_profiling_is_enabled(self):
        from backend.performance import set_profiler_enabled

        set_profiler_enabled(True)
        with mock.patch("torch.profiler.record_function", return_value=nullcontext()) as record:
            with operations_gguf.record_region("forge::test"):
                pass
        record.assert_called_once_with("forge::test")


if __name__ == "__main__":
    unittest.main()
