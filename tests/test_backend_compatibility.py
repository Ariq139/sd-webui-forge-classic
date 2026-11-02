import ast
from contextlib import nullcontext
import json
from pathlib import Path
import unittest
from unittest import mock

import torch

from backend import memory_management, operations
from backend.operations_mixed_precision import mixed_precision_ops


def awq_state(rows):
    metadata = {"format": "awq_w4a16", "group_size": 32}
    return {
        "weight": torch.full((rows, 32), -120, dtype=torch.int8),
        "weight_scale": torch.full((2, rows), 0.25),
        "weight_zero": torch.full((2, rows), 3.0),
        "comfy_quant": torch.tensor(list(json.dumps(metadata).encode()), dtype=torch.uint8),
    }


class BackendCompatibilityTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(operations, "SDPA_BACKEND_PRIORITY"), "prioritized CUDA SDPA wrapper unavailable")
    def test_small_and_large_attention_keep_dtype_and_masked_gqa_guards(self):
        for length in (2, 4096):
            with self.subTest(length=length):
                q = torch.empty((1, 4, length, 8), dtype=torch.float16)
                k = torch.empty((1, 2, length, 8), dtype=torch.float32)
                v = torch.empty((1, 2, length, 8), dtype=torch.float16)
                mask = torch.ones(1, dtype=torch.bool)
                with (
                    mock.patch.object(memory_management, "attention_dtype_alignment_enabled", return_value=True),
                    mock.patch.object(memory_management, "is_nvidia", return_value=False),
                    mock.patch.object(operations, "sdpa_kernel", return_value=nullcontext()) as kernel,
                    mock.patch.object(torch.nn.functional, "scaled_dot_product_attention") as attention,
                ):
                    operations.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
                sent_q, sent_k, sent_v = attention.call_args.args
                self.assertEqual((sent_q.dtype, sent_k.dtype, sent_v.dtype), (torch.float16,) * 3)
                self.assertEqual(sent_k.shape[1], 4)
                self.assertEqual(sent_v.shape[1], 4)
                self.assertFalse(attention.call_args.kwargs["enable_gqa"])
                self.assertIs(attention.call_args.kwargs["attn_mask"], mask)
                kernel.assert_called_once_with(operations.SDPA_BACKEND_PRIORITY, set_priority=True)

    def test_awq_linear_state_dict_round_trip_preserves_groups_and_zeros(self):
        ops = mixed_precision_ops(compute_dtype=torch.float32)
        layer = ops.Linear(64, 4, bias=False, device="cpu")
        layer.load_state_dict(awq_state(4))
        saved = layer.state_dict()
        self.assertIn("weight_zero", saved)
        self.assertNotIn("weight_zeros", saved)
        metadata = json.loads(bytes(saved["comfy_quant"].tolist()))
        self.assertEqual(metadata["group_size"], 32)
        restored = ops.Linear(64, 4, bias=False, device="cpu")
        restored.load_state_dict(saved)
        torch.testing.assert_close(restored.weight.dequantize(), torch.full((4, 64), 3.0))

    def test_awq_embedding_row_lookup(self):
        ops = mixed_precision_ops(compute_dtype=torch.float32)
        embedding = ops.Embedding(3, 64)
        embedding.load_state_dict(awq_state(3))
        with mock.patch.object(memory_management, "w4a16_embedding_rowwise_enabled", return_value=True):
            with torch.inference_mode():
                result = embedding(torch.tensor([[2, 0, 2]]))
        torch.testing.assert_close(result, torch.full((1, 3, 64), 3.0))

    def test_preset_architectures_are_distinct(self):
        path = Path(__file__).resolve().parents[1] / "modules_forge" / "presets.py"
        tree = ast.parse(path.read_text())
        definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PresetArch")
        from enum import Enum

        namespace = {"Enum": Enum}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
        presets = namespace["PresetArch"]
        self.assertIsNot(presets.ltx2, presets.qwen21)
        self.assertEqual(presets.ltx2.name, "ltx2")
        self.assertIn("ltx2", presets.choices())
        self.assertIn("qwen21", presets.choices())
        self.assertEqual(len(presets.__members__), len(list(presets)))
