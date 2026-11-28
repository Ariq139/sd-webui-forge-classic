import pytest
import torch

from backend.native_gqa import compatible_gqa, try_flash_gqa


def test_gqa_shape_and_extension_guards():
    q = torch.zeros(1, 8, 16, 32, dtype=torch.float16)
    k = torch.zeros(1, 2, 16, 32, dtype=torch.float16)
    assert compatible_gqa(q, k, k, None, {"sampling_sigmas": torch.ones(3)})
    assert not compatible_gqa(q, k, k, torch.ones(16, 16), {})
    assert not compatible_gqa(q, k, k, None, {"optimized_attention_override": object()})
    assert not compatible_gqa(q[:, :7], k, k, None, {})
    assert not compatible_gqa(q, k, k.float(), None, {})
    assert not compatible_gqa(q, q, q, None, {})
    assert try_flash_gqa(q, k, k, None, {}) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ratio", [2, 4])
def test_flash_gqa_matches_expanded_heads(ratio):
    from torch.nn import functional as F
    q = torch.randn(1, 4 * ratio, 64, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 4, 64, 64, dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    with torch.no_grad():
        actual = try_flash_gqa(q, k, v, None, {})
        if actual is None:
            pytest.skip("compact Flash GQA unsupported")
        expected = F.scaled_dot_product_attention(q, k.repeat_interleave(ratio, 1), v.repeat_interleave(ratio, 1))
        expected = expected.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)
