import pytest
import torch
from backend.segmented_int8 import segmented_reference


def native_reference(x):
    xf = x.float()
    maximum = torch.nan_to_num(xf.abs(), nan=0.0, posinf=torch.finfo(x.dtype).max).amax(-1, keepdim=True)
    scale = (maximum * (1.0 / 127.0)).clamp_min(1e-30)
    q = (xf / scale.to(x.dtype).float()).to(x.dtype).float().round()
    return torch.nan_to_num(q, nan=-128.0, posinf=127.0, neginf=-128.0).clamp(-128, 127).to(torch.int8), scale


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_segmented_reference_matches_concatenated_quantization(dtype):
    torch.manual_seed(39)
    a = torch.randn(5, 32, dtype=dtype)
    b = torch.randn(5, 48, dtype=dtype)
    a[0].zero_(); b[0].zero_()
    a[1, 0] = float('nan'); b[1, 1] = float('inf')
    a[2, 2] = -float('inf')
    q, s = segmented_reference(a, b)
    expected_q, expected_s = native_reference(torch.cat((a, b), -1))
    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(s, expected_s, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_segmented_kernel_matches_pinned_cuda(dtype):
    pytest.importorskip('triton')
    from backend.native_int8_rows import _cuda_backend
    cuda = _cuda_backend()
    if cuda is None or torch.cuda.get_device_capability() != (8, 6):
        pytest.skip('requires pinned Comfy-Kitchen and SM86')
    from backend.segmented_int8_kernel import launch
    a = torch.randn(7, 256, dtype=dtype, device='cuda')
    b = torch.randn(7, 384, dtype=dtype, device='cuda')
    a[0].zero_(); b[0].zero_()
    a[1, 0] = float('nan'); b[1, 1] = float('inf')
    result = launch(a, b)
    assert result is not None
    expected = cuda.quantize_int8_rowwise(torch.cat((a, b), -1))
    for actual, ref in zip(result, expected):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
