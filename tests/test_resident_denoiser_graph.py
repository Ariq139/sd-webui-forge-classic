import pytest
import torch
from backend.resident_denoiser_graph import ReplaySlot, input_signature, static_inputs, tensor_stamp, update_static


def test_static_signatures_update_values_and_preserve_ownership():
    values = {'x': torch.ones(2, 3), 'option:cond_indices': [0], 'y': None}
    original = input_signature(values)
    static = static_inputs(values)
    values['x'].add_(2)
    assert input_signature(values) == original
    update_static(static, values)
    torch.testing.assert_close(static['x'], values['x'])
    assert static['x'].data_ptr() != values['x'].data_ptr()
    values['option:cond_indices'].append(1)
    assert static['option:cond_indices'] == [0] and input_signature(values) != original
    assert input_signature({'x': torch.ones(2, 3).T}) is None
    assert input_signature({'callback': object()}) is None


def test_weight_stamp_invalidates_mutation_and_rebinding():
    weight = torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
    a = tensor_stamp(weight, storage=True)
    weight.add_(1)
    assert tensor_stamp(weight, storage=True) != a
    a = tensor_stamp(weight, storage=True)
    weight.data = torch.zeros_like(weight)
    assert tensor_stamp(weight, storage=True) != a
    with torch.inference_mode():
        assert tensor_stamp(torch.ones(2), storage=True) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_slot_replays_changed_inputs_and_owns_output():
    device = torch.device('cuda', torch.cuda.current_device())
    slot = ReplaySlot(512 * 1024**2)
    if torch.cuda.mem_get_info(device)[0] < 1024**3:
        pytest.skip('insufficient graph headroom')
    x = torch.ones(2, 16, device=device)
    def invoke(values):
        return values['x'] * 2 + 1
    try:
        with torch.no_grad():
            first = slot.run('shape', device, {'x': x}, invoke)
            if slot.disabled:
                pytest.skip('conservative peak budget')
            x.fill_(2)
            second = slot.run('shape', device, {'x': x}, invoke)
            assert slot.graph is not None
            x.fill_(3)
            third = slot.run('shape', device, {'x': x}, invoke)
            torch.testing.assert_close(first, torch.full_like(x, 3))
            torch.testing.assert_close(second, torch.full_like(x, 5))
            torch.testing.assert_close(third, torch.full_like(x, 7))
    finally:
        slot.clear()
