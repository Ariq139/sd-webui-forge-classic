from types import SimpleNamespace

import pytest
import torch

from backend.nn.llm.llama import Llama2_, Qwen3_4BConfig


class Layer(torch.nn.Module):
    def __init__(self, i, calls):
        super().__init__()
        self.i, self.calls = i, calls

    def forward(self, x, **kwargs):
        self.calls.append(self.i)
        return x + self.i + 1, None


@pytest.mark.parametrize("requested,expected_calls", [([1, 3, 5], 5), (-2, 7)])
def test_qwen_intermediate_only_returns_same_selected_states(requested, expected_calls):
    config = Qwen3_4BConfig()
    # Construct a small model without loading actual encoder weights.
    model = Llama2_.__new__(Llama2_)
    torch.nn.Module.__init__(model)
    model.config = config
    model.normalize_in = False
    model.norm = torch.nn.Identity()
    calls = []
    model.layers = torch.nn.ModuleList([Layer(i, calls) for i in range(8)])
    model.eval()
    x = torch.zeros(1, 2, 128)
    with torch.no_grad():
        full, expected = model(None, embeds=x, intermediate_output=requested, final_layer_norm_intermediate=False)
        assert len(calls) == 8
        calls.clear()
        final, actual = model(None, embeds=x, intermediate_output=requested, final_layer_norm_intermediate=False, intermediate_only=True)
        assert final is None and len(calls) == expected_calls
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
