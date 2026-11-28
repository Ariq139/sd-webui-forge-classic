"""Regression coverage against the real Forge model and sampler call sites."""
from types import MethodType, SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from backend import operations
from backend.diffusion_engine.base import ForgeDiffusionEngine
from backend.nn.anima import SelfCrossAttention, VideoRopePosition3DEmb
from backend.nn.flux import EmbedND, IntegratedFluxTransformer2DModel
from backend.nn.lumina import NextDiT
from backend.sampling import sampling_function as sf
from backend.sampling.condition import compile_conditions
from backend.sampling.invariant_cache import (
    invariant_condition, invariant_revision, sampling_invariant_scope,
)


def test_anima_position_geometry_and_cross_kv_reuse():
    rope = VideoRopePosition3DEmb(24, 1.0, 1.0, 1.0).eval()
    x = torch.zeros(1, 2, 3, 4, 24)
    attention = SelfCrossAttention(8, 6, n_heads=2, head_dim=4).eval()
    context = torch.randn(1, 5, 6)
    first = torch.randn(1, 3, 8)
    second = torch.randn_like(first)
    expected = rope(x, x.device)
    with torch.no_grad(), sampling_invariant_scope(1024 * 1024), invariant_revision(1), invariant_condition(1):
        with mock.patch.object(rope, "_compute_positions", wraps=rope._compute_positions) as compute:
            a = rope(x, x.device, {})
            b = rope(torch.ones_like(x), x.device, {})
            assert a is b and compute.call_count == 1
            torch.testing.assert_close(a, expected, rtol=0, atol=0)
            rope(torch.zeros(1, 2, 4, 3, 24), x.device, {})
            assert compute.call_count == 2
        with mock.patch.object(attention, "compute_attention", side_effect=lambda q, k, v, **kwargs: (q, k, v)):
            q1, k1, v1 = attention(first, context, transformer_options={})
            q2, k2, v2 = attention(second, context.clone(), transformer_options={})
            assert k1 is k2 and v1 is v2
            assert not torch.equal(q1, q2)
            q3, k3, v3 = attention(first, context, transformer_options={"patches": {}})
            for cached, direct in zip((q1, k1, v1), (q3, k3, v3)):
                torch.testing.assert_close(cached, direct, rtol=0, atol=0)


class _MutatingFluxBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, img, txt, **kwargs):
        self.seen.append(txt.clone())
        txt.add_(10)
        return img, txt


class _IdentityFinal(torch.nn.Module):
    def forward(self, x, *args):
        return x


def test_flux_cached_text_projection_is_not_mutated_by_double_blocks():
    model = IntegratedFluxTransformer2DModel.__new__(IntegratedFluxTransformer2DModel)
    torch.nn.Module.__init__(model)
    model.img_in = torch.nn.Identity()
    model.time_in = torch.nn.Linear(256, 8)
    model.guidance_embed = False
    model.txt_norm = None
    model.txt_in = torch.nn.Linear(8, 8)
    model.global_modulation = False
    block = _MutatingFluxBlock()
    model.double_blocks = torch.nn.ModuleList([block])
    model.single_blocks = torch.nn.ModuleList()
    model.final_layer = _IdentityFinal()
    model.eval()
    x = torch.zeros(1, 3, 8)
    context = torch.randn(1, 2, 8)
    with torch.no_grad(), sampling_invariant_scope(1024 * 1024) as cache, invariant_revision(1), invariant_condition(1):
        model.forward_orig(x, None, context, None, torch.ones(1), None, transformer_options={})
        projected = next(value for key, value in cache.entries.items() if key[3] == "flux_text_projection")
        model.forward_orig(x, None, context.clone(), None, torch.zeros(1), None, transformer_options={})
        assert next(value for key, value in cache.entries.items() if key[3] == "flux_text_projection") is projected
    torch.testing.assert_close(block.seen[0], block.seen[1], rtol=0, atol=0)


class _Refiner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, *args, **kwargs):
        self.calls += 1
        return x + 1


def test_zimage_context_refiner_reuses_text_but_not_noise_or_mask():
    model = NextDiT.__new__(NextDiT)
    torch.nn.Module.__init__(model)
    model.patch_size = 2
    model.pad_tokens_multiple = None
    model.x_embedder = torch.nn.Linear(16, 8)
    model.rope_embedder = EmbedND(dim=8, theta=10000, axes_dim=[2, 2, 4])
    context_refiner, noise_refiner = _Refiner(), _Refiner()
    model.context_refiner = torch.nn.ModuleList([context_refiner])
    model.noise_refiner = torch.nn.ModuleList([noise_refiner])
    model.eval()
    x = torch.randn(1, 4, 4, 4)
    text = torch.randn(1, 3, 8)
    with torch.no_grad(), sampling_invariant_scope(1024 * 1024), invariant_revision(1), invariant_condition(1):
        a = model.patchify_and_embed(x, text, None, torch.ones(1), None, {})
        b = model.patchify_and_embed(x + 1, text.clone(), None, torch.zeros(1), None, {})
        assert context_refiner.calls == 1 and noise_refiner.calls == 2
        torch.testing.assert_close(a[0][:, :3], b[0][:, :3], rtol=0, atol=0)
        assert a[4] is b[4]
        model.patchify_and_embed(x, text, torch.ones(1, 3), torch.ones(1), None, {})
        assert context_refiner.calls == 2 and noise_refiner.calls == 3


@pytest.mark.parametrize("kernel_success", [True, False])
def test_gguf_compatible_online_lora_applied_once_on_packed_or_fallback(kernel_success):
    from gguf import GGMLQuantizationType
    from backend.operations_gguf import ParameterGGUF

    layer = operations.ForgeOperationsGGUF.Linear()
    layer.weight = ParameterGGUF(torch.arange(8, dtype=torch.uint8), tensor_type=GGMLQuantizationType.Q4_0, tensor_shape=(2, 4))
    layer.bias = None
    layer.weight_function = [object()]
    layer.bias_function = []
    x = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
    dense = torch.tensor([[0.25, 0.5, -0.5, 0.75], [1.0, 0.0, 0.25, -1.0]])
    up = torch.tensor([[1.0], [2.0]])
    down = torch.tensor([[0.5, 0.0, 1.0, -1.0]])
    base = F.linear(x, dense)
    expected = F.linear(x, dense + 0.5 * (up @ down))
    with (
        mock.patch.object(operations.args, "activation_space_lora", True, create=True),
        mock.patch.object(operations.args, "enable_triton_backend", False, create=True),
        mock.patch.object(operations, "activation_space_lora_patches", return_value=[(up, down, 0.5)]),
        mock.patch.object(operations, "can_use_llamacpp_cuda_linear", return_value=True),
        mock.patch.object(operations, "weights_manual_cast", return_value=(layer.weight, None, None)) as cast,
        mock.patch.object(operations, "try_llamacpp_cuda_linear", return_value=base if kernel_success else None),
        mock.patch.object(operations, "dequantize_tensor", return_value=dense) as dequantize,
    ):
        actual = layer(x)
    assert cast.call_count == 1 and cast.call_args.kwargs["skip_weight_functions"]
    assert dequantize.call_count == (0 if kernel_success else 1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


class _CFGModel:
    def memory_required(self, shape):
        return 0

    def apply_model(self, x, timestep, c_crossattn, **kwargs):
        return torch.ones_like(x) * c_crossattn[:, :1, :1, None]


@pytest.mark.parametrize("batched", [True, False])
def test_full_area_cfg_direct_predictions_match_existing_accumulation(batched):
    x = torch.zeros(2, 1, 8, 8)
    sigma = torch.ones(2)
    cond = compile_conditions(torch.full((2, 1, 1), 3.0), cache_namespace=("positive",))
    uncond = compile_conditions(torch.full((2, 1, 1), 1.0), cache_namespace=("negative",))
    assert sf._unit_full_area(cond[0], x)
    assert not sf._unit_full_area(dict(cond[0], strength=0.5), x)
    assert not sf._unit_full_area(dict(cond[0], mask=torch.ones(2, 8, 8)), x)
    with (
        mock.patch.object(type(sf.shared), "batch_cond_uncond", new_callable=mock.PropertyMock, return_value=batched),
        mock.patch.object(sf.dynamic_args, "ref_latents", []),
        mock.patch.object(sf.memory_management, "get_free_memory", return_value=2**30),
    ):
        with mock.patch.object(sf, "_native_full_area_cfg", return_value=False):
            expected = sf.calc_cond_uncond_batch(_CFGModel(), cond, uncond, x, sigma, {}, {})
        with mock.patch.object(sf, "_native_full_area_cfg", return_value=True):
            actual = sf.calc_cond_uncond_batch(_CFGModel(), cond, uncond, x, sigma, {}, {})
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], torch.full_like(x, 3.0), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], torch.ones_like(x), rtol=0, atol=0)


def test_vae_direct_float_pixels_preserve_layout_and_legacy_contract():
    pixels = torch.tensor([[[[0.125, 0.5, 0.875], [0.234567, 0.876543, 0.9999]]]])
    vae = SimpleNamespace(first_stage_model=SimpleNamespace(process_out=lambda x: x), decode=lambda x, output_device: pixels.clone())
    engine = SimpleNamespace(forge_objects=SimpleNamespace(vae=vae), is_wan=False)
    engine.decode_first_stage_pixels = MethodType(ForgeDiffusionEngine.decode_first_stage_pixels, engine)
    latent = torch.zeros(1, 4, 1, 1, dtype=torch.float16)
    raw = engine.decode_first_stage_pixels(latent, output_device=torch.device("cpu"))
    legacy = ForgeDiffusionEngine.decode_first_stage(engine, latent, output_device=torch.device("cpu"))
    torch.testing.assert_close(raw, pixels.movedim(-1, 1), rtol=0, atol=0)
    torch.testing.assert_close(legacy, (pixels.movedim(-1, 1) * 2 - 1).to(latent.dtype), rtol=0, atol=0)
    assert raw.dtype == torch.float32 and legacy.dtype == latent.dtype


def test_krea_cfg1_direct_path_carries_namespaced_condition_key():
    from backend.sampling.krea_cache import cached_krea_text_features, sampling_krea_cache_scope, krea_text_fusion_revision

    class KreaLikeModel:
        def __init__(self):
            self.features = torch.nn.Identity().eval()
            self.calls = 0

        def apply_model(self, x, timestep, c_crossattn, transformer_options):
            def compute():
                self.calls += 1
                return c_crossattn.clone()
            return cached_krea_text_features(self.features, c_crossattn, transformer_options, compute)

    model = KreaLikeModel()
    x = torch.zeros(1, 1, 8, 8)
    sigma = torch.ones(1)
    text = torch.ones(1, 2, 3)
    with (torch.no_grad(), sampling_krea_cache_scope("test"), krea_text_fusion_revision("r1"),
          sampling_invariant_scope(1024 * 1024), invariant_revision("r1")):
        condition_cache = {}
        for _ in range(2):
            cond = compile_conditions(text, cache_namespace="positive")
            sf.calc_single_condition(model, cond, x, sigma, {}, condition_cache)
        assert model.calls == 1
        for _ in range(2):
            cond = compile_conditions(text)
            sf.calc_single_condition(model, cond, x, sigma, {}, condition_cache)
        assert model.calls == 3

@pytest.mark.parametrize("with_weight", [True, False])
def test_offload_stream_fences_even_on_kernel_failure(with_weight):
    source = torch.ones(1)
    offload_stream = mock.Mock()
    current = mock.Mock()
    signal = (offload_stream, source if with_weight else None, None if with_weight else source)
    failure = RuntimeError("kernel failed")
    with mock.patch.object(operations.memory_management, "current_stream", return_value=current):
        with pytest.raises(RuntimeError) as raised:
            with operations.main_stream_worker(None, None, signal):
                raise failure
    assert raised.value is failure
    offload_stream.wait_stream.assert_called_once_with(current)


def test_gguf_forced_cast_does_not_bypass_legacy_lora_merge():
    from gguf import GGMLQuantizationType
    from backend.operations_gguf import ParameterGGUF

    layer = operations.ForgeOperationsGGUF.Linear()
    layer.weight = ParameterGGUF(torch.arange(8, dtype=torch.uint8), tensor_type=GGMLQuantizationType.Q4_0, tensor_shape=(2, 4))
    layer.bias = None
    layer.weight_function = [object()]
    layer.bias_function = []
    layer.forge_force_cast_weights = True
    x = torch.ones(1, 4)
    dense = torch.ones(2, 4)
    with (
        mock.patch.object(operations.args, "activation_space_lora", True, create=True),
        mock.patch.object(operations, "activation_space_lora_patches") as validate,
        mock.patch.object(operations, "try_llamacpp_cuda_linear") as packed,
        mock.patch.object(operations, "weights_manual_cast", return_value=(dense, None, None)) as cast,
    ):
        torch.testing.assert_close(layer(x), F.linear(x, dense))
    validate.assert_not_called()
    packed.assert_not_called()
    assert not cast.call_args.kwargs["skip_weight_functions"]

def test_full_area_cfg_rejects_custom_conditions_and_unknown_fields():
    from backend.sampling.condition import Condition, ConditionCrossAttn

    class CustomCondition(ConditionCrossAttn):
        pass

    x = torch.zeros(1, 4, 8, 8)
    text = torch.ones(1, 2, 3)
    standard = compile_conditions(text)[0]
    assert sf._unit_full_area(standard, x)
    assert not sf._unit_full_area(dict(standard, custom_option=True), x)
    assert not sf._unit_full_area(dict(standard, model_conds={"c_crossattn": CustomCondition(text)}), x)
    assert not sf._unit_full_area(dict(standard, model_conds={"c_crossattn": Condition(text), "extra": Condition(text)}), x)



@pytest.mark.parametrize("cache_enabled", [False, True])
def test_legacy_anima_prepare_override_and_native_pos_embedder_call(cache_enabled):
    from backend.nn.anima import Anima
    from backend.sampling.invariant_cache import call_native_with_options

    class LegacyAnima(Anima):
        def prepare_embedded_sequence(self, x, padding_mask=None):
            return x, padding_mask, None

    class LegacyPos(torch.nn.Module):
        def forward(self, x, device):
            return x

    x = torch.ones(1, 1, 2, 2, 2)
    with torch.no_grad(), sampling_invariant_scope(4096 if cache_enabled else 0), invariant_revision(1), invariant_condition(1):
        model = LegacyAnima.__new__(LegacyAnima)
        torch.nn.Module.__init__(model)
        assert call_native_with_options(model.prepare_embedded_sequence, Anima.prepare_embedded_sequence,
                                        x, padding_mask=None, transformer_options={})[0] is x
        model = Anima.__new__(Anima)
        torch.nn.Module.__init__(model)
        model.x_embedder = torch.nn.Identity()
        model.pos_embedder = LegacyPos()
        model.eval()
        embedded, positions, _ = model.prepare_embedded_sequence(x, transformer_options={})
        assert embedded is positions


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_lumina_custom_rope_without_native_attributes(cache_enabled):
    model = NextDiT.__new__(NextDiT)
    torch.nn.Module.__init__(model)
    model.patch_size = 2
    model.pad_tokens_multiple = None
    model.x_embedder = torch.nn.Linear(16, 8)

    class LegacyRope(torch.nn.Module):
        def forward(self, ids):
            return ids.unsqueeze(1)

    model.rope_embedder = LegacyRope()
    model.context_refiner = torch.nn.ModuleList([_Refiner()])
    model.noise_refiner = torch.nn.ModuleList([_Refiner()])
    model.eval()
    x, text = torch.randn(1, 4, 4, 4), torch.randn(1, 3, 8)
    with torch.no_grad(), sampling_invariant_scope(4096 if cache_enabled else 0) as cache, invariant_revision(1), invariant_condition(1):
        model.patchify_and_embed(x, text, None, torch.ones(1), None, {})
        model.patchify_and_embed(x, text, None, torch.ones(1), None, {})
        assert model.context_refiner[0].calls == 2
        assert cache is None or not cache.entries


@pytest.mark.parametrize("target", ["container", "activation", "down"])
def test_native_mlp_preserves_instance_forward_overrides(target):
    from backend.nn import flux
    module = torch.nn.Sequential(torch.nn.Identity(), torch.nn.GELU(approximate="tanh"), torch.nn.Identity()).eval()
    child = {"container": module, "activation": module[1], "down": module[2]}[target]
    original = child.forward
    child.forward = lambda x: original(x) + 10
    x = torch.ones(1, 2)
    with torch.no_grad(), mock.patch.object(flux, "single_gpu_flag", return_value=True), mock.patch.object(flux, "try_native_int8_input_act", return_value=None) as fused:
        torch.testing.assert_close(flux.run_native_mlp(module, x), module(x))
        fused.assert_not_called()
