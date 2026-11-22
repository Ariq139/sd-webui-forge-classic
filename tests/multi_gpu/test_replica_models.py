import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from backend.args import dynamic_args
from backend.multi_gpu.parallel import ParallelBatch, ParallelCFG
from backend.multi_gpu.replicas import ReplicaPair
from backend.nn.chroma import IntegratedChromaTransformer2DModel
from backend.nn.flux import IntegratedFluxTransformer2DModel
from backend.nn.krea import SingleStreamDiT
from backend.nn.moss_image import MOSSImage
from backend.nn.qwen import QwenImageTransformer2DModel
from backend.nn.qwen21 import QwenImage21Transformer2DModel
from backend.nn.unet import IntegratedUNet2DConditionModel
from backend.operations import using_forge_operations
from backend.sampling.condition import compile_conditions
from backend.sampling.sampling_function import sampling_function_inner


class ReplicaModel(torch.nn.Module):
    def __init__(self, diffusion_model, video_input=False):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.video_input = video_input
        self.computation_dtype = torch.float32
        self.predictor = SimpleNamespace(percent_to_sigma=lambda value: value)

    def get_dtype(self):
        return torch.float32

    def memory_required(self, input_shape):
        return 0

    def apply_model(self, x, timestep, c_crossattn, transformer_options=None, **kwargs):
        if self.video_input:
            x = x.unsqueeze(2)
        result = self.diffusion_model(x, timestep, c_crossattn, transformer_options=transformer_options or {})
        return result.squeeze(2) if self.video_input else result


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "two CUDA devices required")
class ReplicaArchitectureTests(unittest.TestCase):
    def test_shared_reference_state_disables_replica_execution(self):
        x = torch.randn(2, 2, 4, 4, device="cuda:0")
        cond = compile_conditions(torch.randn(2, 3, 32))
        with mock.patch.object(dynamic_args, "context_handler", None), mock.patch.object(dynamic_args, "ref_latents", [torch.ones(1)]):
            self.assertFalse(ParallelCFG.eligible(cond, cond, {}, x))
            self.assertFalse(ParallelBatch.eligible(cond, None, {}, x))
        with mock.patch.object(dynamic_args, "ref_latents", []), mock.patch.object(dynamic_args, "qwen21", True), mock.patch.object(dynamic_args, "image_slots", [0]):
            self.assertFalse(ParallelCFG.eligible(cond, cond, {}, x))
            self.assertFalse(ParallelBatch.eligible(cond, None, {}, x))
        with mock.patch.object(dynamic_args, "ref_latents", []), mock.patch.object(dynamic_args, "moss", True), mock.patch.object(dynamic_args, "moss_ref_kv", torch.ones(1)):
            self.assertFalse(ParallelCFG.eligible(cond, cond, {}, x))
            self.assertFalse(ParallelBatch.eligible(cond, None, {}, x))

    def check_replicas(self, factory, channels, context_shape, *, video_input=False, flags=None):
        from backend.sampling.sampling_function import _parallel_batch_model_supported

        torch.manual_seed(42)
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = factory()
        for parameter in model.parameters():
            torch.nn.init.uniform_(parameter, -0.02, 0.02)
        model = ReplicaModel(model.eval(), video_input=video_input)
        self.assertTrue(_parallel_batch_model_supported(SimpleNamespace(model=model)))
        flag_context = mock.patch.multiple(dynamic_args, **flags) if flags else contextlib.nullcontext()
        with mock.patch.object(dynamic_args, "ref_latents", []), mock.patch.object(dynamic_args, "context_handler", None), flag_context, ReplicaPair(model, ("cuda:0", "cuda:1")) as pair:
            primary, secondary = pair.models
            batch_adapter = ParallelBatch(primary, secondary, pair.devices)
            cfg_adapter = ParallelCFG(primary, secondary, pair.devices)
            try:
                for count in (2, 3):
                    x = torch.randn(count, channels, 4, 4, device="cuda:0")
                    timestep = torch.ones(count, device="cuda:0")
                    cond = compile_conditions(torch.randn(count, *context_shape))
                    uncond = compile_conditions(torch.randn(count, *context_shape))
                    cond[0]["batch_indices"] = list(range(count))
                    self.assertTrue(batch_adapter.eligible(cond, None, {}, x))
                    self.assertTrue(cfg_adapter.eligible(cond, uncond, {}, x))
                    with torch.inference_mode():
                        batch_actual = sampling_function_inner(primary, x, timestep, None, cond, 1.0, return_full=True, parallel_cfg=batch_adapter)
                        batch_expected = sampling_function_inner(primary, x, timestep, None, cond, 1.0, return_full=True)
                        cfg_actual = sampling_function_inner(primary, x, timestep, uncond, cond, 2.0, return_full=True, parallel_cfg=cfg_adapter)
                        cfg_expected = sampling_function_inner(primary, x, timestep, uncond, cond, 2.0, return_full=True)
                    for actual, expected in zip(batch_actual + cfg_actual, batch_expected + cfg_expected):
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            finally:
                batch_adapter.close()
                cfg_adapter.close()

    def test_krea_cfg_and_batch_replicas(self):
        self.check_replicas(lambda: SingleStreamDiT(features=32, tdim=32, txtdim=32, heads=2, kvheads=2, multiplier=1, layers=3, patch=2, channels=2, txtlayers=2, txtheads=2, txtkvheads=2), 2, (3, 2, 32), video_input=True)

    def test_krea_replica_lifecycle_with_online_and_merged_lora(self):
        from backend import memory_management as mm
        from backend.args import args
        from backend.patcher.unet import UnetPatcher
        from backend.sampling import sampling_function as sampling_module

        for online in (False, True):
            with self.subTest(online=online), using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
                model = SingleStreamDiT(features=32, tdim=32, txtdim=32, heads=2, kvheads=2, multiplier=1, layers=2, patch=2, channels=2, txtlayers=2, txtheads=2, txtkvheads=2)
                for parameter in model.parameters():
                    torch.nn.init.uniform_(parameter, -0.02, 0.02)
                unet = UnetPatcher(ReplicaModel(model.eval(), video_input=True), torch.device("cuda:0"), torch.device("cpu"))
                key = "diffusion_model.blocks.0.attn.wq.weight"
                unet.add_patches({key: (torch.full_like(model.blocks[0].attn.wq.weight, 0.01),)}, strength_patch=0.5, online_mode=online)
                x = torch.randn(3, 2, 4, 4, device="cuda:0")
                timestep = torch.ones(3, device="cuda:0")
                cond = compile_conditions(torch.randn(3, 3, 2, 32))
                with mock.patch.object(args, "parallel_batch", 1), mock.patch.object(args, "parallel_cfg", None), mock.patch.object(mm, "current_loaded_models", []), mock.patch.object(dynamic_args, "ref_latents", []), mock.patch.object(dynamic_args, "context_handler", None), mock.patch.object(sampling_module, "_parallel_batch_callbacks_safe", return_value=True), mock.patch.object(sampling_module, "_parallel_cfg_cache", None):
                    try:
                        sampling_module.sampling_prepare(unet, x, cfg_scale=1.0, cfg1_single_condition=True)
                        adapter = sampling_module._load_parallel_cfg(unet, cond, None, {}, x, batch_mode=True)
                        self.assertIsNotNone(adapter)
                        with torch.inference_mode():
                            actual = sampling_function_inner(unet.model, x, timestep, None, cond, 1.0, return_full=True, parallel_cfg=adapter)
                            expected = sampling_function_inner(unet.model, x, timestep, None, cond, 1.0, return_full=True)
                        for observed, reference in zip(actual, expected):
                            torch.testing.assert_close(observed, reference, atol=2e-4, rtol=2e-4)
                    finally:
                        sampling_module._release_parallel_cfg(unet)
                        mm.free_memory(float("inf"), torch.device("cuda:0"))

    def test_flux_cfg_and_batch_replicas(self):
        self.check_replicas(lambda: IntegratedFluxTransformer2DModel(in_channels=2, out_channels=2, vec_in_dim=None, context_in_dim=32, hidden_size=32, mlp_ratio=2, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000, patch_size=1, qkv_bias=True, guidance_embed=False), 2, (3, 32))

    def test_chroma_cfg_and_batch_replicas(self):
        self.check_replicas(lambda: IntegratedChromaTransformer2DModel(in_channels=2, out_channels=2, context_in_dim=32, hidden_size=32, mlp_ratio=2, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000, patch_size=1, qkv_bias=True, in_dim=64, out_dim=32, hidden_dim=32, n_layers=1, txt_ids_dims=[], vec_in_dim=0), 2, (3, 32))

    def test_unet_cfg_and_batch_replicas(self):
        self.check_replicas(lambda: IntegratedUNet2DConditionModel(in_channels=4, model_channels=32, out_channels=4, num_res_blocks=1, channel_mult=(1,), num_heads=1, transformer_depth=[0], transformer_depth_middle=-1, transformer_depth_output=[0, 0]), 4, (3, 32))

    def test_moss_text_only_cfg_and_batch_replicas(self):
        self.check_replicas(lambda: MOSSImage(dim=32, ffn_dim=64, num_heads=2, num_layers=3, text_dim=32, latent_channels=2), 2, (3, 32), flags={"moss": True, "moss_ref_kv": None})

    def test_qwen_cfg_and_batch_replicas(self):
        self.check_replicas(lambda: QwenImageTransformer2DModel(patch_size=2, in_channels=8, out_channels=2, num_layers=3, attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=32, axes_dims_rope=(4, 6, 6)), 2, (3, 32), video_input=True)

    def test_qwen21_text_only_cfg_and_batch_replicas(self):
        self.check_replicas(lambda: QwenImage21Transformer2DModel(in_channels=2, out_channels=2, num_layers=3, attention_head_dim=16, num_attention_heads=2, context_in_dim=32, mlp_ratio=2, axes_dims_rope=(4, 6, 6), fused_mlp=False), 2, (3, 32), flags={"qwen21": True, "image_slots": None})
