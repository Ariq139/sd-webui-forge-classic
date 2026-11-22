import copy
import unittest
from unittest import mock

import torch

from backend import memory_management as mm
from backend.args import args
from backend.multi_gpu.patcher import ShardedAnimaPatcher
from backend.multi_gpu.patches import device_aware_options
from backend.nn.flux import IntegratedFluxTransformer2DModel
from backend.nn.krea import SingleStreamDiT
from backend.nn.unet import IntegratedUNet2DConditionModel
from backend.nn.moss_image import MOSSImage
from backend.nn.qwen import QwenImageTransformer2DModel
from backend.nn.qwen21 import QwenImage21Transformer2DModel
from backend.nn.wan import WanModel
from backend.nn.lumina import NextDiT
from backend.nn.ernie import ErnieImageModel
from backend.nn.chroma import IntegratedChromaTransformer2DModel
from backend.nn.pixeldit.pid import PidNet
from backend.args import dynamic_args
from backend.operations import using_forge_operations


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.diffusion_model = model
        self.computation_dtype = torch.float32


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "two CUDA devices required")
class AdapterTests(unittest.TestCase):
    def check_model(self, factory, inputs, splits):
        torch.manual_seed(12)
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = factory()
        for param in model.parameters():
            torch.nn.init.uniform_(param, -0.02, 0.02)
        reference = copy.deepcopy(model).eval().to("cuda:0")
        inputs = [x.to("cuda:0") for x in inputs]
        with torch.inference_mode():
            expected = reference(*[x.clone() for x in inputs])
        del reference
        wrapper = Wrapper(model.eval())
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(wrapper, torch.device("cuda:0"), torch.device("cpu"))
            for split in splits:
                patcher._planned_split = split
                with mock.patch.object(patcher, "device_memory_requirements", return_value={torch.device("cuda:0"): 0, torch.device("cuda:1"): 0}):
                    mm.load_models_gpu([patcher])
                with torch.inference_mode():
                    actual = model(*[x.clone() for x in inputs])
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
                mm.free_memory(float("inf"), torch.device("cuda:1"))

    def test_krea(self):
        self.check_model(lambda: SingleStreamDiT(features=32, tdim=32, txtdim=32, heads=2, kvheads=2, multiplier=1, layers=3, patch=2, channels=2, txtlayers=2, txtheads=2, txtkvheads=2), [torch.randn(1, 2, 1, 4, 4), torch.ones(1), torch.randn(1, 3, 2, 32)], [1, 2])

    def test_krea_frees_other_models_before_initial_placement(self):
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = SingleStreamDiT(features=32, tdim=32, txtdim=32, heads=2, kvheads=2, multiplier=1, layers=2, patch=2, channels=2, txtlayers=2, txtheads=2, txtkvheads=2)
        with mock.patch.object(args, "multi_gpu_anima", 1):
            patcher = ShardedAnimaPatcher(Wrapper(model), torch.device("cuda:0"), torch.device("cpu"))
            with mock.patch.object(mm, "free_memory") as free:
                protected = [object()]
                patcher.prepare_sharded_load(protected)
                free.assert_called_once_with(float("inf"), torch.device("cuda:0"), keep_loaded=protected)

    def test_flux(self):
        self.check_model(lambda: IntegratedFluxTransformer2DModel(in_channels=2, out_channels=2, vec_in_dim=None, context_in_dim=32, hidden_size=32, mlp_ratio=2, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000, patch_size=1, qkv_bias=True, guidance_embed=False), [torch.randn(1, 2, 4, 4), torch.ones(1), torch.randn(1, 3, 32)], [1, 3])

    def test_flux_global_modulation(self):
        self.check_model(lambda: IntegratedFluxTransformer2DModel(in_channels=2, out_channels=2, vec_in_dim=None, context_in_dim=32, hidden_size=32, mlp_ratio=2, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000, patch_size=1, qkv_bias=True, guidance_embed=False, global_modulation=True), [torch.randn(1, 2, 4, 4), torch.ones(1), torch.randn(1, 3, 32)], [1, 3])

    def test_unet(self):
        self.check_model(lambda: IntegratedUNet2DConditionModel(in_channels=4, model_channels=32, out_channels=4, num_res_blocks=1, channel_mult=(1,), num_heads=1, transformer_depth=[0], transformer_depth_middle=-1, transformer_depth_output=[0, 0]), [torch.randn(1, 4, 8, 8), torch.ones(1)], [1, 3])

    def test_moss(self):
        self.check_model(lambda: MOSSImage(dim=32, ffn_dim=64, num_heads=2, num_layers=3, text_dim=32, latent_channels=2), [torch.randn(1, 2, 4, 4), torch.ones(1), torch.randn(1, 3, 32)], [1, 2])

    def test_moss_reference_cache(self):
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = MOSSImage(dim=32, ffn_dim=64, num_heads=2, num_layers=3, text_dim=32, latent_channels=2).eval()
        for param in model.parameters():
            torch.nn.init.uniform_(param, -0.02, 0.02)
        reference = copy.deepcopy(model).to("cuda:0")
        ref = torch.randn(1, 2, 2, 2, device="cuda:0")
        x, t, context = torch.randn(1, 2, 4, 4, device="cuda:0"), torch.ones(1, device="cuda:0"), torch.randn(1, 3, 32, device="cuda:0")
        with mock.patch.object(dynamic_args, "ref_latents", [ref]), mock.patch.object(dynamic_args, "moss_reference_method", "isolated"), mock.patch.object(dynamic_args, "moss_ref_kv", None):
            with torch.inference_mode():
                expected = reference(x, t, context)
                expected_kv = dynamic_args.moss_ref_kv.clone()
            dynamic_args.moss_ref_kv = None
            del reference
            with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
                patcher = ShardedAnimaPatcher(Wrapper(model), torch.device("cuda:0"), torch.device("cpu"))
                patcher._planned_split = 1
                with mock.patch.object(patcher, "device_memory_requirements", return_value={torch.device("cuda:0"): 0, torch.device("cuda:1"): 0}):
                    mm.load_models_gpu([patcher])
                with torch.inference_mode():
                    actual = model(x, t, context)
                torch.testing.assert_close(dynamic_args.moss_ref_kv, expected_kv, atol=2e-4, rtol=2e-4)
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
                mm.free_memory(float("inf"), torch.device("cuda:1"))

    def test_qwen(self):
        self.check_model(lambda: QwenImageTransformer2DModel(patch_size=2, in_channels=8, out_channels=2, num_layers=3, attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=32, axes_dims_rope=(4, 6, 6)), [torch.randn(1, 2, 1, 4, 4), torch.ones(1), torch.randn(1, 3, 32)], [1, 2])

    def test_qwen21(self):
        self.check_model(lambda: QwenImage21Transformer2DModel(in_channels=2, out_channels=2, num_layers=3, attention_head_dim=16, num_attention_heads=2, context_in_dim=32, mlp_ratio=2, axes_dims_rope=(4, 6, 6), fused_mlp=False), [torch.randn(1, 2, 4, 4), torch.ones(1), torch.randn(1, 3, 32)], [1, 2])

    def test_qwen21_reference_segments(self):
        with mock.patch.object(dynamic_args, "ref_latents", [torch.randn(1, 2, 2, 2)]), mock.patch.object(dynamic_args, "image_slots", [1]):
            self.check_model(lambda: QwenImage21Transformer2DModel(in_channels=2, out_channels=2, num_layers=3, attention_head_dim=16, num_attention_heads=2, context_in_dim=32, mlp_ratio=2, axes_dims_rope=(4, 6, 6), fused_mlp=False), [torch.randn(1, 2, 4, 4), torch.ones(1), torch.randn(1, 3, 32)], [1, 2])

    def test_qwen21_patch_sees_input_on_block_device(self):
        seen = []
        replacement_devices = []
        def replace(values, helpers):
            replacement_devices.append((values["img"].device, values["vec"].device))
            return helpers["original_block"](values)
        def patch(values):
            seen.append((values["img"].device, values["x"].device))
            return values
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = QwenImage21Transformer2DModel(in_channels=2, out_channels=2, num_layers=3, attention_head_dim=16, num_attention_heads=2, context_in_dim=32, mlp_ratio=2, axes_dims_rope=(4, 6, 6), fused_mlp=False).eval()
        for param in model.parameters():
            torch.nn.init.uniform_(param, -0.02, 0.02)
        x, t, context = torch.randn(1, 2, 4, 4, device="cuda:0"), torch.ones(1, device="cuda:0"), torch.randn(1, 3, 32, device="cuda:0")
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(Wrapper(model), torch.device("cuda:0"), torch.device("cpu"))
            patcher._planned_split = 1
            with mock.patch.object(patcher, "device_memory_requirements", return_value={torch.device("cuda:0"): 0, torch.device("cuda:1"): 0}):
                mm.load_models_gpu([patcher])
            with torch.inference_mode():
                model(x, t, context, transformer_options=device_aware_options({"patches": {"single_block": [patch]}, "patches_replace": {"dit": {("single_block", 1): replace}}}))
            self.assertEqual(replacement_devices, [(torch.device("cuda:1"), torch.device("cuda:1"))])
            self.assertEqual(seen, [(torch.device("cuda:0"), torch.device("cuda:0")), (torch.device("cuda:1"), torch.device("cuda:1")), (torch.device("cuda:1"), torch.device("cuda:1"))])
            mm.free_memory(float("inf"), torch.device("cuda:1"))

    def test_wan(self):
        self.check_model(lambda: WanModel(patch_size=(1, 2, 2), in_dim=2, dim=32, ffn_dim=64, freq_dim=32, text_dim=32, out_dim=2, num_heads=2, num_layers=3), [torch.randn(1, 2, 1, 4, 4), torch.ones(1), torch.randn(1, 3, 32)], [1, 2])

    def test_lumina(self):
        self.check_model(lambda: NextDiT(patch_size=2, in_channels=2, dim=32, n_layers=3, n_refiner_layers=1, n_heads=2, multiple_of=16, cap_feat_dim=32, axes_dims=[4, 6, 6], axes_lens=[16, 32, 32]), [torch.randn(1, 2, 4, 4), torch.ones(1), torch.randn(1, 3, 32)], [1, 2])

    def test_ernie(self):
        self.check_model(lambda: ErnieImageModel(hidden_size=32, num_attention_heads=2, num_layers=3, ffn_hidden_size=64, in_channels=2, out_channels=2, text_in_dim=32, rope_axes_dim=(4, 6, 6)), [torch.randn(1, 2, 4, 4), torch.ones(1), torch.randn(1, 3, 32)], [1, 2])

    def test_chroma(self):
        self.check_model(lambda: IntegratedChromaTransformer2DModel(in_channels=2, out_channels=2, context_in_dim=32, hidden_size=32, mlp_ratio=2, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000, patch_size=1, qkv_bias=True, in_dim=64, out_dim=32, hidden_dim=32, n_layers=1, txt_ids_dims=[], vec_in_dim=0), [torch.randn(1, 2, 4, 4), torch.ones(1), torch.randn(1, 3, 32)], [1, 3])

    def test_pid(self):
        with mock.patch.object(dynamic_args, "lq_latent", (torch.randn(1, 2, 2, 2), torch.ones(1))):
            self.check_model(lambda: PidNet(lq_latent_channels=2, lq_hidden_dim=8, lq_num_res_blocks=0, lq_interval=1, sr_scale=1, latent_spatial_down_factor=2, rope_ref_h=8, rope_ref_w=8, in_channels=2, num_groups=2, hidden_size=32, pixel_hidden_size=8, pixel_attn_hidden_size=16, pixel_num_groups=2, patch_depth=2, pixel_depth=3, patch_size=2, txt_embed_dim=32, txt_max_length=8), [torch.randn(1, 2, 4, 4), torch.ones(1), torch.randn(1, 3, 32)], [1, 2])

    def test_device_aware_unet_patches_and_online_weights(self):
        class Offset:
            def __init__(self):
                self.value = torch.tensor(0.01)
                self.devices = []
            def to(self, device):
                self.value = self.value.to(device)
                return self
            def __call__(self, value, *unused):
                self.devices.append(value.device.index)
                return value + self.value

        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = IntegratedUNet2DConditionModel(in_channels=4, model_channels=32, out_channels=4, num_res_blocks=1, channel_mult=(1,), num_heads=1, transformer_depth=[0], transformer_depth_middle=-1, transformer_depth_output=[0, 0])
        for param in model.parameters():
            torch.nn.init.uniform_(param, -0.02, 0.02)
        model.eval()
        root = Wrapper(model)
        x, t = torch.randn(1, 4, 8, 8, device="cuda:0"), torch.ones(1, device="cuda:0")
        patch = Offset()
        options = {"patches": {"input_block_patch": [patch]}}
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(root, torch.device("cuda:0"), torch.device("cpu"))
            patcher._planned_split = 1
            key = "diffusion_model.input_blocks.1.0.in_layers.2.weight"
            module = model.input_blocks[1][0].in_layers[2]
            original = module.weight.detach().clone()
            patcher.add_patches({key: (torch.ones_like(original),)}, strength_patch=0.01, online_mode=True)
            with mock.patch.object(patcher, "device_memory_requirements", return_value={torch.device("cuda:0"): 0, torch.device("cuda:1"): 0}):
                mm.load_models_gpu([patcher])
            self.assertEqual(module.weight.device, torch.device("cuda:1"))
            self.assertTrue(module.parameters_manual_cast)
            self.assertEqual(len(module.weight_function), 1)
            wrappers = module.weight_function
            with torch.inference_mode():
                module.weight_function = []
                baseline = model(x, t, transformer_options=device_aware_options(options))
                module.weight_function = wrappers
                patched = model(x, t, transformer_options=device_aware_options(options))
            self.assertFalse(torch.equal(baseline, patched))
            self.assertEqual(set(patch.devices), {0, 1})
            mm.free_memory(float("inf"), torch.device("cuda:1"))
            self.assertFalse(module.weight_function)
            torch.testing.assert_close(module.weight, original)

    def test_flux_replacement_callback_on_secondary(self):
        calls = []
        def replace(arguments, callbacks):
            calls.append((arguments["img"].device.index, torch.cuda.current_device()))
            return callbacks["original_block"](arguments)

        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = IntegratedFluxTransformer2DModel(in_channels=2, out_channels=2, vec_in_dim=None, context_in_dim=32, hidden_size=32, mlp_ratio=2, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000, patch_size=1, qkv_bias=True, guidance_embed=False)
        for param in model.parameters():
            torch.nn.init.uniform_(param, -0.02, 0.02)
        root = Wrapper(model.eval())
        x, t, c = torch.randn(1, 2, 4, 4, device="cuda:0"), torch.ones(1, device="cuda:0"), torch.randn(1, 3, 32, device="cuda:0")
        with mock.patch.object(args, "multi_gpu_anima", 1), mock.patch.object(mm, "current_loaded_models", []):
            patcher = ShardedAnimaPatcher(root, torch.device("cuda:0"), torch.device("cpu"))
            patcher._planned_split = 1
            with mock.patch.object(patcher, "device_memory_requirements", return_value={torch.device("cuda:0"): 0, torch.device("cuda:1"): 0}):
                mm.load_models_gpu([patcher])
            options = device_aware_options({"patches_replace": {"dit": {("double_block", 1): replace}}})
            with torch.inference_mode():
                model(x, t, c, transformer_options=options)
            self.assertEqual(calls, [(1, 1)])
            mm.free_memory(float("inf"), torch.device("cuda:1"))
