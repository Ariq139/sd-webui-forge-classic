# https://github.com/comfyanonymous/ComfyUI/blob/v0.7.0/comfy/cli_args.py

import argparse
import enum

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument("--loglevel", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID", help="Set the id of device to use (all other devices will not be visible)")
parser.add_argument("--disable-gpu-warning", action="store_true", help="Disable the low VRAM warnings")
parser.add_argument("--activation-space-lora", action="store_true", help="Run compatible standard online linear LoRAs in activation space; results may differ slightly from dense merging")
parser.add_argument("--invariant-cache-mib", type=int, default=0, metavar="MIB", help="Experimental single-GPU sampling cache budget in MiB; 0 disables")
parser.add_argument("--shared-activation-quantization", action="store_true", help="Experimentally reuse INT8 row quantization between compatible projections on the pinned eager backend")
parser.add_argument("--native-shared-int8-rows", action="store_true", help="Extend shared activation quantization to eligible SM86 CUDA CUTLASS projections (requires --shared-activation-quantization)")
parser.add_argument("--shared-qkv-output", action="store_true", help="Experimental disjoint shared native INT8 GEMM output allocation (requires shared CUDA INT8 rows)")
parser.add_argument("--pack-int8-gate-up", action="store_true", help="Experimental load-time CPU INT8 gate/up coalescing for offloaded native projections; incompatible with quantized pinning")
parser.add_argument("--segmented-int8-linear", action="store_true", help="Experimental SM86 Triton quantization of FLUX attention and MLP segments without a float concatenation (unrotated resident INT8 only)")
parser.add_argument("--resident-denoiser-graph-mib", type=int, default=0, metavar="MIB", help="Experimental resident CFG1 FLUX/Z-Image CUDA graph admission/retention budget; 0 disables; requires --direct-full-area-cfg")
parser.add_argument("--weight-prefetch-mib", type=int, default=0, metavar="MIB", help="Experimental bounded pinned CPU weight look-ahead for Anima, FLUX, Z-Image and Krea; 0 disables")
parser.add_argument("--fused-denoiser-pointwise", action="store_true", help="Experimentally compile pure denoiser modulation and gated-MLP expressions; may slightly change rounding")
parser.add_argument("--native-lumina-gqa", action="store_true", help="Use native PyTorch flash grouped-query attention in compatible unmasked Lumina/Z-Image layers")
parser.add_argument("--native-krea-gqa", action="store_true", help="Avoid expanding Krea KV heads when native PyTorch flash GQA is available")
parser.add_argument("--krea-reference-cache", action="store_true", help="Reuse fixed native Krea reference projections within the invariant cache budget")
parser.add_argument("--direct-full-area-cfg", action="store_true", help="Skip redundant condition copies and unit-weight CFG accumulation for native full-area denoising")
parser.add_argument("--vae-direct-pixels", action="store_true", help="Use native full VAE float32 pixels directly; avoids latent-dtype rounding and may slightly change output")
parser.add_argument("--fast-image-postprocess", action="store_true", help="Convert CHW float images directly into contiguous HWC uint8 output")
parser.add_argument("--native-token-assembly", action="store_true", help="Experimental native Z-Image token assembly and uniform unpatchification")
parser.add_argument("--static-sigma-control", action="store_true", help="Experimental versioned CPU sigma control flow; GPU arithmetic is unchanged")
parser.add_argument("--qwen-intermediate-only", action="store_true", help="Skip unused final Qwen text encoder layers for native Z-Image and Klein conditioning")
parser.add_argument("--qwen-batched-prompts", action="store_true", help="Experimentally batch pairs of equal-length native Qwen prompts when the encoder is resident")
parser.add_argument("--flux-fused-qk-rope", action="store_true", help="Experimental native FLUX/Klein fused Q/K normalization and rotary")
parser.add_argument("--native-int8-mlp-activation", action="store_true", help="Experimental ConvRot INT8 activation-quantizer fusion; can change rounding")
parser.add_argument("--flux-reference-cache", action="store_true", help="Cache immutable native FLUX/Klein reference tokens and compatible dense projections with the invariant cache budget")
parser.add_argument("--target-only-output-head", action="store_true", help="Experimentally skip output projection for discarded reference and caption tokens; can change rounding")
parser.add_argument("--assembled-condition-cache", action="store_true", help="Cache native assembled conditioning tensors within the invariant cache budget")
parser.add_argument("--conditioning-reconstruction-cache", action="store_true", help="Experimental one-entry native conditioning reconstruction cache (up to 32 MiB)")
parser.add_argument("--pin-quantized-weights", action="store_true", help="Permit eligible packed GGUF and supported Comfy-Kitchen weight storage to use existing shared-memory pinning limits")
parser.add_argument("--nunchaku-rope-cache", action="store_true", help="Experimental sampling-scoped Nunchaku Z-Image packed rotary reuse; requires invariant cache budget")
parser.add_argument("--fused-flux-residual", action="store_true", help="Experimental in-place fused FLUX gated residual updates; can change rounding")

parser.add_argument("--text-enc-device", type=str, default=None, metavar="DEVICE", help='Set the device to load text encoder (e.g. "cuda:1")')
parser.add_argument("--vae-device", type=str, default=None, metavar="DEVICE", help='Set the device to load VAE (e.g. "cuda:1")')
parser.add_argument("--multi-gpu", "--multi-gpu-anima", dest="multi_gpu_anima", type=int, nargs="?", const=1, default=None, metavar="CUDA_ID", help="Automatically use a second CUDA device (default: 1)")
parser.add_argument("--multi-gpu-auto", action="store_true", help="Choose replicas or model sharding when the model loads")
parser.add_argument("--multi-gpu-sharding", action="store_true", help="Use model sharding instead of automatic multi-GPU selection")
parser.add_argument("--multi-gpu-split", choices=("balanced_memory", "prefer_primary"), default="balanced_memory", help="Choose a memory-balanced or primary-biased multi-GPU split")
parser.add_argument("--parallel-cfg", type=int, default=None, metavar="CUDA_ID", help="Experimentally evaluate CFG branches on independent GPU replicas")
parser.add_argument("--parallel-batch", type=int, default=None, metavar="CUDA_ID", help="Experimentally split eligible CFG 1 image batches across independent GPU replicas")
parser.add_argument("--parallel-batch-adaptive", action="store_true", help="Experimentally measure and adjust the parallel batch split during sampling")
parser.add_argument("--parallel-cfg-keep-resident", action="store_true", help="Keep an idle parallel CFG replica on its GPU when possible; it remains eligible for memory eviction")

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--force-fp32", action="store_true", help="Force fp32")
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--fp32-unet", action="store_true", help="Store the diffusion model in fp32")
fpunet_group.add_argument("--bf16-unet", action="store_true", help="Store the diffusion model in bf16")
fpunet_group.add_argument("--fp16-unet", action="store_true", help="Store the diffusion model in fp16")
fpunet_group.add_argument("--fp8_e4m3fn-unet", action="store_true", help="Store the diffusion model in fp8_e4m3fn")
fpunet_group.add_argument("--fp8_e5m2-unet", action="store_true", help="Store the diffusion model in fp8_e5m2")
fpunet_group.add_argument("--fp8_e8m0fnu-unet", action="store_true", help="Store the diffusion model in fp8_e8m0fnu")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--fp32-vae", action="store_true", help="Store the VAE in full precision fp32")
fpvae_group.add_argument("--bf16-vae", action="store_true", help="Store the VAE in bf16")
fpvae_group.add_argument("--fp16-vae", action="store_true", help="Store the VAE in fp16 (might cause black images)")
fpvae_group.add_argument("--fp8_e4m3fn-vae", action="store_true", help="Store VAE Conv/Linear weights in fp8_e4m3fn (compute stays in the selected VAE dtype)")
fpvae_group.add_argument("--fp8_e5m2-vae", action="store_true", help="Store VAE Conv/Linear weights in fp8_e5m2 (compute stays in the selected VAE dtype)")

parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--fp32-text-enc", action="store_true", help="Store the text encoder in fp32")
fpte_group.add_argument("--bf16-text-enc", action="store_true", help="Store the text encoder in bf16")
fpte_group.add_argument("--fp16-text-enc", action="store_true", help="Store the text encoder in fp16")
fpte_group.add_argument("--fp8_e4m3fn-text-enc", action="store_true", help="Store the text encoder in fp8_e4m3fn")
fpte_group.add_argument("--fp8_e5m2-text-enc", action="store_true", help="Store the text encoder in fp8_e5m2")

parser.add_argument("--cpu-text-enc", action="store_true", help="Run the text encoder on the CPU")

parser.add_argument("--use-ck-attention", action="store_true", help="use Comfy-Kitchen attention")
parser.add_argument("--use-pytorch-cross-attention", action="store_true", help="Use the PyTorch cross attention (override sageattention/flash_attn/xformers)")
parser.add_argument("--force-xformers-vae", action="store_true", help="Force VAE to use xformers attention (meant to be used with --use-pytorch-cross-attention)")
parser.add_argument("--force-upcast-attention", action="store_true", help="Always upcast to fp32 during attention")
parser.add_argument("--force-w4a8", action="store_true", help="Always use w4a8 when loading convrot_w4a4 models")

parser.add_argument("--sage", action="store_true", help="install sageattention")
parser.add_argument("--flash", action="store_true", help="install flash_attn")
parser.add_argument("--xformers", action="store_true", help="install xformers")
parser.add_argument("--nunchaku", action="store_true", help="install nunchaku for SVDQ inference")
parser.add_argument("--pynvml", action="store_true", help="install pyNVML for accurate VRAM tracking")

parser.add_argument("--disable-sage", action="store_true", help="disable sageattention")
parser.add_argument("--disable-flash", action="store_true", help="disable flash_attn")
parser.add_argument("--disable-xformers", action="store_true", help="disable xformers")

parser.add_argument("--deterministic", action="store_true", help="Use slower deterministic algorithms when possible")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--gpu-only", action="store_true", help="Store and run everything on the GPU")
vram_group.add_argument("--highvram", action="store_true", help="Keeps models in VRAM after usage")
vram_group.add_argument("--lowvram", action="store_true", help="Split the diffusion model in parts to use less VRAM")
vram_group.add_argument("--novram", action="store_true", help="When even LOW_VRAM is still not enough")
vram_group.add_argument("--cpu", action="store_true", help="Use the CPU for everything (slow)")

parser.add_argument("--reserve-vram", type=float, default=None, metavar="GB", help="Set the amount of VRAM you want to reserve for other software (by default some amount is reserved)")
parser.add_argument("--disable-smart-memory", action="store_true", help="Aggressively offload to RAM instead of keeping models in VRAM when possible")
parser.add_argument("--force-non-blocking", action="store_true", help="Use non-blocking operations for all applicable tensors")

parser.add_argument("--cuda-malloc", action="store_true", help="improve memory allocation")
parser.add_argument("--cuda-stream", type=int, nargs="?", metavar="NUM_STREAMS", const=2, help="improve offloading")
parser.add_argument("--pin-shared-memory", action="store_true", help="improve RAM utilization")
parser.add_argument("--expandable-segments", action="store_true", help="improve memory allocation ; experimental")

parser.add_argument("--fast-fp8", action="store_true", help="torch._scaled_mm")
parser.add_argument("--fast-fp16", action="store_true", help="torch.backends.cuda.matmul.allow_fp16_accumulation")
parser.add_argument("--autotune", action="store_true", help="torch.backends.cudnn.benchmark")

parser.add_argument("--mmap-torch-files", action="store_true", help="Use mmap when loading ckpt/pt files")
parser.add_argument("--disable-mmap", action="store_true", help="Don't use mmap when loading safetensors")

parser.add_argument("--tiled-conv2d", type=int, default=0, metavar="TILE_SIZE", choices=[0, 64, 128, 256, 512], help="reduce VAE memory usage ; increase processing time")
parser.add_argument("--enable-triton-backend", action="store_true", help="Enable the use of Triton backend in comfy-kitchen and supported GGUF K-quant layers")


class SageAttentionFuncs(enum.Enum):
    auto = "auto"
    fp16_triton = "fp16_triton"
    fp16_cuda = "fp16_cuda"
    fp8_cuda = "fp8_cuda"
    fp8_cuda_pp = "fp8_cuda++"
    sageattn3 = "sageattn3"


sage = parser.add_argument_group(description="SageAttention")
sage.add_argument("--sage-function", type=SageAttentionFuncs, default=SageAttentionFuncs.auto)


def configure_multi_gpu(parsed):
    """Expand the automatic multi-GPU option into the existing execution flags."""
    if parsed.multi_gpu_anima is not None and not parsed.multi_gpu_sharding:
        parsed.multi_gpu_auto = True
        if parsed.parallel_cfg is None:
            parsed.parallel_cfg = parsed.multi_gpu_anima
        if parsed.parallel_batch is None:
            parsed.parallel_batch = parsed.multi_gpu_anima
        parsed.parallel_batch_adaptive = True
        parsed.parallel_cfg_keep_resident = True
    elif parsed.multi_gpu_sharding:
        parsed.multi_gpu_auto = False
    return parsed


args, _ = parser.parse_known_args()
configure_multi_gpu(args)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    import torch

    from backend.misc.context_windows import IndexListContextHandler


class _DynamicArgsMeta(type):
    def get(cls, key, default=None):
        return getattr(cls, key, default)

    def __getitem__(cls, key, default=None):
        return getattr(cls, key, default)

    def __setitem__(cls, key, value):
        setattr(cls, key, value)

    def __contains__(cls, key):
        return hasattr(cls, key)


class dynamic_args(metaclass=_DynamicArgsMeta):
    """Some parameters that are used throughout the Webui"""

    embedding_dir: "os.PathLike" = None
    """set in modules/sd_models/forge_model_reload"""
    forge_unet_storage_dtype: "torch.dtype" = None
    """set in modules/sd_models/forge_model_reload"""
    online_lora: bool = False
    """patch LoRAs on-the-fly"""
    kontext: bool = False
    """Flux Kontext"""
    edit: bool = False
    """Qwen-Image-Edit"""
    qwen21: bool = False
    """Qwen-Image-2.1"""
    nunchaku: bool = False
    """Nunchaku (SVDQ) Models"""
    klein: bool = False
    """Flux.2 Klein"""
    wan: bool = False
    """Wan 2.2"""
    pid: bool = False
    """PiD"""
    anima: bool = False
    """Anima"""
    krea2: bool = False
    """Krea 2"""
    moss: bool = False
    """Moss-Image"""
    ref_latents: list["torch.Tensor"] = []
    """Reference Latent(s) for Edit Models"""
    moss_reference_method: str = None
    """reference_latents_method for Moss-Image"""
    moss_ref_kv: "torch.Tensor" = None
    """Cached per-block reference K/V for Moss-Image isolated reference methods"""
    concat_latent: "torch.Tensor" = None
    """Input Latent for Wan 2.2 I2V"""
    lq_latent: list["torch.Tensor", "torch.Tensor"] = [None, None]
    """lq_latent & degrade_sigma for PiD"""
    context_handler: "IndexListContextHandler" = None
    """Context Handler for PiD"""
    image_slots: list[int] = None
    """Image Slots for Qwen-Image-2.1"""
    is_referencing: bool = False
    """Appending Reference Latent(s) (by. ImageStitch)"""
    ops: str = None
    """Operations for the Diffusion Model"""
    last_extra_generation_params: dict[str, str] = {}
    """Infotext captured during `get_learned_conditioning`"""
    loading_refiner: bool = False
    """Do not reset when loading Refiner"""

    @classmethod
    def reset(cls):
        if cls.loading_refiner:
            return

        cls.ref_latents.clear()
        cls.moss_reference_method = None
        cls.moss_ref_kv = None
        cls.concat_latent = None
        cls.lq_latent = [None, None]
        cls.context_handler = None
        cls.image_slots = None

    @classmethod
    def can_reference(cls) -> bool:
        return any(getattr(cls, key) for key in ("kontext", "edit", "qwen21", "klein", "wan", "anima", "krea2", "moss"))
