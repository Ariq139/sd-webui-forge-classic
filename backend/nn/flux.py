# https://github.com/Comfy-Org/ComfyUI/tree/v0.9.0/comfy/ldm/flux
# Reference: https://github.com/black-forest-labs/flux

import math
from dataclasses import dataclass
from contextlib import nullcontext

import torch
from einops import rearrange, repeat
from torch import nn

from backend import memory_management
from backend.args import dynamic_args
from backend.attention import attention_function
from backend.quant_ops import ck
from backend.sampling.invariant_cache import cached_invariant
from backend.activation_quantization import shared_activation_scope
from backend.weight_prefetch import weight_prefetch_scope
from backend.fused_pointwise import silu_gate, norm_affine, can_fuse_silu
from backend.utils import fp16_fix, pad_to_patch_size
from backend.single_gpu_fastpaths import dense_tokenwise_head, reference_signature, single_gpu_flag
from backend.sampling.invariant_cache import invariant_scope_active, retain_reference_sources, invariants_enabled
from backend.sampling.invariant_cache import cache_dependencies_safe
from backend.operations_mixed_precision import try_native_int8_input_act


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor, mask=None, transformer_options={}) -> torch.Tensor:
    if pe is not None:
        q, k = ck.apply_rope(q, k, pe)
    heads = q.shape[1]
    x = attention_function(q, k, v, heads, skip_reshape=True, mask=mask, transformer_options=transformer_options)
    return x


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    if not memory_management.supports_fp64(pos.device):
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, bias=True):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class YakMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with shared_activation_scope(x, (self.gate_proj, self.up_proj)):
            gate = self.gate_proj(x)
            if (single_gpu_flag("native_int8_mlp_activation") and gate.dtype == torch.bfloat16
                    and type(self) is YakMLP and type(self.act_fn) is nn.SiLU
                    and not self.act_fn.inplace and cache_dependencies_safe(self)):
                up = self.up_proj(x)
                fused = try_native_int8_input_act(self.down_proj, (gate, up), "swiglu")
                if fused is not None:
                    return fused
                hidden = self.act_fn(gate) * up
            elif can_fuse_silu(self.act_fn, gate):
                hidden = silu_gate(gate, self.up_proj(x))
            else:
                hidden = self.act_fn(gate) * self.up_proj(x)
        down_proj = self.down_proj(hidden)
        return down_proj


def run_native_mlp(module, x):
    if (not single_gpu_flag("native_int8_mlp_activation")
            or type(module) is not nn.Sequential or len(module) != 3
            or not cache_dependencies_safe(module)):
        return module(x)
    act = module[1]
    if type(act) is SiLUActivation and type(act.gate_fn) is nn.SiLU and not act.gate_fn.inplace:
        kind = "swiglu"
    elif type(act) is nn.GELU and act.approximate == "tanh":
        kind = "gelu_tanh"
    else:
        return module(x)
    hidden = module[0](x)
    fused = try_native_int8_input_act(module[2], hidden, kind)
    return module[2](act(hidden)) if fused is None else fused


def build_mlp(hidden_size: int, mlp_hidden_dim: int, mlp_silu_act: bool = False, yak_mlp: bool = False) -> nn.Module:
    if yak_mlp:
        return YakMLP(hidden_size, mlp_hidden_dim)
    if mlp_silu_act:
        return nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )
    else:
        return nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim)
        self.key_norm = nn.RMSNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

    def try_fused_rope(self, q, k, v, pe, options):
        if (not single_gpu_flag("flux_fused_qk_rope") or pe is None
                or not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16)
                or q.dtype != k.dtype or q.dtype != v.dtype
                or q.device != k.device or q.device != v.device or q.device != pe.device
                or q.shape != k.shape or q.shape != v.shape
                or torch.is_grad_enabled() or torch.is_autocast_enabled(q.device.type)
                or not isinstance(options, dict)
                or options.keys() - {"cond_or_uncond", "sigmas", "cond_mark", "cond_indices",
                                     "uncond_indices", "sampling_sigmas", "total_blocks", "block_type", "block_index"}
                or not cache_dependencies_safe(self)):
            return None
        if (pe.ndim != 6 or pe.shape[0] not in (1, q.shape[0])
                or pe.shape[1] != 1 or pe.shape[2] != q.shape[2]
                or pe.shape[3:] != (q.shape[-1] // 2, 2, 2)):
            return None
        from backend.operations import ForgeOperations
        from comfy_kitchen import registry
        from comfy_kitchen.exceptions import NoCapableBackendError
        scales = []
        eps = []
        for norm in (self.query_norm, self.key_norm):
            if (type(norm) not in (nn.RMSNorm, ForgeOperations.RMSNorm)
                    or getattr(norm, "add", False)
                    or getattr(norm, "weight_function", None) or getattr(norm, "bias_function", None)
                    or type(norm.weight) is not torch.nn.Parameter
                    or norm.weight.device != q.device or norm.weight.dtype != q.dtype):
                return None
            scales.append(norm.weight)
            eps.append(norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps)
        if eps[0] != eps[1]:
            return None
        kwargs = dict(q=q, k=k, freqs_cis=pe, q_scale=scales[0], k_scale=scales[1], epsilon=eps[0])
        try:
            if registry.get_capable_backend("rms_rope", kwargs) not in ("cuda", "triton"):
                return None
        except NoCapableBackendError:
            return None
        return ck.rms_rope(q, k, pe, scales[0], scales[1], eps[0])


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, proj_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


def _shard_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, ModulationOut):
        return ModulationOut(_shard_to(value.shift, device), _shard_to(value.scale, device), _shard_to(value.gate, device))
    if isinstance(value, tuple):
        return tuple(_shard_to(item, device) for item in value)
    return value


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, bias=True):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=bias)

    def forward(self, vec: torch.Tensor) -> tuple["ModulationOut"]:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        out = self.lin(nn.functional.silu(vec)).chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


def apply_norm_scale_shift(norm, tensor, scale, shift, modulation_dims=None):
    if modulation_dims is None and shift is not None:
        return norm_affine(norm, tensor, scale, shift)
    return apply_mod(norm(tensor), 1 + scale, shift, modulation_dims)


def apply_mod(tensor: torch.Tensor, m_mult: float, m_add: bool = None, modulation_dims: torch.Tensor = None) -> torch.Tensor:
    if modulation_dims is None:
        if m_add is not None:
            return torch.addcmul(m_add, tensor, m_mult)
        else:
            return tensor * m_mult
    else:
        for d in modulation_dims:
            tensor[:, d[0] : d[1]] *= m_mult[:, d[2]]
            if m_add is not None:
                tensor[:, d[0] : d[1]] += m_add[:, d[2]]
        return tensor


def gated_residual_(x, output, gate, modulation_dims=None):
    if (single_gpu_flag("fused_flux_residual") and modulation_dims is None
            and type(x) is torch.Tensor and type(output) is torch.Tensor
            and x.is_cuda and not torch.is_grad_enabled()
            and x.dtype == output.dtype == gate.dtype
            and x.device == output.device == gate.device
            and x.shape == output.shape):
        # Fused rounding is intentional and opt-in. Keep segmented modulation
        # and mixed-dtype arithmetic on the original expression.
        x.addcmul_(output, gate)
    else:
        x += apply_mod(output, gate, None, modulation_dims)
    return x


class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return silu_gate(x1, x2) if can_fuse_silu(self.gate_fn, x1) else self.gate_fn(x1) * x2


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, modulation=True, mlp_silu_act=False, proj_bias=True, yak_mlp=False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.modulation = modulation

        if self.modulation:
            self.img_mod = Modulation(hidden_size, double=True)

        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.img_mlp = build_mlp(hidden_size, mlp_hidden_dim, mlp_silu_act=mlp_silu_act, yak_mlp=yak_mlp)

        if self.modulation:
            self.txt_mod = Modulation(hidden_size, double=True)

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.txt_mlp = build_mlp(hidden_size, mlp_hidden_dim, mlp_silu_act=mlp_silu_act, yak_mlp=yak_mlp)

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor, attn_mask=None, modulation_dims_img=None, modulation_dims_txt=None, transformer_options={}):
        if self.modulation:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
        else:
            (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

        # prepare image for attention
        img_modulated = apply_norm_scale_shift(self.img_norm1, img, img_mod1.scale, img_mod1.shift, modulation_dims_img)
        img_qkv = self.img_attn.qkv(img_modulated)
        del img_modulated
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del img_qkv
        img_pair = (self.img_attn.norm.try_fused_rope(img_q, img_k, img_v, pe[:, :, txt.shape[1]:], transformer_options)
                    if single_gpu_flag("flux_fused_qk_rope") and pe is not None and type(self.img_attn.norm) is QKNorm else None)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v) if img_pair is None else img_pair

        # prepare txt for attention
        txt_modulated = apply_norm_scale_shift(self.txt_norm1, txt, txt_mod1.scale, txt_mod1.shift, modulation_dims_txt)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        del txt_modulated
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del txt_qkv
        txt_pair = (self.txt_attn.norm.try_fused_rope(txt_q, txt_k, txt_v, pe[:, :, :txt.shape[1]], transformer_options)
                    if single_gpu_flag("flux_fused_qk_rope") and pe is not None and type(self.txt_attn.norm) is QKNorm else None)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v) if txt_pair is None else txt_pair

        if img_pair is not None or txt_pair is not None:
            # Exactly one rotation per stream, even if only one fused kernel is eligible.
            if img_pair is None:
                img_q, img_k = ck.apply_rope(img_q, img_k, pe[:, :, txt.shape[1]:])
            if txt_pair is None:
                txt_q, txt_k = ck.apply_rope(txt_q, txt_k, pe[:, :, :txt.shape[1]])
            pe = None
        del img_pair, txt_pair

        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v
        # run actual attention
        attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
        del q, k, v

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img = gated_residual_(img, self.img_attn.proj(img_attn), img_mod1.gate, modulation_dims_img)
        del img_attn
        img = gated_residual_(img, run_native_mlp(self.img_mlp, apply_norm_scale_shift(self.img_norm2, img, img_mod2.scale, img_mod2.shift, modulation_dims_img)), img_mod2.gate, modulation_dims_img)

        # calculate the txt blocks
        txt = gated_residual_(txt, self.txt_attn.proj(txt_attn), txt_mod1.gate, modulation_dims_txt)
        del txt_attn
        txt = gated_residual_(txt, run_native_mlp(self.txt_mlp, apply_norm_scale_shift(self.txt_norm2, txt, txt_mod2.scale, txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, modulation_dims_txt)

        txt = fp16_fix(txt)

        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_scale: float = None, modulation=True, mlp_silu_act=False, bias=True, yak_mlp=False):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp_hidden_dim_first = self.mlp_hidden_dim
        self.yak_mlp = yak_mlp
        if mlp_silu_act:
            self.mlp_hidden_dim_first = int(hidden_size * mlp_ratio * 2)
            self.mlp_act = SiLUActivation()
        else:
            self.mlp_act = nn.GELU(approximate="tanh")

        if self.yak_mlp:
            self.mlp_hidden_dim_first *= 2
            self.mlp_act = nn.SiLU()

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim_first, bias=bias)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=bias)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if modulation:
            self.modulation = Modulation(hidden_size, double=False)
        else:
            self.modulation = None

    def forward(self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor, attn_mask=None, modulation_dims=None, transformer_options={}) -> torch.Tensor:
        if self.modulation:
            mod, _ = self.modulation(vec)
        else:
            mod = vec

        qkv, mlp = torch.split(self.linear1(apply_norm_scale_shift(self.pre_norm, x, mod.scale, mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del qkv
        pair = self.norm.try_fused_rope(q, k, v, pe, transformer_options) if type(self.norm) is QKNorm else None
        if pair is None:
            q, k = self.norm(q, k, v)
        else:
            q, k = pair
            pe = None
        del pair

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
        del q, k, v
        # compute activation in mlp stream, cat again and run second linear layer
        if self.yak_mlp:
            gate, up = mlp[..., self.mlp_hidden_dim_first // 2 :], mlp[..., : self.mlp_hidden_dim_first // 2]
            mlp = silu_gate(gate, up) if can_fuse_silu(self.mlp_act, gate) else self.mlp_act(gate) * up
        else:
            mlp = self.mlp_act(mlp)
        output = self.linear2(torch.cat((attn, mlp), 2))
        x = gated_residual_(x, output, mod.gate, modulation_dims)

        x = fp16_fix(x)

        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, bias=True):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=bias))

    def forward(self, x: torch.Tensor, vec: torch.Tensor, modulation_dims=None) -> torch.Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]

        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = apply_norm_scale_shift(self.norm_final, x, scale, shift, modulation_dims)
        x = self.linear(x)
        return x


class IntegratedFluxTransformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        vec_in_dim: int,
        context_in_dim: int,
        hidden_size: int,
        mlp_ratio: float,
        num_heads: int,
        depth: int,
        depth_single_blocks: int,
        axes_dim: list[int],
        theta: int,
        patch_size: int,
        qkv_bias: bool,
        guidance_embed: bool,
        txt_ids_dims: list[int] = [],
        global_modulation: bool = False,
        mlp_silu_act: bool = False,
        ops_bias: bool = True,
        default_ref_method: str = "offset",
        ref_index_scale: float = 1.0,
        yak_mlp: bool = False,
        txt_norm: bool = False,
    ):
        super().__init__()

        self.guidance_embed = guidance_embed
        self.patch_size = patch_size
        self.in_channels = in_channels * patch_size * patch_size
        self.out_channels = out_channels * patch_size * patch_size

        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")

        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.axes_dim = axes_dim

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=ops_bias)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, bias=ops_bias)
        if vec_in_dim is not None:
            self.vec_in_dim = vec_in_dim
            self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, bias=ops_bias) if guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size, bias=ops_bias)

        self.txt_norm = nn.RMSNorm(context_in_dim) if txt_norm else None

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    modulation=global_modulation is False,
                    mlp_silu_act=mlp_silu_act,
                    proj_bias=ops_bias,
                    yak_mlp=yak_mlp,
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    modulation=global_modulation is False,
                    mlp_silu_act=mlp_silu_act,
                    bias=ops_bias,
                    yak_mlp=yak_mlp,
                )
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, bias=ops_bias)

        if global_modulation:
            self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, bias=False)
            self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, bias=False)
            self.single_stream_modulation = Modulation(self.hidden_size, double=False, bias=False)

        self.txt_ids_dims = txt_ids_dims
        self.global_modulation = global_modulation
        self.default_ref_method = default_ref_method
        self.ref_index_scale = ref_index_scale

    def forward_orig(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor = None,
        control=None,
        transformer_options: dict[str, list] = {},
        attn_mask: torch.Tensor = None,
        position_geometry=None,
        projected_img=False,
        target_tokens=None,
    ) -> torch.Tensor:

        patches = transformer_options.get("patches", {})
        patches_replace = transformer_options.get("patches_replace", {})
        shard = getattr(self, "_anima_shard_devices", None)
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        if not projected_img:
            img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if hasattr(self, "vec_in_dim"):
            if y is None:
                y = torch.zeros((img.shape[0], self.vec_in_dim), device=img.device, dtype=img.dtype)
            vec = vec + self.vector_in(y[:, : self.vec_in_dim])

        def project_text():
            features = self.txt_norm(txt) if self.txt_norm is not None else txt
            return self.txt_in(features)

        # DoubleStreamBlock updates its text stream in-place. The cached
        # projection must never be exposed to those writes.
        txt = cached_invariant(self, "flux_text_projection", txt, transformer_options,
                               project_text, clone=True, geometry=(img.dtype, img.device),
                               dependencies=(self.txt_norm, self.txt_in))

        vec_orig = vec
        if self.global_modulation:
            vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(vec_orig))

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids})
                img = out["img"]
                txt = out["txt"]
                img_ids = out["img_ids"]
                txt_ids = out["txt_ids"]

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            if position_geometry is not None and not patches and not patches_replace:
                pe = cached_invariant(self.pe_embedder, "flux_rope", ids, transformer_options,
                                      lambda: self.pe_embedder(ids),
                                      geometry=(position_geometry, tuple(self.pe_embedder.axes_dim), self.pe_embedder.theta),
                                      geometry_only=True)
            else:
                pe = self.pe_embedder(ids)
        else:
            pe = None

        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.double_blocks)
        transformer_options["block_type"] = "double"
        with weight_prefetch_scope(img.device, transformer_options, enabled=not shard and not patches and not patches_replace and control is None and not self.training) as prefetch:
            for i, block in enumerate(self.double_blocks):
                if prefetch is not None:
                    prefetch.stage(self.double_blocks[i + 1] if i + 1 < len(self.double_blocks) else None, img.dtype)
                if shard and i == shard[2]:
                    img, txt, vec, pe, attn_mask = (_shard_to(v, shard[1]) for v in (img, txt, vec, pe, attn_mask))
                transformer_options["block_index"] = i
                if ("double_block", i) in blocks_replace:

                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"), transformer_options=args.get("transformer_options"))
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attn_mask": attn_mask, "transformer_options": transformer_options}, {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    with torch.cuda.device(img.device) if shard else nullcontext():
                        img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

                if control is not None:  # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img[:, : add.shape[1]] += add.to(img.device) if shard else add
                if prefetch is not None:
                    prefetch.discard(block)

        img = fp16_fix(img)
        img = torch.cat((txt, img), 1)

        if self.global_modulation:
            vec, _ = self.single_stream_modulation(vec_orig)
            if shard:
                vec = _shard_to(vec, img.device)

        transformer_options["total_blocks"] = len(self.single_blocks)
        transformer_options["block_type"] = "single"
        with weight_prefetch_scope(img.device, transformer_options, enabled=not shard and not patches and not patches_replace and control is None and not self.training) as prefetch:
            for i, block in enumerate(self.single_blocks):
                if prefetch is not None:
                    prefetch.stage(self.single_blocks[i + 1] if i + 1 < len(self.single_blocks) else None, img.dtype)
                if shard and len(self.double_blocks) + i == shard[2]:
                    img, vec, pe, attn_mask = (_shard_to(v, shard[1]) for v in (img, vec, pe, attn_mask))
                transformer_options["block_index"] = i
                if ("single_block", i) in blocks_replace:

                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"), transformer_options=args.get("transformer_options"))
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attn_mask": attn_mask, "transformer_options": transformer_options}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    with torch.cuda.device(img.device) if shard else nullcontext():
                        img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

                if control is not None:  # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] : txt.shape[1] + add.shape[1], ...] += add.to(img.device) if shard else add
                if prefetch is not None:
                    prefetch.discard(block)

        img = img[:, txt.shape[1] :, ...]

        if shard:
            img = img.to(shard[0])

        if (target_tokens is not None and target_tokens < img.shape[1]
                and single_gpu_flag("target_only_output_head")
                and type(self) is IntegratedFluxTransformer2DModel
                and type(self.final_layer) is LastLayer and not self.training
                and not torch.is_grad_enabled() and not patches and not patches_replace and control is None
                and transformer_options.keys() <= {"cond_or_uncond", "sigmas", "cond_mark", "cond_indices",
                                                   "uncond_indices", "sampling_sigmas", "total_blocks", "block_type", "block_index"}
                and (vec_orig.ndim == 2 or (vec_orig.ndim == 3 and vec_orig.shape[1] == 1))
                and dense_tokenwise_head(self.final_layer)):
            img = img[:, :target_tokens]
        return self.final_layer(img, vec_orig)  # (N, T, patch_size ** 2 * out_channels)

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        bs, c, h, w = x.shape
        patch_size = self.patch_size
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size

        h_offset = (h_offset + (patch_size // 2)) // patch_size
        w_offset = (w_offset + (patch_size // 2)) // patch_size

        steps_h = h_len
        steps_w = w_len

        img_ids = torch.zeros((steps_h, steps_w, len(self.axes_dim)), device=x.device, dtype=torch.float32)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=steps_h, device=x.device, dtype=torch.float32).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=steps_w, device=x.device, dtype=torch.float32).unsqueeze(0)
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    def forward(self, x, timestep, context, y=None, guidance=None, control=None, transformer_options={}, **kwargs):
        bs, c, h_orig, w_orig = x.shape
        patch_size = self.patch_size

        h_len = (h_orig + (patch_size // 2)) // patch_size
        w_len = (w_orig + (patch_size // 2)) // patch_size
        img, img_ids = self.process_img(x)
        img_tokens = img.shape[1]

        ref_latents: list[torch.Tensor] = dynamic_args.ref_latents

        reference_key = None
        project_refs = False
        if (ref_latents and type(self) is IntegratedFluxTransformer2DModel
                and single_gpu_flag("flux_reference_cache") and invariant_scope_active()
                and invariants_enabled(self, x, transformer_options) and control is None):
            reference_key = reference_signature(ref_latents)
            if reference_key is not None and retain_reference_sources(ref_latents):
                from backend.operations import ForgeOperations
                project_refs = (type(self.img_in) in (nn.Linear, ForgeOperations.Linear)
                                and dense_tokenwise_head(self.img_in)
                                and self.img_in.weight.device == x.device)
            else:
                reference_key = None
        if bool(ref_latents):
            img_parts = [self.img_in(img) if project_refs else img]
            id_parts = [img_ids]
            h = 0
            w = 0
            index = 0
            ref_latents_method = kwargs.get("ref_latents_method", self.default_ref_method)
            for ref_number, ref in enumerate(ref_latents):
                if ref_latents_method == "index":
                    index += self.ref_index_scale
                    h_offset = 0
                    w_offset = 0
                elif ref_latents_method == "uxo":
                    index = 0
                    h_offset = h_len * patch_size + h
                    w_offset = w_len * patch_size + w
                    h += ref.shape[-2]
                    w += ref.shape[-1]
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                def prepare_ref():
                    tokens, ids = self.process_img(ref.to(x), index=index, h_offset=h_offset, w_offset=w_offset)
                    return (self.img_in(tokens) if project_refs else tokens), ids

                if reference_key is None:
                    kontext, kontext_ids = prepare_ref()
                else:
                    kontext, kontext_ids = cached_invariant(
                        self, "flux_reference_tokens", ref, transformer_options, prepare_ref,
                        geometry=(reference_key[ref_number], x.device, x.dtype, index, h_offset, w_offset,
                                  tuple(self.axes_dim), patch_size, project_refs),
                        dependencies=(self.img_in,) if project_refs else ())
                if img.size(0) > 1:
                    kontext = kontext.expand(img.shape[0], *kontext.shape[1:])
                img_parts.append(kontext)
                if img_ids.size(0) > 1:
                    kontext_ids = kontext_ids.expand(img_ids.shape[0], *kontext_ids.shape[1:])
                id_parts.append(kontext_ids)
            img = torch.cat(img_parts, dim=1)
            img_ids = torch.cat(id_parts, dim=1)

        txt_ids = torch.zeros((bs, context.shape[1], len(self.axes_dim)), device=x.device, dtype=torch.float32)

        if len(self.txt_ids_dims) > 0:
            for i in self.txt_ids_dims:
                txt_ids[:, :, i] = torch.linspace(0, context.shape[1] - 1, steps=context.shape[1], device=x.device, dtype=torch.float32)

        geometry = (h_len, w_len, context.shape[1], tuple(self.txt_ids_dims), patch_size)
        if ref_latents:
            geometry = None if reference_key is None else (*geometry, reference_key, ref_latents_method, self.ref_index_scale)
        # Preserve custom subclass/extension signatures when neither new path is active.
        extra = {}
        if project_refs:
            extra["projected_img"] = True
        if ref_latents and single_gpu_flag("target_only_output_head") and type(self) is IntegratedFluxTransformer2DModel:
            extra["target_tokens"] = img_tokens
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control, transformer_options,
                                attn_mask=kwargs.get("attention_mask", None), position_geometry=geometry, **extra)
        out = out[:, :img_tokens]

        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=self.patch_size, pw=self.patch_size)[:, :, :h_orig, :w_orig]
