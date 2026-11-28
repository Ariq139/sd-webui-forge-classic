# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/ldm/krea2/model.py
# https://github.com/lbouaraba/comfyui-krea2edit/blob/main/__init__.py

from collections import OrderedDict
from threading import RLock
from typing import Optional
from contextlib import nullcontext
from weakref import WeakKeyDictionary

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from backend.args import dynamic_args
from backend.attention import attention_function, attention_pytorch
from backend.memory_management import cast_to
from backend.misc.image_resize import adaptive_resize
from backend.nn.flux import EmbedND, timestep_embedding, run_native_mlp
from backend.operations import repeat_kv_for_gqa
from backend.performance import record_region
from backend.quant_ops import ck
from backend.weight_prefetch import weight_prefetch_scope
from backend.activation_quantization import shared_activation_scope
from backend.fused_pointwise import silu_gate, fusion_enabled
from backend.sampling.krea_cache import cached_krea_text_features
from backend.sampling.invariant_cache import cached_invariant, cache_dependencies_safe, invariants_enabled, retain_reference_sources
from backend.single_gpu_fastpaths import single_gpu_flag, reference_signature
from backend.operations_mixed_precision import try_native_int8_input_act
from backend.native_gqa import try_flash_gqa
from backend.gate_up_packing import try_packed_gate_up
from backend.utils import pad_to_patch_size


_KREA_POSITION_CACHE = WeakKeyDictionary()
_KREA_POSITION_CACHE_LOCK = RLock()
_KREA_POSITION_CACHE_MAX_BYTES = 8 * 1024 * 1024


def _imgids(bs: int, frame: int, h_: int, w_: int, device: torch.device) -> torch.Tensor:
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    if frame:
        ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    # The IDs are read only and immediately concatenated into a new tensor.
    return ids.reshape(1, h_ * w_, 3).expand(bs, -1, -1)


def clear_krea_position_cache(embedder=None):
    """Release cached Krea positional embeddings, optionally for one embedder."""
    with _KREA_POSITION_CACHE_LOCK:
        if embedder is None:
            _KREA_POSITION_CACHE.clear()
        else:
            _KREA_POSITION_CACHE.pop(embedder, None)


def krea_position_frequencies(embedder, text_length, height, width, ref_grids, device):
    ref_grids = tuple(tuple(int(axis) for axis in grid) for grid in ref_grids)
    key = (device.type, device.index, int(text_length), int(height), int(width), ref_grids)

    def compute():
        text_ids = torch.zeros(1, text_length, 3, device=device, dtype=torch.float32)
        if ref_grids:
            ref_ids = [_imgids(1, index + 1, grid_h, grid_w, device) for index, (grid_h, grid_w) in enumerate(ref_grids)]
            image_ids = _imgids(1, 0, height, width, device)
            position_ids = torch.cat((text_ids, *ref_ids, image_ids), dim=1)
        else:
            image_ids = _imgids(1, 0, height, width, device)
            position_ids = torch.cat((text_ids, image_ids), dim=1)
        return embedder(position_ids)

    if torch.is_grad_enabled():
        return compute()

    with _KREA_POSITION_CACHE_LOCK:
        cache = _KREA_POSITION_CACHE.get(embedder)
        cached = cache["entries"].get(key) if cache is not None else None
        if cached is not None:
            cache["entries"].move_to_end(key)
            with record_region("forge::krea_position_cache_hit"):
                return cached

    frequencies = compute()
    size = frequencies.numel() * frequencies.element_size()
    if size <= 0 or size > _KREA_POSITION_CACHE_MAX_BYTES:
        return frequencies

    with _KREA_POSITION_CACHE_LOCK:
        cache = _KREA_POSITION_CACHE.setdefault(embedder, {"entries": OrderedDict(), "bytes": 0})
        cached = cache["entries"].get(key)
        if cached is not None:
            cache["entries"].move_to_end(key)
            with record_region("forge::krea_position_cache_hit"):
                return cached
        while cache["entries"] and cache["bytes"] + size > _KREA_POSITION_CACHE_MAX_BYTES:
            _, evicted = cache["entries"].popitem(last=False)
            cache["bytes"] -= evicted.numel() * evicted.element_size()
        cache["entries"][key] = frequencies
        cache["bytes"] += size
    return frequencies


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.empty(features))

    def fp32_weight(self, x):
        return cast_to(self.scale, dtype=torch.float32, device=x.device) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), weight=self.fp32_weight(x).to(x.dtype), eps=self.eps)

    def modulated(self, x, scale, shift):
        if torch.is_grad_enabled():
            return torch.addcmul(shift, 1 + scale, self(x))
        # Fold the learned RMSNorm weight into the AdaLN scale before rounding.
        scale = (self.fp32_weight(x) * (1 + scale.float()) - 1).to(x.dtype)
        return ck.rms_adaln(x, scale, shift, self.eps)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x):
        with shared_activation_scope(x, (self.gate, self.up)):
            pair = try_packed_gate_up(x, self.gate, self.up)
            gate = self.gate(x) if pair is None else pair[0]
            def up_projection():
                return self.up(x) if pair is None else pair[1]
            if (single_gpu_flag("native_int8_mlp_activation") and gate.dtype == torch.bfloat16
                    and type(self) is SwiGLU and cache_dependencies_safe(self)):
                up = up_projection()
                fused = try_native_int8_input_act(self.down, (gate, up), "swiglu")
                if fused is not None:
                    return fused
                hidden = silu_gate(gate, up) if fusion_enabled(gate) else F.silu(gate).mul_(up)
            elif fusion_enabled(gate):
                hidden = silu_gate(gate, up_projection())
            else:
                hidden = F.silu(gate).mul_(up_projection())
        return self.down(hidden)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: Optional[int] = None, bias: bool = False):
        super().__init__()

        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        with shared_activation_scope(x, (self.wq, self.wk, self.wv, self.gate)):
            q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = q.unflatten(-1, (self.heads, self.headdim))
        k = k.unflatten(-1, (self.kvheads, self.headdim))
        v = v.unflatten(-1, (self.kvheads, self.headdim))
        if freqs is not None and not torch.is_grad_enabled():
            qnorm, knorm = self.qknorm.qnorm, self.qknorm.knorm
            q, k = ck.rms_rope_(q, k, freqs.transpose(1, 2), qnorm.fp32_weight(q), knorm.fp32_weight(k), qnorm.eps)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
        else:
            q, k = self.qknorm(q.transpose(1, 2), k.transpose(1, 2))
            if freqs is not None:
                q, k = ck.apply_rope(q, k, freqs)
        v = v.transpose(1, 2)
        out = None
        if (type(self) is Attention and single_gpu_flag("native_krea_gqa")
                and attention_function is attention_pytorch):
            out = try_flash_gqa(q, k, v, mask, transformer_options)
        if out is None:
            k, v = repeat_kv_for_gqa(k, v, self.heads, 1)
            out = attention_function(q, k, v, self.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
        return self.wo(out * F.sigmoid(gate))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.lin = nn.Parameter(torch.empty(2, dim))

    def forward(self, vec):
        out = vec + cast_to(self.lin, dtype=vec.dtype, device=vec.device).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.lin = nn.Parameter(torch.empty(6 * dim))

    def forward(self, vec):
        out = vec + cast_to(self.lin, dtype=vec.dtype, device=vec.device)
        return out.chunk(6, dim=-1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, mask=None, transformer_options={}):
        x.add_(self.attn(self.prenorm(x), mask=mask, transformer_options=transformer_options))
        x.add_(self.mlp(self.postnorm(x)))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.layerwise_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])

    def forward(self, x, mask=None, transformer_options={}):
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None, transformer_options=transformer_options)
        x = self.projector(x.unflatten(0, (b, l)).transpose(2, 3)).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask, transformer_options=transformer_options)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, vec, freqs, mask=None, transformer_options={}):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x.addcmul_(pregate, self.attn(self.prenorm.modulated(x, prescale, preshift), freqs, mask, transformer_options=transformer_options))
        x.addcmul_(postgate, self.mlp(self.postnorm.modulated(x, postscale, postshift)))
        return x


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels):
        super().__init__()

        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        return self.linear(self.norm.modulated(x, scale, shift))


class SingleStreamDiT(nn.Module):
    def __init__(
        self,
        features=6144,
        tdim=256,
        txtdim=2560,
        heads=48,
        kvheads=12,
        multiplier=4,
        layers=28,
        patch=2,
        channels=16,
        bias=False,
        theta=1e3,
        txtlayers=12,
        txtheads=20,
        txtkvheads=20,
        **kwargs,
    ):
        super().__init__()

        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        headdim = features // heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"axes {axes} sum != headdim {headdim}"
        self.pe_embedder = EmbedND(dim=headdim, theta=int(theta), axes_dim=axes)

        self.first = nn.Linear(channels * patch**2, features, bias=True)
        self.blocks = nn.ModuleList([SingleStreamBlock(features, heads, multiplier, bias, kvheads) for _ in range(layers)])
        self.tmlp = nn.Sequential(
            nn.Linear(tdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim),
            nn.Linear(txtdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.last = LastLayer(features, patch, channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features * 6),
        )

    def forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
        x = x.squeeze(2)
        bs, c, H_orig, W_orig = x.shape
        patch = self.patch

        x = pad_to_patch_size(x, (patch, patch))
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        ref_latents: list[torch.Tensor] = dynamic_args.ref_latents

        # Preserve the ordinary reference path, including resize and projection
        # order, unless the native immutable reference contract is established.
        _edit = bool(ref_latents)
        ref_key = None
        if (_edit and type(self) is SingleStreamDiT and single_gpu_flag("krea_reference_cache")
                and invariants_enabled(self, x, transformer_options)
                and cache_dependencies_safe(self.first)
                and not getattr(self.first, "weight_function", None)
                and not getattr(self.first, "bias_function", None)):
            ref_key = reference_signature(ref_latents)
            if ref_key is not None and not retain_reference_sources(ref_latents):
                ref_key = None

        def prepare_reference(ref):
            if x.shape[0] == 2:
                ref = torch.cat((ref, ref), dim=0)
            if ref.shape != x.shape:
                ref = adaptive_resize(ref, W, H, "area", "center")
            return (pad_to_patch_size(ref.to(x), (patch, patch), padding_mode="replicate"),
                    (ref.shape[-2] // patch, ref.shape[-1] // patch))

        if _edit and ref_key is None:
            prepared = [prepare_reference(ref) for ref in ref_latents]
            refs, ref_grids = map(list, zip(*prepared))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        img = self.first(img)

        def project_reference(ref):
            return self.first(rearrange(ref, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))

        if _edit:
            if ref_key is None:
                refs = [project_reference(ref) for ref in refs]
            else:
                refs, ref_grids = [], []
                for ref, signature in zip(ref_latents, ref_key):
                    def compute():
                        prepared, grid = prepare_reference(ref)
                        return project_reference(prepared), grid
                    projected, grid = cached_invariant(
                        self, "krea_reference_projection", ref, transformer_options, compute,
                        geometry=(signature, tuple(x.shape), x.dtype, x.device, patch),
                        geometry_only=True, dependencies=(self.first,))
                    refs.append(projected)
                    ref_grids.append(grid)

        t = run_native_mlp(self.tmlp, timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)

        def compute_text_features():
            with record_region("forge::krea_text_fusion"):
                # Text fusion has in-place residuals; preserve cached input ownership.
                source = context.clone() if type(self) is SingleStreamDiT and single_gpu_flag("direct_full_area_cfg") else context
                fused = self.txtfusion(source, mask=None, transformer_options=transformer_options)
                return self.txtmlp(fused)

        if _edit:
            context = compute_text_features()
        else:
            context = cached_krea_text_features(self, context, transformer_options, compute_text_features,
                                                dependencies=(self.txtfusion, self.txtmlp))

        txtlen, imglen = context.shape[1], img.shape[1]
        if _edit:
            reflen = sum(ref.shape[1] for ref in refs)
            combined = torch.cat((context, *refs, img), dim=1)
        else:
            reflen = 0
            combined = torch.cat((context, img), dim=1)

        device = combined.device
        freqs = krea_position_frequencies(self.pe_embedder, txtlen, h_, w_, ref_grids if _edit else (), device)

        shard = getattr(self, "_anima_shard_devices", None)
        with weight_prefetch_scope(combined.device, transformer_options, enabled=not shard and not _edit and not self.training) as prefetch:
            for index, block in enumerate(self.blocks):
                if prefetch is not None:
                    prefetch.stage(self.blocks[index + 1] if index + 1 < len(self.blocks) else None, combined.dtype)
                if shard and index == shard[2]:
                    combined = combined.to(shard[1])
                    tvec = tvec.to(shard[1])
                    freqs = freqs.to(shard[1])
                with torch.cuda.device(combined.device) if shard else nullcontext():
                    combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)
                if prefetch is not None:
                    prefetch.discard(block)

        combined = combined[:, txtlen + reflen : txtlen + reflen + imglen]
        if shard:
            combined = combined.to(shard[0])

        out = self.last(combined, t)
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
        out = out[:, :, :H_orig, :W_orig]

        return out.unsqueeze(2)
