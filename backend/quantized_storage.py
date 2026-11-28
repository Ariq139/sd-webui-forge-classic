"""Version-pinned storage adapters for supported packed weight wrappers."""
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError

import torch

from backend.packed_weight_identity import TransferSource


@lru_cache(maxsize=1)
def _layout_types():
    try:
        if version("comfy-kitchen") != "0.2.40":
            return None
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
    except (ImportError, PackageNotFoundError):
        return None
    return QuantizedTensor, TensorWiseINT8Layout.Params


def _full_cpu_storage(t):
    return (type(t) is torch.Tensor and t.device.type == "cpu" and t.is_contiguous()
            and t.storage_offset() == 0 and t.numel() > 0
            and t.numel() * t.element_size() == t.untyped_storage().nbytes())


def _version(t):
    try:
        return t._version
    except RuntimeError:
        return None


@dataclass(frozen=True, eq=False)
class Int8Storage:
    source: object
    parts: tuple
    metadata: tuple
    versions: tuple
    stamps: tuple

    @property
    def transfer_parts(self):
        return tuple(stamp.raw for stamp in self.stamps)

    @property
    def orig_dtype(self):
        return self.metadata[0]

    def matches(self, source):
        if self.source is not source or None in self.versions:
            return False
        other = int8_storage(source)
        return (other is not None and self.metadata == other.metadata
                and all(a is b for a, b in zip(self.parts, other.parts))
                and self.versions == other.versions
                and all(stamp.matches(part, part) for stamp, part in zip(self.stamps, other.parts)))

    def rebuild(self, parts):
        cls, _ = _layout_types()
        dtype, shape, is_weight, convrot, group_size, transposed = self.metadata
        params = _layout_types()[1](scale=parts[1], orig_dtype=dtype, orig_shape=shape,
                                    is_weight=is_weight, convrot=convrot,
                                    convrot_groupsize=group_size, transposed=transposed)
        return cls(parts[0], "TensorWiseINT8Layout", params)


def int8_storage(source):
    if type(source).__name__ != "QuantizedTensor":
        return None
    types = _layout_types()
    if types is None or type(source) is not types[0]:
        return None
    if source._layout_cls != "TensorWiseINT8Layout":
        return None
    p = source._params
    if (type(p) is not types[1] or not p.is_weight or p.transposed
            or p.orig_dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        return None
    q, s = source._qdata, p.scale
    if (not _full_cpu_storage(q) or not _full_cpu_storage(s)
            or q.dtype != torch.int8 or q.ndim != 2 or s.dtype != torch.float32
            or tuple(q.shape) != tuple(p.orig_shape) or s.numel() not in (1, q.shape[0])):
        return None
    return Int8Storage(source, (q, s),
                       (p.orig_dtype, tuple(p.orig_shape), p.is_weight, p.convrot,
                        p.convrot_groupsize, p.transposed),
                       (_version(source), _version(q), _version(s)),
                       tuple(TransferSource(part, part) for part in (q, s))
                       if _version(q) is not None and _version(s) is not None else ())


@lru_cache(maxsize=1)
def _extra_layout_types():
    try:
        if version("comfy-kitchen") != "0.2.40":
            return None
        from comfy_kitchen.tensor import (
            QuantizedTensor, TensorCoreFP8Layout, TensorCoreAWQW4A16Layout,
            TensorCoreConvRotW4A4Layout, AsymW4A8Int8Layout,
        )
    except (ImportError, PackageNotFoundError):
        return None
    return QuantizedTensor, {
        "TensorCoreFP8Layout": TensorCoreFP8Layout.Params,
        "TensorCoreFP8E4M3Layout": TensorCoreFP8Layout.Params,
        "TensorCoreFP8E5M2Layout": TensorCoreFP8Layout.Params,
        "TensorCoreAWQW4A16Layout": TensorCoreAWQW4A16Layout.Params,
        "TensorCoreConvRotW4A4Layout": TensorCoreConvRotW4A4Layout.Params,
        "AsymW4A8Int8Layout": AsymW4A8Int8Layout.Params,
    }


def _layout_fields(layout, q, p):
    """Explicitly admit known, untransposed full-storage checkpoint layouts."""
    n, k = p.orig_shape
    if min(n, k) <= 0 or q.ndim != 2 or getattr(p, "transposed", False):
        return None
    if layout.startswith("TensorCoreFP8"):
        expected = torch.float8_e5m2 if layout.endswith("E5M2Layout") else torch.float8_e4m3fn
        if q.dtype == expected and tuple(q.shape) == (n, k) and p.scale.dtype == torch.float32 and p.scale.numel() == 1:
            return ("scale",), ()
    elif layout == "TensorCoreAWQW4A16Layout":
        g = p.group_size
        if (type(g) is int and g > 0 and k % 2 == k % g == 0 and q.dtype == torch.int8
                and tuple(q.shape) == (n, k // 2)
                and tuple(p.scale.shape) == tuple(p.zeros.shape) == (k // g, n)
                and p.scale.dtype == p.zeros.dtype and p.scale.dtype in (torch.float16, torch.bfloat16)):
            return ("scale", "zeros"), (("group_size", g), ("transposed", False))
    elif layout == "TensorCoreConvRotW4A4Layout":
        g, r = p.quant_group_size, p.convrot_groupsize
        if (type(g) is int and g == 64 and type(r) is int and r in (16, 64, 256)
                and k % g == k % r == 0 and q.dtype == torch.int8 and tuple(q.shape) == (n, k // 2)
                and p.scale.dtype == torch.float32 and p.scale.numel() in (n, n * (k // g))
                and p.linear_dtype in ("int4", "int8")):
            return ("scale",), (("quant_group_size", g), ("convrot_groupsize", r),
                                ("linear_dtype", p.linear_dtype), ("transposed", False))
    elif layout == "AsymW4A8Int8Layout":
        g, r = p.group_size, p.convrot_groupsize
        if (not p.mma_packed and type(g) is int and g >= 4 and type(r) is int and r > 0
                and k % 16 == k % g == k % r == 0 and (16 % g == 0 or g % 16 == 0)
                and q.dtype == torch.int8 and q.shape[0] == n and q.shape[1] * 8 in (4 * k, 6 * k)
                and tuple(p.scale.shape) == (n, k // g) and p.scale.dtype in (torch.float8_e4m3fn, torch.float32)
                and type(p.s_channel) is torch.Tensor and tuple(p.s_channel.shape) == (n,)
                and p.s_channel.dtype == torch.float32):
            names = ["scale", "s_channel"]
            optional = []
            for field, shape in (("correction", (k // g, n)), ("codebook", (16,))):
                value = getattr(p, field)
                if value is None:
                    optional.append((field, None))
                elif type(value) is torch.Tensor and tuple(value.shape) == shape and value.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    names.append(field)
                else:
                    return None
            return tuple(names), (("group_size", g), ("convrot_groupsize", r),
                                  ("mma_packed", False), ("transposed", False), *optional)
    return None


@dataclass(frozen=True, eq=False)
class PackedStorage:
    source: object
    layout: str
    params_type: type
    field_names: tuple
    parts: tuple
    metadata: tuple
    versions: tuple
    stamps: tuple

    @property
    def orig_dtype(self):
        return dict(self.metadata)["orig_dtype"]

    @property
    def transfer_parts(self):
        return tuple(stamp.raw for stamp in self.stamps)

    def matches(self, source):
        if self.source is not source:
            return False
        other = packed_storage(source)
        return (type(other) is PackedStorage and self.layout == other.layout
                and self.params_type is other.params_type and self.field_names == other.field_names
                and self.metadata == other.metadata and self.versions == other.versions
                and all(a is b for a, b in zip(self.parts, other.parts))
                and all(stamp.matches(part, part) for stamp, part in zip(self.stamps, other.parts)))

    def rebuild(self, parts):
        # Use captured metadata rather than reading potentially replaced params.
        params = self.params_type(**dict(self.metadata), **dict(zip(self.field_names, parts[1:])))
        return type(self.source)(parts[0], self.layout, params)


def packed_storage(source):
    """Snapshot supported original packed storage, never dequantized weights."""
    if type(source).__name__ != "QuantizedTensor":
        return None
    base = _layout_types()
    if base is not None and type(source) is base[0] and source._layout_cls == "TensorWiseINT8Layout":
        return int8_storage(source)
    types = _extra_layout_types()
    if types is None or type(source) is not types[0]:
        return None
    layout, p, q = source._layout_cls, source._params, source._qdata
    if (type(p) is not types[1].get(layout) or not _full_cpu_storage(q)
            or p.orig_dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or len(p.orig_shape) != 2):
        return None
    try:
        spec = _layout_fields(layout, q, p)
        if spec is None:
            return None
        names, metadata = spec
        parts = (q, *(getattr(p, name) for name in names))
        if not all(_full_cpu_storage(t) for t in parts):
            return None
        versions = tuple(_version(t) for t in (source, *parts))
        if None in versions:
            return None
        metadata = (("orig_dtype", p.orig_dtype), ("orig_shape", tuple(p.orig_shape)), *metadata)
        # Reject any new or unexpected parameter fields rather than dropping them.
        from dataclasses import fields
        if {f.name for f in fields(p)} != set(names) | {name for name, _ in metadata}:
            return None
        return PackedStorage(source, layout, type(p), names, parts, metadata, versions,
                             tuple(TransferSource(t, t) for t in parts))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
