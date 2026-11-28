"""Version-pinned storage adapter for ordinary weight-only INT8 wrappers."""
from dataclasses import dataclass, replace
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
        return cls(parts[0], "TensorWiseINT8Layout", replace(self.source._params, scale=parts[1]))


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
