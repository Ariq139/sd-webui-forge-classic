"""Bounded, disjoint output views and CPU-only INT8 gate/up coalescing."""
import torch

MAX_OUTPUT_BYTES = 32 * 1024**2
MAX_PACK_BYTES = 256 * 1024**2


def joined_rows(a, b):
    """Recognize two contiguous slices covering precisely one packed allocation."""
    if (type(a) is not torch.Tensor or type(b) is not torch.Tensor
            or a.ndim != 2 or b.ndim != 2 or a.device != b.device or a.dtype != b.dtype
            or not a.is_contiguous() or not b.is_contiguous() or a.shape[1] != b.shape[1]
            or not a.numel() or not b.numel() or a.storage_offset() != 0
            or b.storage_offset() != a.numel()
            or a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
            or a.untyped_storage().nbytes() != (a.numel() + b.numel()) * a.element_size()):
        return None
    return a.as_strided((a.shape[0] + b.shape[0], a.shape[1]), (a.shape[1], 1), 0)


def coalesce_rows(a, b):
    if joined_rows(a, b) is not None:
        return a, b
    if (a.device.type != "cpu" or b.device != a.device or a.dtype != torch.int8
            or b.dtype != a.dtype or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]
            or any(not t.is_contiguous() or t.storage_offset() != 0 or not t.numel()
                   or t.is_pinned() or t.untyped_storage().nbytes() != t.numel() * t.element_size()
                   for t in (a, b))
            or a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
            or a.numel() + b.numel() > MAX_PACK_BYTES):
        return None
    packed = torch.cat((a, b), dim=0)
    return packed[:a.shape[0]], packed[a.shape[0]:]


class ProjectionDestinations:
    """One forward group owns disjoint contiguous [M,N] GEMM destinations."""
    def __init__(self, source, widths, budget=MAX_OUTPUT_BYTES):
        self.source = source
        self.widths = dict(widths)
        self.offsets = {}
        self.used = set()
        self.buffer = None
        self.m = source.numel() // source.shape[-1]
        self.total = 0
        for module, width in widths:
            self.offsets[module] = self.total
            self.total += self.m * width
        self.enabled = (len(self.widths) == len(widths) and self.m > 2
                        and all(type(n) is int and n > 0 and n % 16 == 0 for n in self.widths.values())
                        and self.total * source.element_size() <= budget)

    def take(self, module):
        if not self.enabled or module not in self.widths or module in self.used:
            return None
        if self.buffer is None:
            size = self.total * self.source.element_size()
            if self.source.is_cuda and torch.cuda.mem_get_info(self.source.device)[0] < size + 384 * 1024**2:
                self.enabled = False
                return None
            try:
                self.buffer = self.source.new_empty(self.total)
            except torch.cuda.OutOfMemoryError:
                self.enabled = False
                return None
        self.used.add(module)
        n = self.widths[module]
        start = self.offsets[module]
        return self.buffer[start:start + self.m * n].view(self.m, n)
