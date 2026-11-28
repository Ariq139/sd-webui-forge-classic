"""Bounded block look-ahead for immutable pinned CPU weights.

No module parameter is replaced. A consumed copy is owned by the current
compute stream; unconsumed copies stay owned by the staging stream.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from backend.args import args
from backend.performance import record_region

_ACTIVE = ContextVar("forge_weight_prefetch", default=None)
_SAFE_OPTIONS = frozenset({"cond_or_uncond", "sigmas", "cond_mark", "cond_indices",
                           "uncond_indices", "sampling_sigmas", "total_blocks",
                           "block_type", "block_index"})
_MIN_HEADROOM = 384 * 1024 * 1024


class WeightPrefetch:
    def __init__(self, device, budget):
        self.device = device
        self.budget = budget
        self.stream = torch.cuda.Stream(device=device)
        self.pending = {}
        self.bytes = 0

    def stage(self, block, dtype):
        if block is None:
            return
        free, _ = torch.cuda.mem_get_info(self.device)
        allowance = min(self.budget // 2, self.budget - self.bytes,
                        max(0, free - _MIN_HEADROOM))
        if allowance <= 0:
            return
        from backend.operations_gguf import ParameterGGUF

        with torch.cuda.stream(self.stream):
            for module in block.modules():
                source = getattr(module, "weight", None)
                if (not isinstance(source, torch.Tensor) or source.device.type != "cpu"
                        or getattr(module, "weight_function", None)
                        or getattr(module, "bias_function", None)
                        or getattr(module, "weight_scale", None) is not None
                        or getattr(module, "_forward_hooks", None)
                        or getattr(module, "_forward_pre_hooks", None)
                        or module in self.pending):
                    continue
                packed = (type(source) is ParameterGGUF
                          and getattr(source, "gguf_cls", None) is not None
                          and not source.baked)
                if packed:
                    raw = source.data
                    if raw.dtype != torch.uint8:
                        continue
                elif type(source) in (torch.Tensor, torch.nn.Parameter) and source.dtype == dtype:
                    raw = source
                else:
                    continue
                if not raw.is_contiguous() or not raw.is_pinned():
                    continue
                try:
                    version = source._version
                except RuntimeError:
                    continue
                size = raw.numel() * raw.element_size()
                if not size or size > allowance:
                    continue
                try:
                    with record_region("forge::weight_prefetch_copy"):
                        value = torch.empty_like(raw, device=self.device)
                        value.copy_(raw, non_blocking=True)
                        ready = torch.cuda.Event()
                        ready.record(self.stream)
                except torch.cuda.OutOfMemoryError:
                    # The ordinary cast path remains the authoritative fallback.
                    break
                self.pending[module] = (source, version, value, ready, packed, size)
                self.bytes += size
                allowance -= size

    def take(self, module, source, device, dtype):
        if device != self.device:
            return None
        entry = self.pending.pop(module, None)
        if entry is None:
            return None
        original, version, value, ready, packed, size = entry
        self.bytes -= size
        if original is not source:
            return None
        try:
            if source._version != version:
                return None
        except RuntimeError:
            return None
        if (packed and dtype is not None) or (not packed and dtype not in (None, value.dtype)):
            return None
        current = torch.cuda.current_stream(device)
        with record_region("forge::weight_prefetch_wait"):
            current.wait_event(ready)
        value.record_stream(current)
        return source.copy_with_data(value) if packed else value

    def discard(self, block):
        for module in block.modules():
            entry = self.pending.pop(module, None)
            if entry is not None:
                self.bytes -= entry[-1]

    def close(self):
        # Stream-ordered lifetime, not a host/device-wide synchronization.
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        self.pending.clear()
        self.bytes = 0


@contextmanager
def weight_prefetch_scope(device, options, enabled=True):
    budget = getattr(args, "weight_prefetch_mib", 0) * 1024 * 1024
    enabled = (enabled and budget > 0 and device.type == "cuda"
               and not torch.is_grad_enabled()
               and isinstance(options, dict) and options.keys() <= _SAFE_OPTIONS
               and not any(getattr(args, key, None) is not None
                           for key in ("multi_gpu_anima", "parallel_cfg", "parallel_batch")))
    state = WeightPrefetch(device, budget) if enabled else None
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        try:
            if state is not None:
                state.close()
        finally:
            _ACTIVE.reset(token)


def take_prefetched_weight(module, source, device, dtype):
    state = _ACTIVE.get()
    return None if state is None else state.take(module, source, device, dtype)
