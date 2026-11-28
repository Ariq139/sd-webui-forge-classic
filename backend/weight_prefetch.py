"""Bounded block look-ahead for immutable pinned CPU weights.

No module parameter is replaced. Copies remain charged until the staging
or compute stream has finished using them. A scope owns its CPU sources
until their asynchronous copies finish, including externally pinned memory.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from backend.args import args
from backend.packed_weight_identity import TransferSource
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
        self.stream = None
        self.last_copy = None
        self.copy_owners = None
        self.pending = {}
        self.in_use = {}
        self.retired = []
        self.bytes = 0

    def _reap(self):
        remaining = []
        for entry, event in self.retired:
            if event.query():
                self.bytes -= entry[-1]
            else:
                remaining.append((entry, event))
        self.retired = remaining

    def stage(self, block, dtype):
        if block is None:
            return
        self._reap()
        # Permit a single large packed matrix to occupy the remaining budget;
        # in-flight and retired buffers are still charged until completion.
        allowance = self.budget - self.bytes
        if allowance <= 0:
            return
        from backend.operations_gguf import ParameterGGUF

        checked_memory = False
        for module in block.modules():
            source = getattr(module, "weight", None)
            if (not isinstance(source, torch.Tensor) or source.device.type != "cpu"
                    or getattr(module, "weight_function", None)
                    or getattr(module, "bias_function", None)
                    or getattr(module, "weight_scale", None) is not None
                    or getattr(module, "_forward_hooks", None)
                    or getattr(module, "_forward_pre_hooks", None)
                    or module in self.pending or module in self.in_use):
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
            elif getattr(args, "pin_quantized_weights", False):
                from backend.quantized_storage import int8_storage
                storage = int8_storage(source)
                if storage is None or None in storage.versions or storage.orig_dtype != dtype:
                    continue
                packed = "int8"
                raw = storage.parts
            else:
                continue
            parts = raw if packed == "int8" else (raw,)
            if any(not part.is_contiguous() or not part.is_pinned() for part in parts):
                continue
            try:
                version = storage if packed == "int8" else TransferSource(source, raw)
            except RuntimeError:
                continue
            parts = storage.transfer_parts if packed == "int8" else (version.raw,)
            size = sum(part.numel() * part.element_size() for part in parts)
            if not size or size > allowance:
                continue
            # Fully resident blocks should not create a stream or query VRAM.
            if not checked_memory:
                free, _ = torch.cuda.mem_get_info(self.device)
                allowance = min(allowance, max(0, free - _MIN_HEADROOM))
                checked_memory = True
                if allowance <= 0:
                    break
                if size > allowance:
                    continue
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(self.stream):
                try:
                    values = tuple(torch.empty_like(part, device=self.device) for part in parts)
                    value = values if packed == "int8" else values[0]
                except torch.cuda.OutOfMemoryError:
                    # Only a failed speculative allocation is recoverable here.
                    break
                ready = torch.cuda.Event()
                self.last_copy = None  # close must synchronize if recording fails
                self.copy_owners = (parts, values)
                with record_region("forge::weight_prefetch_copy"):
                    for dst, src in zip(values, parts):
                        dst.copy_(src, non_blocking=True)
                    ready.record(self.stream)
                self.last_copy = ready
            self.pending[module] = (source, version, value, ready, packed, size)
            self.copy_owners = None
            self.bytes += size
            allowance -= size
            if allowance <= 0:
                break

    def take(self, module, source, device, dtype):
        if device != self.device:
            return None
        entry = self.pending.pop(module, None)
        if entry is None:
            return None
        original, version, value, ready, packed, _ = entry
        try:
            valid = (version.matches(source) if packed == "int8"
                     else version.matches(source, source.data if packed else source))
        except RuntimeError:
            valid = False
        incompatible = (dtype not in (None, version.orig_dtype) if packed == "int8"
                        else dtype is not None if packed else dtype not in (None, value.dtype))
        if not valid or incompatible:
            self.retired.append((entry, ready))
            return None
        current = torch.cuda.current_stream(device)
        with record_region("forge::weight_prefetch_wait"):
            current.wait_event(ready)
        for part in value if packed == "int8" else (value,):
            part.record_stream(current)
        self.in_use[module] = (entry, current)
        if packed == "int8":
            return version.rebuild(value)
        return source.copy_with_data(value) if packed else value

    def discard(self, block):
        # One completion event per compute stream, after the block's kernels
        # have been submitted. Do not uncharge consumed or unused copies early.
        consumed = {}
        for module in block.modules():
            entry = self.pending.pop(module, None)
            if entry is not None:
                self.retired.append((entry, entry[3]))
            used = self.in_use.pop(module, None)
            if used is not None:
                entry, current = used
                consumed.setdefault(current, []).append(entry)
        for current, entries in consumed.items():
            done = torch.cuda.Event()
            done.record(current)
            self.retired.extend((entry, done) for entry in entries)
        self._reap()

    def close(self):
        # CPU weight owners may be unpinned immediately after a failed or
        # completed forward. Wait only for outstanding H2D work, not compute.
        # record_stream protects consumed GPU storage after this scope exits.
        if self.stream is not None:
            if self.last_copy is None:
                self.stream.synchronize()
            elif not self.last_copy.query():
                self.last_copy.synchronize()
        self.copy_owners = None
        self.pending.clear()
        self.in_use.clear()
        self.retired.clear()
        self.bytes = 0


@contextmanager
def weight_prefetch_scope(device, options, enabled=True):
    budget = getattr(args, "weight_prefetch_mib", 0) * 1024 * 1024
    enabled = (enabled and budget > 0 and device.type == "cuda"
               and not torch.is_grad_enabled()
               and not torch.nn.modules.module._global_forward_hooks
               and not torch.nn.modules.module._global_forward_pre_hooks
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
