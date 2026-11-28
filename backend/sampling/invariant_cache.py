"""Opt-in, sampling-scoped storage for immutable single-GPU features."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from backend.performance import record_region

_CACHE = ContextVar("forge_invariant_cache", default=None)
_REVISION = ContextVar("forge_invariant_revision", default=None)
_CONDITION = ContextVar("forge_invariant_condition", default=None)
_SAFE_OPTIONS = frozenset({"cond_or_uncond", "sigmas", "cond_mark", "cond_indices", "uncond_indices", "sampling_sigmas"})


def _bytes(value):
    if isinstance(value, torch.Tensor):
        return value.untyped_storage().nbytes() if value.device.type != "meta" else 0
    if isinstance(value, tuple):
        return sum(_bytes(item) for item in value)
    return 0


class InvariantCache:
    def __init__(self, max_bytes):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes_used = 0
        self.entries = {}
        self.sources = {}

    def put(self, key, value):
        size = _bytes(value)
        # Admission rather than LRU eviction avoids thrashing when all layers
        # are visited in the same order on each denoising evaluation.
        if key in self.entries or not size or self.bytes_used + size > self.max_bytes:
            return False
        self.entries[key] = value
        self.bytes_used += size
        return True

    def clear(self):
        self.entries.clear()
        self.sources.clear()
        self.bytes_used = 0


@contextmanager
def sampling_invariant_scope(max_bytes):
    cache = InvariantCache(max_bytes) if max_bytes > 0 else None
    token = _CACHE.set(cache)
    try:
        yield cache
    finally:
        if cache is not None:
            cache.clear()
        _CACHE.reset(token)


@contextmanager
def invariant_revision(revision):
    token = _REVISION.set(revision)
    try:
        yield
    finally:
        _REVISION.reset(token)


@contextmanager
def invariant_condition(key):
    token = _CONDITION.set(key)
    try:
        yield
    finally:
        _CONDITION.reset(token)


def clear_invariants():
    cache = _CACHE.get()
    if cache is not None:
        cache.clear()


def invariants_enabled(module, tensor, options):
    return (_CACHE.get() is not None and _REVISION.get() is not None
            and _CONDITION.get() is not None and not torch.is_grad_enabled()
            and isinstance(tensor, torch.Tensor) and not module.training
            and isinstance(options, dict) and options.keys() <= _SAFE_OPTIONS)


def cached_invariant(module, kind, tensor, options, compute, *, geometry=(), clone=False):
    """Cache only an explicitly proved invariant, never a denoiser output."""
    if not invariants_enabled(module, tensor, options):
        return compute()
    cache = _CACHE.get()
    key = (_REVISION.get(), _CONDITION.get(), module, kind, tensor.device,
           tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), geometry)
    try:
        hash(key)
    except TypeError:
        return compute()
    value = cache.entries.get(key)
    if value is not None:
        with record_region("forge::invariant_cache_hit::" + kind):
            return value.clone() if clone else value
    with record_region("forge::invariant_cache_miss::" + kind):
        value = compute()
    if cache.put(key, value) and clone:
        return value.clone()
    return value


def cached_adapter_cast(source, inputs, cast):
    """Reuse a converted adapter only while its source/version is unchanged."""
    cache = _CACHE.get()
    if (cache is None or torch.is_grad_enabled() or not isinstance(source, torch.Tensor)
            or (source.device == inputs.device and source.dtype == inputs.dtype)):
        return cast(source, inputs.device, inputs.dtype)
    try:
        version = source._version
    except RuntimeError:
        return cast(source, inputs.device, inputs.dtype)
    key = ("adapter", id(source), version, inputs.device, inputs.dtype)
    value = cache.entries.get(key)
    if value is not None and cache.sources.get(id(source)) is source:
        return value
    value = cast(source, inputs.device, inputs.dtype)
    if cache.put(key, value):
        cache.sources[id(source)] = source
    return value
