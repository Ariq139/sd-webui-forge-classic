"""Sampling-scoped cache for Krea's prompt-only text-fusion layers."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from weakref import WeakSet

import torch

from backend.performance import record_region


_MAX_CACHE_BYTES = 16 * 1024 * 1024
_CACHE = ContextVar("forge_krea_text_fusion_cache", default=None)
_CONDITION_KEY = ContextVar("forge_krea_text_fusion_condition", default=None)
_MODEL_REVISION = ContextVar("forge_krea_text_fusion_revision", default=None)


class _SamplingCache:
    def __init__(self, namespace, max_bytes):
        self.namespace = namespace
        self.max_bytes = max_bytes
        self.bytes_used = 0
        self.entries = OrderedDict()
        self.position_embedders = WeakSet()

    def get(self, key):
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
        return value

    def put(self, key, value):
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            return
        size = value.numel() * value.element_size()
        if size <= 0 or size > self.max_bytes:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes_used -= previous.numel() * previous.element_size()
        while self.entries and self.bytes_used + size > self.max_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.bytes_used -= evicted.numel() * evicted.element_size()
        self.entries[key] = value
        self.bytes_used += size

    def clear(self):
        self.entries.clear()
        self.bytes_used = 0


@contextmanager
def sampling_krea_cache_scope(namespace, max_bytes=_MAX_CACHE_BYTES, position_embedder=None):
    """Keep cached prompt features alive only for one sampler invocation."""
    current = _CACHE.get()
    if current is not None:
        if position_embedder is not None:
            current.position_embedders.add(position_embedder)
        yield current
        return

    cache = _SamplingCache(namespace, max_bytes)
    if position_embedder is not None:
        cache.position_embedders.add(position_embedder)
    token = _CACHE.set(cache)
    try:
        yield cache
    finally:
        cache.clear()
        try:
            if cache.position_embedders:
                from backend.nn.krea import clear_krea_position_cache

                for embedder in tuple(cache.position_embedders):
                    clear_krea_position_cache(embedder)
        finally:
            _CACHE.reset(token)


@contextmanager
def krea_text_fusion_condition(key):
    token = _CONDITION_KEY.set(key)
    try:
        yield
    finally:
        _CONDITION_KEY.reset(token)


@contextmanager
def krea_text_fusion_revision(revision):
    token = _MODEL_REVISION.set(revision)
    try:
        yield
    finally:
        _MODEL_REVISION.reset(token)


def clear_current_krea_cache():
    cache = _CACHE.get()
    if cache is not None:
        cache.clear()


def _safe_transformer_options(options):
    if not isinstance(options, dict):
        return False
    # These fields describe the current sampler grouping. Krea's text fusion
    # does not use them. Any other field may carry an extension or attention
    # override that changes the result, so cache conservatively bypasses it.
    allowed = {"cond_or_uncond", "sigmas", "cond_mark", "cond_indices", "uncond_indices"}
    return options.keys() <= allowed


def cached_krea_text_features(module, context, transformer_options, compute):
    """Return cached Krea text features when the semantic conditioning key is stable."""
    cache = _CACHE.get()
    condition_key = _CONDITION_KEY.get()
    if (cache is None or condition_key is None or not isinstance(context, torch.Tensor)
            or torch.is_grad_enabled() or module.training or not _safe_transformer_options(transformer_options)):
        return compute()

    try:
        version = context._version
    except RuntimeError:  # inference tensors do not expose a version counter
        version = None

    key = (
        cache.namespace,
        _MODEL_REVISION.get(),
        id(module),
        condition_key,
        context.device.type,
        context.device.index,
        context.dtype,
        tuple(context.shape),
        tuple(context.stride()),
        version,
    )
    try:
        hash(key)
    except TypeError:
        return compute()

    cached = cache.get(key)
    if cached is not None:
        with record_region("forge::krea_text_fusion_cache_hit"):
            return cached

    output = compute()
    cache.put(key, output)
    return output
