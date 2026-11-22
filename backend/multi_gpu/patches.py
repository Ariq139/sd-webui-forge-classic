"""Run device-aware extension callbacks on the device of their input."""

from contextlib import nullcontext

import torch


def _tensor_device(value):
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, (list, tuple)):
        for item in value:
            device = _tensor_device(item)
            if device is not None:
                return device
    if isinstance(value, dict):
        for item in value.values():
            device = _tensor_device(item)
            if device is not None:
                return device
    return None


class DevicePatch:
    def __init__(self, callback):
        self.original = callback
        self.callback = callback
        self.device = None
        self.move_incomplete = False

    def _move(self, device):
        if hasattr(self.callback, "to"):
            moved = self.callback.to(device)
            if moved is not None:
                if not callable(moved):
                    raise TypeError("Extension callback .to(device) must return a callable or None")
                self.callback = moved

    def __call__(self, *args, **kwargs):
        device = _tensor_device(args) or _tensor_device(kwargs)
        with torch.cuda.device(device) if device is not None and device.type == "cuda" else nullcontext():
            if device is not None and (self.move_incomplete or device != self.device):
                self.move_incomplete = True
                self._move(device)
                self.device = device
                self.move_incomplete = False
            return self.callback(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.callback, name)

    def restore(self, device):
        if self.move_incomplete or (self.device is not None and self.device != device):
            with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                self._move(device)
            self.move_incomplete = False
        self.device = device


def device_aware_options(options):
    """Copy callback containers without copying tensors or extension state."""
    memo = {}

    def wrap(value):
        if isinstance(value, dict):
            return {key: wrap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(wrap(item) for item in value)
        if callable(value):
            if id(value) not in memo:
                memo[id(value)] = DevicePatch(value)
            return memo[id(value)]
        return value

    result = options.copy()
    for key in ("patches", "patches_replace", "block_modifiers", "block_inner_modifiers", "group_norm_wrapper"):
        if key in result:
            result[key] = wrap(result[key])
    return result


def restore_patch_devices(options, device, sources=()):
    seen = set()
    replacements = {}
    first_error = None

    def restore(value):
        nonlocal first_error
        if isinstance(value, DevicePatch):
            if id(value) not in seen:
                seen.add(id(value))
                try:
                    value.restore(device)
                except Exception as error:
                    if first_error is None:
                        first_error = error
                finally:
                    replacements[id(value.original)] = value.callback
        elif isinstance(value, dict):
            for item in value.values():
                restore(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                restore(item)

    for key in ("patches", "patches_replace", "block_modifiers", "block_inner_modifiers", "group_norm_wrapper"):
        if key in options:
            restore(options[key])

    # Sampling makes shallow copies and may build temporary merged patch lists.
    # Write replacements back into every owning container, not only that copy.
    visited = {}

    def update(value):
        replacement = replacements.get(id(value))
        if replacement is not None:
            return replacement
        if isinstance(value, dict):
            if id(value) in visited:
                return visited[id(value)]
            visited[id(value)] = value
            for key, item in value.items():
                value[key] = update(item)
        elif isinstance(value, list):
            if id(value) in visited:
                return visited[id(value)]
            visited[id(value)] = value
            for index, item in enumerate(value):
                value[index] = update(item)
        elif isinstance(value, tuple):
            original_id = id(value)
            if original_id in visited:
                return visited[original_id]
            visited[original_id] = value
            value = tuple(update(item) for item in value)
            visited[original_id] = value
        return value

    for source in sources:
        if source is not None:
            for key in ("patches", "patches_replace", "block_modifiers", "block_inner_modifiers", "group_norm_wrapper"):
                if key in source:
                    source[key] = update(source[key])
    if first_error is not None:
        raise first_error
