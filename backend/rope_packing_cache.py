"""Bounded-by-forward view identity, with explicit immutable geometry reuse."""
import torch


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def view_key(tensor):
    # A pointer alone cannot distinguish strided views or reallocated storage.
    return (tensor.device, tensor.dtype, tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
            _version(tensor))


class PackedRopeViews:
    def __init__(self):
        self.local = {}
        self.source = None
        self.geometry = None
        self.source_key = None

    def bind(self, source, geometry):
        self.source = source
        self.geometry = geometry
        self.source_key = view_key(source)

    def persistent_geometry(self, view):
        if self.source is None or view_key(self.source) != self.source_key:
            return None
        key = view_key(view)
        if key[:3] != self.source_key[:3] or key[-1] != self.source_key[-1]:
            return None
        return (self.geometry, key[3] - self.source_key[3], key[4], key[5])

    def get(self, view, pack, persistent=None):
        key = view_key(view)
        entry = self.local.get(key)
        if entry is not None:
            return entry[1]
        geometry = self.persistent_geometry(view)
        value = persistent(view, geometry, pack) if geometry is not None and persistent is not None else pack(view)
        # Keep the source view alive until forward cleanup to prevent allocator
        # pointer reuse; retaining views does not duplicate their storage.
        self.local[key] = (view, value)
        return value

    def clear(self):
        self.local.clear()
        self.source = self.geometry = self.source_key = None
