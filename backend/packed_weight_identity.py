"""Retain actual transfer-source storage and reject pointer/layout changes."""
import torch


def _signature(tensor):
    return (tensor.device, tensor.dtype, tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))


class TransferSource:
    def __init__(self, owner, raw, metadata=()):
        self.owner = owner
        # The detached alias keeps the submitted storage alive even if owner.data is rebound.
        self.raw = raw.detach()
        self.signature = _signature(raw)
        self.version = owner._version
        self.metadata = metadata

    def matches(self, owner, raw, metadata=()):
        try:
            return (owner is self.owner and owner._version == self.version
                    and _signature(raw) == self.signature and metadata == self.metadata)
        except RuntimeError:
            return False
