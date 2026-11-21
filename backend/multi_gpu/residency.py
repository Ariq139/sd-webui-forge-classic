"""Explicit records for a model whose parts may reside on several devices.

The records are deliberately separate from Forge's single-device loader until
the loader and patcher can enforce their lifecycle together.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch


@dataclass(frozen=True)
class Residency:
    """Expected device of every parameter and buffer in a logical model."""

    tensors: Mapping[str, torch.device]

    def __post_init__(self):
        normalized = {name: torch.device(device) for name, device in self.tensors.items()}
        if any(device.type == "cuda" and device.index is None for device in normalized.values()):
            raise ValueError("CUDA devices must have explicit indices")
        object.__setattr__(self, "tensors", MappingProxyType(normalized))

    @property
    def devices(self):
        return frozenset(self.tensors.values())

    def validate(self, model):
        """Check the complete live model, including tied tensors and buffers."""
        actual = dict(model.named_parameters(remove_duplicate=False))
        actual.update(model.named_buffers(remove_duplicate=False))
        missing = sorted(self.tensors.keys() - actual.keys())
        unexpected = sorted(actual.keys() - self.tensors.keys())
        misplaced = sorted(
            (name, str(actual[name].device), str(device))
            for name, device in self.tensors.items()
            if name in actual and actual[name].device != device
        )
        if missing or unexpected or misplaced:
            raise ValueError(
                f"Model residency mismatch: missing={missing}, unexpected={unexpected}, misplaced={misplaced}"
            )
