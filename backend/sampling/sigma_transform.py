"""Versioned fixed-schedule transforms; never replace dynamic solver values."""
import torch


class SigmaTransform:
    def __init__(self, sigmas, transform, enabled, revision=lambda: None):
        self.sigmas = sigmas
        self.transform = transform
        self.revision = revision
        self.enabled = (enabled and type(sigmas) is torch.Tensor and sigmas.ndim == 1
                        and sigmas.dtype in (torch.float32, torch.float64))
        self.key = self.values = None

    def value(self, index):
        if self.enabled:
            try:
                key = (self.sigmas._version, self.revision())
            except RuntimeError:
                key = None
            if key is not None:
                if self.values is None or key != self.key:
                    self.values = self.transform(self.sigmas)
                    self.key = key
                return self.values[index]
        return self.transform(self.sigmas[index])
