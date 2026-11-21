"""Conservative compute precision for heterogeneous CUDA devices."""

import torch


def sharded_compute_dtype(primary_dtype, supported_dtypes):
    """Keep FP32 requests and avoid FP16 for models that do not support it."""
    if primary_dtype == torch.float32 or torch.float16 not in supported_dtypes:
        return torch.float32
    return torch.float16
