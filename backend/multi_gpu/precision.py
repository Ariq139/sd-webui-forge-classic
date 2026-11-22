"""Conservative compute precision for heterogeneous CUDA devices."""

import torch


def replica_memory_requirement(state_dict, reserve_bytes):
    """Estimate one replica's checkpoint storage plus a caller-provided reserve."""
    return sum(t.numel() * t.element_size() for t in state_dict.values() if isinstance(t, torch.Tensor)) + reserve_bytes


def replica_fits_memory(state_dict, free_bytes, reserve_bytes):
    return free_bytes >= replica_memory_requirement(state_dict, reserve_bytes)


def sharded_compute_dtype(primary_dtype, supported_dtypes):
    """Keep FP32 requests and avoid FP16 for models that do not support it."""
    if primary_dtype == torch.float32 or torch.float16 not in supported_dtypes:
        return torch.float32
    return torch.float16


def replica_compute_dtype(primary_dtype, supported_dtypes, devices):
    """Choose a compute dtype supported by both complete model replicas."""
    if primary_dtype == torch.float32:
        return torch.float32
    if primary_dtype == torch.bfloat16 and torch.bfloat16 in supported_dtypes:
        try:
            supported = []
            for device in devices:
                with torch.cuda.device(device):
                    supported.append(torch.cuda.is_bf16_supported(including_emulation=False))
            if all(supported):
                return torch.bfloat16
        except (RuntimeError, TypeError):
            pass
    if torch.float16 in supported_dtypes:
        return torch.float16
    return torch.float32
