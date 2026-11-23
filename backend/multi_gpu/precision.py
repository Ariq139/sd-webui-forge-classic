"""Conservative compute precision for heterogeneous CUDA devices."""

import torch


def replica_memory_requirement(state_dict, reserve_bytes):
    """Estimate one replica's checkpoint storage plus a caller-provided reserve."""
    return sum(t.numel() * t.element_size() for t in state_dict.values() if isinstance(t, torch.Tensor)) + reserve_bytes


def replica_fits_memory(state_dict, free_bytes, reserve_bytes):
    return free_bytes >= replica_memory_requirement(state_dict, reserve_bytes)


def automatic_multi_gpu_mode(state_dict, gpu_free_bytes, host_free_bytes, reserve_bytes, *, replicas_supported, sharding_supported):
    """Choose a conservative placement from current memory, not GPU capacity labels."""
    size = replica_memory_requirement(state_dict, 0)
    if size <= 0 or len(gpu_free_bytes) != 2 or reserve_bytes < 0 or host_free_bytes < 0 or any(free < 0 for free in gpu_free_bytes):
        return "single"
    if (replicas_supported
            and all(free >= size + reserve_bytes for free in gpu_free_bytes)
            and host_free_bytes >= size + 512 * 1024 ** 2):
        return "replicas"
    return "sharding" if sharding_supported else "single"


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
