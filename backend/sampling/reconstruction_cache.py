"""One-entry native conditioning reconstruction cache, scoped to the sampler."""
import torch


def condition_source_key(value, dict_type=dict):
    if type(value) is torch.Tensor:
        try:
            version = value._version
        except RuntimeError:
            # Native inference schedule entries are read-only by contract.
            version = None
        return (id(value), version, value.device, value.dtype, value.data_ptr(),
                tuple(value.shape), tuple(value.stride()))
    if type(value) in (dict, dict_type) and value and value.keys() <= {"crossattn", "vector", "guidance"}:
        fields = tuple((name, condition_source_key(tensor)) for name, tensor in value.items())
        if all(key is not None for _, key in fields):
            return (id(value), fields)
    return None


def _storage_bytes(value, devices):
    if type(value) is torch.Tensor:
        if value.is_cuda:
            devices.add(value.device)
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sum(_storage_bytes(t, devices) for t in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_storage_bytes(t, devices) for t in value)
    return 0


class ReconstructionCache:
    def __init__(self, max_bytes=32 * 1024**2):
        self.max_bytes = max_bytes
        self.clear()

    def clear(self):
        self.key = self.sources = self.value = None

    def get(self, key, cond, uncond, build, *, enabled):
        if not enabled or key is None:
            self.clear()
            return build()
        if (self.key == key and self.sources is not None
                and self.sources[0] is cond and self.sources[1] is uncond):
            return self.value
        self.clear()
        value = build()
        devices = set()
        size = _storage_bytes(value, devices)
        if size <= self.max_bytes and all(torch.cuda.mem_get_info(device)[0] >= 256 * 1024**2 + size
                                         for device in devices):
            self.key, self.sources, self.value = key, (cond, uncond), value
        return value
