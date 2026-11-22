"""Conservative contiguous two-device weight placement primitives."""

from dataclasses import dataclass


class InsufficientDeviceMemory(ValueError):
    """No contiguous partition fits the current device budgets."""


@dataclass(frozen=True)
class Budget:
    free_bytes: int
    reserve_bytes: int

    def __post_init__(self):
        if self.free_bytes < 0 or self.reserve_bytes < 0:
            raise ValueError("Memory values must be non-negative")

    @property
    def available(self):
        return max(0, self.free_bytes - self.reserve_bytes)


def contiguous_split(block_sizes, first_budget, second_budget, *, first_fixed=0, second_fixed=0, policy="balanced_memory"):
    """Return the split index with the smallest maximum budget utilization.

    Fixed allocations and reserves must include all non-block weights and
    estimated live activations/workspaces. This function cannot measure those
    values or guarantee that a generation fits.
    """
    sizes = tuple(block_sizes)
    if policy not in ("balanced_memory", "prefer_primary"):
        raise ValueError(f"Unknown multi-GPU split policy: {policy}")
    if len(sizes) < 2 or any(size < 0 for size in sizes) or min(first_fixed, second_fixed) < 0:
        raise ValueError("At least two non-negative blocks and non-negative fixed costs are required")
    a, b = first_budget.available, second_budget.available
    total = sum(sizes)
    prefix = 0
    candidates = []
    for index, size in enumerate(sizes[:-1], start=1):
        prefix += size
        used_a, used_b = first_fixed + prefix, second_fixed + total - prefix
        if used_a <= a and used_b <= b:
            utilization = max(used_a / a if a else 0, used_b / b if b else 0)
            candidates.append((utilization, index))
    if not candidates:
        raise InsufficientDeviceMemory("No contiguous split fits both device budgets")
    if policy == "prefer_primary":
        return max(index for _, index in candidates)
    return min(candidates)[1]
