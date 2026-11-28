"""Associate each regrouped prompt with its original img2img source."""
from contextlib import contextmanager


def source_indices(indices, original_size, source_count):
    if original_size < 1 or source_count not in (1, original_size):
        raise ValueError("Grouped img2img requires one source image or one per requested batch slot.")
    if any(index < 0 for index in indices):
        raise ValueError("Negative generation index.")
    return [0 if source_count == 1 else index % original_size for index in indices]


@contextmanager
def group_sources(p, sources, original_size, indices):
    """Initialize the current group using its own source images, then restore input list."""
    original = p.init_images
    selected = source_indices(indices, original_size, len(sources))
    p.init_images = list(sources) if len(sources) == 1 else [sources[i] for i in selected]
    try:
        yield
    finally:
        p.init_images = original
