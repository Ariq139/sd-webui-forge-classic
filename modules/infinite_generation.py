"""Bounded state and metadata helpers for native infinite generation."""
from contextlib import contextmanager
from copy import copy


def fit_batch(values, count, fallback, *, step=0, cycle=False):
    """Keep script-provided values; fill missing entries without truncating a cycle."""
    values = list(values or [fallback])
    if cycle:
        return [values[i % len(values)] for i in range(count)]
    missing = max(0, count - len(values))
    if step:
        last = values[-1]
        values.extend(last + step * i for i in range(1, missing + 1))
    elif missing:
        values.extend([values[-1]] * missing)
    return values[:count]


def normalize_cycle(p, requested_batch_size, seed_is_sequence, subseed_is_sequence):
    """Align all per-image arrays before splitting a logical cycle into LoRA groups."""
    count = max(1, requested_batch_size, len(p.all_prompts or ()))
    p.all_prompts = fit_batch(p.all_prompts, count, p.prompt)
    p.all_negative_prompts = fit_batch(p.all_negative_prompts, count, p.negative_prompt)
    p.all_seeds = fit_batch(p.all_seeds, count, p.seed, step=0 if p.subseed_strength else 1, cycle=seed_is_sequence)
    p.all_subseeds = fit_batch(p.all_subseeds, count, p.subseed, step=1, cycle=subseed_is_sequence)
    if getattr(p, "enable_hr", False):
        p.all_hr_prompts = fit_batch(p.all_hr_prompts, count, p.hr_prompt or p.all_prompts[0])
        p.all_hr_negative_prompts = fit_batch(p.all_hr_negative_prompts, count, p.hr_negative_prompt or p.all_negative_prompts[0])
    return count


def reset_hr_prompts(p, batch_size):
    """Start each cycle from full HR templates, never the previous LoRA group."""
    if getattr(p, "enable_hr", False):
        p.all_hr_prompts = fit_batch(getattr(p, "_infinite_hr_prompts_template", None), batch_size, p.hr_prompt or p.prompt)
        p.all_hr_negative_prompts = fit_batch(getattr(p, "_infinite_hr_negative_prompts_template", None), batch_size, p.hr_negative_prompt or p.negative_prompt)


def intermediate_index(p, index):
    """Map an intermediate image to the metadata array that owns its prompt."""
    if p.infinite_generation:
        return index
    indices = getattr(p, "_current_batch_indices", None)
    if indices is not None:
        return indices[index]
    return p.iteration * p.batch_size + index


def require_bounded_retention(infinite, retain_latents, retain_pixels):
    if infinite and (retain_latents or retain_pixels):
        raise ValueError("Infinite generation cannot retain all latents or pixels. Disable request-wide retention (including scripts that require it) or use a finite Batch Count.")


@contextmanager
def generation_state(state, infinite):
    previous = state.infinite_generation, state.infinite_batch
    state.infinite_generation = infinite
    state.infinite_batch = 0
    try:
        yield
    finally:
        state.infinite_generation, state.infinite_batch = previous


def stop_after_current(state, enabled):
    return enabled and not state.stopping_generation and (state.infinite_generation or state.job_count > 1)


def progress_text(state):
    return f"Infinite batch {max(1, state.infinite_batch)} (current pass)"


# Keep one committed result alongside the in-flight batch. The saved images,
# metadata and callback-visible pixels must always refer to the same batch.
_RESULT_FIELDS = (
    "batch_size", "iteration", "all_prompts", "all_negative_prompts", "all_seeds", "all_subseeds",
    "prompts", "negative_prompts", "seeds", "subseeds", "all_hr_prompts", "all_hr_negative_prompts",
    "main_prompt", "main_negative_prompt", "extra_generation_params", "extra_result_images", "pixels_after_sampling",
)


def capture_completed(p, images, infotexts, video_path):
    return list(images), list(infotexts), video_path, {name: copy(getattr(p, name)) for name in _RESULT_FIELDS if hasattr(p, name)}


def restore_completed(p, snapshot):
    images, infotexts, video_path, fields = snapshot
    for name, value in fields.items():
        setattr(p, name, value)
    return images, infotexts, video_path
