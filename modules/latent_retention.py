from contextlib import contextmanager


@contextmanager
def retain_latents(processing, retain_all):
    """Keep each batch available to callbacks; request-wide retention is opt-in."""
    processing._retain_all_latents = retain_all
    processing.latents_after_sampling.clear()
    try:
        yield
    finally:
        if not retain_all:
            processing.latents_after_sampling.clear()


def scripts_require_latents(runner, script_args):
    scripts = list(runner.alwayson_scripts)
    selected = script_args.get(0, 0) if isinstance(script_args, dict) else (script_args[0] if script_args else 0)
    if isinstance(selected, int) and 0 < selected <= len(runner.selectable_scripts):
        scripts.append(runner.selectable_scripts[selected - 1])
    return any(getattr(script, "requires_all_latents", False) for script in scripts)
