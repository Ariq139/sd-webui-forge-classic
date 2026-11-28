"""Validate native Diffusers requests without loading model weights."""
import math
import secrets

from modules_forge.native_requests import positive_integer


def finite_number(value, label, minimum=None, maximum=None):
    try:
        value = float(value)
        if not math.isfinite(value) or minimum is not None and value < minimum or maximum is not None and value > maximum:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"Invalid {label}: expected a finite number in the supported range.") from None
    return value


def actual_seed(value):
    if value is None or str(value).strip() in ("", "-1") or value == -1:
        return secrets.randbelow(2**31)
    try:
        seed = int(value)
        if seed < 0 or seed >= 2**63 or not isinstance(value, (int, str)) and value != seed:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Seed must be -1 or a nonnegative integer below 2**63.") from None
    return seed


def ltx_dimensions(width, height, frames, fps, steps, temporal_ratio=8):
    width = positive_integer(width, "Width")
    height = positive_integer(height, "Height")
    frames = positive_integer(frames, "Frames")
    fps = positive_integer(fps, "FPS")
    steps = positive_integer(steps, "Sampling steps")
    if min(width, height) < 64 or width % 32 or height % 32:
        raise ValueError("LTX width and height must be at least 64 and divisible by 32.")
    if frames < 9 or (frames - 1) % temporal_ratio:
        raise ValueError(f"LTX frames must be at least 9 and follow {temporal_ratio}n+1.")
    if fps > 60 or steps > 150:
        raise ValueError("LTX FPS must be 1-60 and steps 1-150.")
    return width, height, frames, fps, steps


def ltx_cross_timestep(config, mode="Automatic"):
    if mode == "LTX-2.3":
        return True
    if mode == "LTX-2.0":
        return False
    if mode != "Automatic":
        raise ValueError("Unknown LTX cross-timestep mode.")
    # In pinned Diffusers, prompt modulation is introduced for LTX-2.3.
    get = config.get if hasattr(config, "get") else lambda key, default: getattr(config, key, default)
    return bool(get("cross_attn_mod", False) or get("audio_cross_attn_mod", False))


def audio_override(audio, video, label):
    audio = finite_number(audio, label, 0)
    # Diffusers 0.39.0 uses `audio_value or video_value`, not an explicit None test.
    if audio == 0 and video != 0:
        raise ValueError(f"{label}=0 cannot be independent of nonzero video guidance in Diffusers 0.39.0.")
    return audio


def ideogram_guidance(steps, guidance, polish):
    steps = positive_integer(steps, "Sampling steps")
    guidance = finite_number(guidance, "CFG scale", 0, 24)
    if not polish:
        return {"guidance_scale": guidance, "guidance_schedule": None}
    # The pinned quality recipe finishes with three CFG=3 polishing steps.
    # Preserve the user's main CFG and keep at least one main step.
    tail = min(3, steps - 1)
    return {"guidance_scale": None, "guidance_schedule": [guidance] * (steps - tail) + [min(guidance, 3.0)] * tail}
