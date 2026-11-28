"""Avoid redundant native image normalization and CPU scratch allocation."""
import numpy as np
import torch


def can_decode_direct_pixels(model, latent, native_engine, native_vae):
    """Do not bypass custom decode methods or tensor-subclass contracts."""
    if (type(latent) is not torch.Tensor or latent.ndim not in (4, 5)
            or not latent.is_floating_point()):
        return False
    vae = getattr(getattr(model, "forge_objects", None), "vae", None)
    return (type(vae) is native_vae
            and getattr(getattr(model, "decode_first_stage", None), "__func__", None) is native_engine.decode_first_stage
            and getattr(getattr(model, "decode_first_stage_pixels", None), "__func__", None) is native_engine.decode_first_stage_pixels
            and getattr(getattr(vae, "decode", None), "__func__", None) is native_vae.decode
            and getattr(vae, "process_output", None) is native_vae.process_output)


def chw_to_uint8(image):
    # Keep legacy multiplication then truncation semantics, including
    # grayscale replication; the caller checks shape and normalizes pixels.
    if image.shape[0] == 1:
        image = np.broadcast_to(image, (3, *image.shape[1:]))
    image = np.moveaxis(image, 0, 2)
    output = np.empty(image.shape, dtype=np.uint8)
    np.multiply(image, 255.0, out=output, casting="unsafe")
    return output
