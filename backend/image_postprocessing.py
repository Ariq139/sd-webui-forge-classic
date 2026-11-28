"""Avoid an image-sized float scratch array during final CPU conversion."""
import numpy as np


def chw_to_uint8(image):
    # Keep legacy multiplication then truncation semantics, including
    # grayscale replication; the caller checks shape and normalizes pixels.
    if image.shape[0] == 1:
        image = np.broadcast_to(image, (3, *image.shape[1:]))
    image = np.moveaxis(image, 0, 2)
    output = np.empty(image.shape, dtype=np.uint8)
    np.multiply(image, 255.0, out=output, casting="unsafe")
    return output
