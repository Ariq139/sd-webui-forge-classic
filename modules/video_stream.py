"""Checked, atomic FFmpeg raw-frame export without accumulating all frames."""
import os
import subprocess
import tempfile
from itertools import chain

import numpy as np


def first_frame(frames):
    iterator = iter(frames)
    try:
        frame = next(iterator)
    except StopIteration:
        raise ValueError("No video frames to save.") from None
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Video frames must be HxWx3 uint8 RGB arrays.")
    return frame.shape, chain((frame,), iterator)


def encode_frames(command, output_path, shape, frames):
    """Only publish the output after successful frame iteration and encoding."""
    directory = os.path.dirname(os.path.abspath(output_path))
    suffix = os.path.splitext(output_path)[1]
    fd, temporary = tempfile.mkstemp(prefix=".forge-video-", suffix=suffix, dir=directory)
    os.close(fd)
    process = None
    try:
        process = subprocess.Popen([*command, temporary], stdin=subprocess.PIPE)
        for frame in frames:
            if not isinstance(frame, np.ndarray) or frame.shape != shape or frame.dtype != np.uint8:
                raise ValueError("All output video frames must have the same RGB shape and uint8 dtype.")
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
        os.replace(temporary, output_path)
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            try:
                process.stdin.close()
            except OSError:
                pass
            process.wait()
        if os.path.exists(temporary):
            os.unlink(temporary)
