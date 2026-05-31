"""HSV color-space decoupling and reconstruction (proposal Steps 1 & 6).

The pipeline isolates the V (Value/luminance) channel for enhancement while
leaving H (Hue) and S (Saturation) completely untouched, which preserves color
fidelity. All functions operate on float32 RGB in [0, 1]; in that domain OpenCV
yields H in [0, 360) and S, V in [0, 1].
"""
import cv2
import numpy as np


def rgb_to_hsv(rgb):
    """RGB float32 [0,1] -> HSV (H in [0,360), S and V in [0,1])."""
    return cv2.cvtColor(np.asarray(rgb, dtype=np.float32), cv2.COLOR_RGB2HSV)


def hsv_to_rgb(hsv):
    """HSV -> RGB float32 [0,1]."""
    return cv2.cvtColor(np.asarray(hsv, dtype=np.float32), cv2.COLOR_HSV2RGB)


def decompose(rgb):
    """Split RGB into (h, s, v) channels.

    Only ``v`` should be modified downstream; ``h`` and ``s`` must be carried
    through unchanged and handed back to :func:`reconstruct`.
    """
    h, s, v = cv2.split(rgb_to_hsv(rgb))
    return h, s, v


def reconstruct(h, s, v_new):
    """Recombine the original H, S with a processed V channel into RGB.

    ``v_new`` is clipped to [0, 1] before merging; the final RGB is clipped too
    to guard against tiny floating-point overshoots from the HSV->RGB transform.
    """
    v = np.clip(v_new, 0.0, 1.0).astype(np.float32)
    hsv = cv2.merge([
        np.asarray(h, dtype=np.float32),
        np.asarray(s, dtype=np.float32),
        v,
    ])
    return np.clip(hsv_to_rgb(hsv), 0.0, 1.0)


def roundtrip_error(rgb):
    """Max absolute per-pixel error of an identity RGB->HSV->RGB round-trip.

    Used as a sanity check that decoupling/reconstruction is lossless (a healthy
    value for float32 input is well below 1e-3).
    """
    h, s, v = decompose(rgb)
    rt = reconstruct(h, s, v)
    return float(np.max(np.abs(rt - np.clip(rgb, 0.0, 1.0))))
