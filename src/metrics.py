"""Evaluation metrics (proposal §V.B).

  * PSNR    — full-reference, dB, higher is better (Eq. 6); via scikit-image.
  * SSIM    — full-reference, higher is better (Eq. 7); via scikit-image.
  * BRISQUE — no-reference, lower is better; via the `brisque` package.
  * runtime — mean wall-clock ms per frame (proposal §V.B.4).

All image inputs are RGB float32 in [0,1] (data_range=1.0); BRISQUE converts to
uint8 internally.
"""
import time
import types

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(reference, image, data_range=1.0):
    """Peak Signal-to-Noise Ratio in dB (Eq. 6); higher is better."""
    return float(peak_signal_noise_ratio(reference, image, data_range=data_range))


def ssim(reference, image, data_range=1.0):
    """Structural Similarity Index for RGB (Eq. 7); higher is better."""
    return float(structural_similarity(
        reference, image, data_range=data_range, channel_axis=-1))


# --- BRISQUE (no-reference) ---
# The `brisque` package has an upstream numpy>=2 bug: its AGGD fit divides by
# `array.shape` (a tuple) instead of the element count, so feature values come
# back wrapped in size-1 arrays and float(f) raises in scale_features. The
# numeric values are correct, so we override scale_features to coerce them.
# The scorer is built once (model load is not free) and reused.
_brisque_obj = None


def _scale_features_fixed(self, features):
    mn = np.array([float(np.ravel(m)[0]) for m in self.scale_params["min_"]], dtype=np.float64)
    mx = np.array([float(np.ravel(m)[0]) for m in self.scale_params["max_"]], dtype=np.float64)
    fl = np.array([float(np.ravel(f)[0]) for f in features], dtype=np.float64)
    return -1.0 + (2.0 / (mx - mn) * (fl - mn))


def _get_brisque():
    global _brisque_obj
    if _brisque_obj is None:
        from brisque import BRISQUE

        obj = BRISQUE(url=False)
        obj.scale_features = types.MethodType(_scale_features_fixed, obj)
        _brisque_obj = obj
    return _brisque_obj


def brisque(image):
    """No-reference BRISQUE score; lower is better. Accepts float [0,1] or uint8 RGB."""
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return float(_get_brisque().score(img))


def evaluate(reference, enhanced, data_range=1.0):
    """All quality metrics for an enhanced image vs its reference.

    Returns ``{"psnr": ..., "ssim": ..., "brisque": ...}`` (runtime is measured
    separately on the method via :func:`runtime_ms`).
    """
    return {
        "psnr": psnr(reference, enhanced, data_range),
        "ssim": ssim(reference, enhanced, data_range),
        "brisque": brisque(enhanced),
    }


def runtime_ms(fn, *args, repeats=5, warmup=1, **kwargs):
    """Mean wall-clock runtime per call in milliseconds (proposal §V.B.4)."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    return (time.perf_counter() - start) / repeats * 1000.0
