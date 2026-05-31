"""SSR branch (proposal Step 4).

Single-Scale Retinex on the isolated V (Value) channel. Retinex theory
decouples reflectance from illumination in the logarithmic domain:

    R(x, y) = log(I(x, y)) - log(I(x, y) * F(x, y))            (Eq. 1)

where ``*`` is 2-D convolution and F is a normalized spatial Gaussian of
standard deviation sigma (Eq. 2). The Gaussian-blurred V estimates the
illumination; subtracting it in log space leaves the reflectance/detail map.
Larger sigma (the grid search uses {60, 80, 120}) estimates illumination at a
coarser, more global scale.

Two implementation details not fixed by the proposal:
  * ``eps`` guards log(0) on pure-black pixels.
  * the raw log-domain output is not in [0,1], so it is mapped back via robust
    percentile clipping + min-max scaling, making V_SSR directly fusable with
    V_CLAHE (proposal Step 5).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULT_SIGMA = 80.0
EPS = 1e-3
CLIP_PERCENTILES = (1.0, 99.0)


def single_scale_retinex(v, sigma=DEFAULT_SIGMA, eps=EPS):
    """Raw log-domain SSR reflectance map (Eq. 1); not normalized to [0,1]."""
    v = np.clip(np.asarray(v, dtype=np.float32), 0.0, 1.0)
    illumination = gaussian_filter(v, sigma=float(sigma))  # I * F, F sums to 1
    return np.log(v + eps) - np.log(illumination + eps)


def normalize(r, percentiles=CLIP_PERCENTILES):
    """Map a log-domain map to [0,1] via percentile clip + min-max scaling."""
    lo, hi = np.percentile(r, percentiles)
    if hi - lo < 1e-12:
        return np.zeros_like(r, dtype=np.float32)
    out = (np.clip(r, lo, hi) - lo) / (hi - lo)
    return out.astype(np.float32)


def enhance_value(v, sigma=DEFAULT_SIGMA, eps=EPS, percentiles=CLIP_PERCENTILES):
    """Apply SSR to a float V channel in [0,1]; returns V_SSR in [0,1]."""
    r = single_scale_retinex(v, sigma=sigma, eps=eps)
    return normalize(r, percentiles)
