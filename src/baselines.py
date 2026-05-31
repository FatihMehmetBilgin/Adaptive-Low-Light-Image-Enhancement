"""Baseline and ablation methods (proposal §III.A).

Reference methods the proposed parallel fusion is compared against:
  * GHE   — Global Histogram Equalization (no local context; amplifies shadow noise).
  * Gamma — V_out = A * V_in^gamma (global, no regional adaptation).
  * Sequential CLAHE->SSR and SSR->CLAHE — ablations that *chain* the two branches
    (one feeds the other) instead of running them in parallel, to quantify the
    artifact amplification the parallel architecture avoids.

For a fair comparison every method enhances only the V channel and keeps H, S
untouched (same color handling as the proposed pipeline); the only variable is
the enhancement algorithm. The sequential ablations reuse the exact CLAHE and
SSR branch functions, so the contrast is purely parallel-fusion vs chaining.
"""
import cv2
import numpy as np

from . import clahe_branch, color, ssr_branch

DEFAULT_GAMMA = 0.5
DEFAULT_A = 1.0


def _apply(rgb, value_fn):
    """Run a V-channel enhancer and reconstruct RGB (H, S preserved)."""
    h, s, v = color.decompose(rgb)
    return color.reconstruct(h, s, value_fn(v))


# --- V-level enhancers (the algorithms) ---

def ghe_value(v):
    """Global Histogram Equalization on the V channel."""
    v8 = (np.clip(v, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return cv2.equalizeHist(v8).astype(np.float32) / 255.0


def gamma_value(v, gamma=DEFAULT_GAMMA, a=DEFAULT_A):
    """Gamma correction V_out = A * V_in^gamma (gamma < 1 brightens)."""
    v = np.clip(np.asarray(v, dtype=np.float32), 0.0, 1.0)
    return np.clip(a * np.power(v, gamma), 0.0, 1.0).astype(np.float32)


def sequential_clahe_ssr_value(v, clip_limit=clahe_branch.DEFAULT_CLIP_LIMIT,
                               sigma=ssr_branch.DEFAULT_SIGMA):
    """Ablation: CLAHE first, its output then fed into SSR (chained)."""
    v_clahe = clahe_branch.enhance_value(v, clip_limit=clip_limit)
    return ssr_branch.enhance_value(v_clahe, sigma=sigma)


def sequential_ssr_clahe_value(v, clip_limit=clahe_branch.DEFAULT_CLIP_LIMIT,
                               sigma=ssr_branch.DEFAULT_SIGMA):
    """Ablation: SSR first, its output then fed into CLAHE (chained)."""
    v_ssr = ssr_branch.enhance_value(v, sigma=sigma)
    return clahe_branch.enhance_value(v_ssr, clip_limit=clip_limit)


# --- RGB-level wrappers (uniform rgb->rgb interface for experiments) ---

def ghe(rgb):
    return _apply(rgb, ghe_value)


def gamma_correction(rgb, gamma=DEFAULT_GAMMA, a=DEFAULT_A):
    return _apply(rgb, lambda v: gamma_value(v, gamma=gamma, a=a))


def sequential_clahe_ssr(rgb, clip_limit=clahe_branch.DEFAULT_CLIP_LIMIT,
                         sigma=ssr_branch.DEFAULT_SIGMA):
    return _apply(rgb, lambda v: sequential_clahe_ssr_value(
        v, clip_limit=clip_limit, sigma=sigma))


def sequential_ssr_clahe(rgb, clip_limit=clahe_branch.DEFAULT_CLIP_LIMIT,
                         sigma=ssr_branch.DEFAULT_SIGMA):
    return _apply(rgb, lambda v: sequential_ssr_clahe_value(
        v, clip_limit=clip_limit, sigma=sigma))
