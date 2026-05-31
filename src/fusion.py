"""Entropy-based adaptive fusion (proposal Step 5).

The two enhanced V channels are blended with a per-image weight derived from
their Shannon entropy, so the branch carrying more structural information is
emphasized automatically (no static, grid-searched blend parameter):

    H(V)  = -sum_i p(i) log p(i)                                  (Eq. 3)
    alpha = H(V_CLAHE) / (H(V_CLAHE) + H(V_SSR))                  (Eq. 4)
    V_new = alpha * V_CLAHE + (1 - alpha) * V_SSR                 (Eq. 5)

Entropy is computed from a fixed 256-level histogram over [0,1] so both
channels are compared on the same quantization (V_CLAHE is uint8-derived while
V_SSR is continuous). The log base only rescales H and cancels in the alpha
ratio, so base-2 (bits) is used purely for interpretability.
"""
import numpy as np

ENTROPY_BINS = 256


def shannon_entropy(v, bins=ENTROPY_BINS):
    """Shannon entropy (bits) of a V channel in [0,1] via a `bins`-level histogram."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    counts, _ = np.histogram(v, bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def fusion_weight(v_clahe, v_ssr, bins=ENTROPY_BINS):
    """Adaptive fusion weight alpha (Eq. 4); falls back to 0.5 if both are flat."""
    h_clahe = shannon_entropy(v_clahe, bins)
    h_ssr = shannon_entropy(v_ssr, bins)
    total = h_clahe + h_ssr
    if total < 1e-12:
        return 0.5
    return h_clahe / total


def fuse(v_clahe, v_ssr, bins=ENTROPY_BINS):
    """Entropy-weighted blend of the two V channels (Eq. 5).

    Returns ``(v_new, alpha)`` with v_new in [0,1] (both inputs are in [0,1]).
    """
    alpha = fusion_weight(v_clahe, v_ssr, bins)
    v_new = (
        alpha * np.asarray(v_clahe, dtype=np.float32)
        + (1.0 - alpha) * np.asarray(v_ssr, dtype=np.float32)
    )
    return v_new.astype(np.float32), alpha
