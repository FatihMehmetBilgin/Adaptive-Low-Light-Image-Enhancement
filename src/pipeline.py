"""End-to-end hybrid pipeline (proposal §IV — "Proposed Parallel Fusion").

Wiring of the full method:

    RGB --HSV--> isolate V ---+--> CLAHE branch --\\
                              |                     entropy fusion --> V_new --> RGB
                              +--> SSR branch   --/

The CLAHE and SSR branches both read the *same* original V independently; neither
branch's output feeds the other. This parallel (independent-branch) architecture
is the project's core proposition — it avoids the artifact amplification that a
sequential CLAHE->SSR / SSR->CLAHE chain suffers (those live in baselines.py as
ablations). H and S are never touched, preserving color fidelity.
"""
from typing import NamedTuple

import numpy as np

from . import clahe_branch, color, fusion, ssr_branch


class PipelineResult(NamedTuple):
    rgb: np.ndarray       # final enhanced RGB, float32 [0,1]
    v: np.ndarray         # original (input) V channel
    v_clahe: np.ndarray   # CLAHE branch output
    v_ssr: np.ndarray     # SSR branch output
    v_new: np.ndarray     # fused V channel
    alpha: float          # entropy-based fusion weight


def enhance_detailed(rgb, clip_limit=clahe_branch.DEFAULT_CLIP_LIMIT,
                     sigma=ssr_branch.DEFAULT_SIGMA) -> PipelineResult:
    """Run the full hybrid pipeline, returning the result and its intermediates."""
    h, s, v = color.decompose(rgb)
    # Parallel, independent branches on the same V (the branch funcs are pure,
    # so passing v to both is equivalent to cloning it — neither mutates v).
    v_clahe = clahe_branch.enhance_value(v, clip_limit=clip_limit)
    v_ssr = ssr_branch.enhance_value(v, sigma=sigma)
    v_new, alpha = fusion.fuse(v_clahe, v_ssr)
    rgb_out = color.reconstruct(h, s, v_new)
    return PipelineResult(rgb_out, v, v_clahe, v_ssr, v_new, alpha)


def enhance(rgb, clip_limit=clahe_branch.DEFAULT_CLIP_LIMIT,
            sigma=ssr_branch.DEFAULT_SIGMA):
    """Run the full hybrid pipeline and return the enhanced RGB (float32 [0,1])."""
    return enhance_detailed(rgb, clip_limit=clip_limit, sigma=sigma).rgb
