"""CLAHE branch (proposal Step 3).

Contrast Limited Adaptive Histogram Equalization on the isolated V (Value)
channel. The channel is divided into an 8x8 grid of tiles; each tile histogram
is clipped at ``clip_limit`` (the excess mass redistributed evenly) before
equalization. This raises local contrast without promoting sensor noise to
false structure — clip limits above ~4.0 start amplifying Poisson shot noise,
which is why the grid search stays within {1.5, 2.0, 3.0}.

OpenCV's CLAHE operates on 8-bit integer images, so the float V in [0,1] is
scaled to uint8, equalized, then mapped back to float [0,1]. LOL inputs are
8-bit, so this quantization is faithful to the source data.
"""
import cv2
import numpy as np

TILE_GRID_SIZE = (8, 8)
DEFAULT_CLIP_LIMIT = 2.0


def enhance_value(v, clip_limit=DEFAULT_CLIP_LIMIT, tile_grid_size=TILE_GRID_SIZE):
    """Apply CLAHE to a float V channel in [0,1].

    Returns the enhanced V_CLAHE map as float32 in [0,1].
    """
    v8 = (np.clip(v, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size),
    )
    out8 = clahe.apply(v8)
    return out8.astype(np.float32) / 255.0
