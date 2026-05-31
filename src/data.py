"""Data loading for the LOL low-light dataset.

Expected layout (proposal §V.A):
    data/lol/our485/{low,high}/   -> training split (hyperparameter tuning)
    data/lol/eval15/{low,high}/   -> test split (reporting)
Files are matched by identical filenames across low/ and high/. Tuning the
config on the train split and reporting on the test split avoids selecting
hyperparameters on the same images they are evaluated on (selection bias).
"""
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = _PROJECT_ROOT / "data" / "lol"
SPLITS = {"train": "our485", "eval": "eval15"}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def split_dirs(split):
    """Return ``(low_dir, high_dir)`` for a named split ('train' or 'eval')."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
    base = DATA_ROOT / SPLITS[split]
    return base / "low", base / "high"


def load_image(path, as_float=True):
    """Read an image as RGB.

    Returns float32 in [0, 1] when ``as_float`` (the canonical internal format
    used by the rest of the pipeline), otherwise the raw uint8 array.
    """
    path = str(path)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if as_float:
        return rgb.astype(np.float32) / 255.0
    return rgb


def save_image(path, rgb):
    """Write an RGB image (float [0,1] or uint8) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb)
    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def list_lol_pairs(split="eval"):
    """Return a sorted list of ``(name, low_path, high_path)`` matched pairs."""
    low_dir, high_dir = split_dirs(split)
    if not low_dir.is_dir():
        raise FileNotFoundError(
            f"LOL '{split}' low/ directory not found: {low_dir}\n"
            f"Place the LOL {SPLITS[split]} split under "
            f"data/lol/{SPLITS[split]}/{{low,high}}/."
        )
    pairs = []
    for low_path in sorted(low_dir.iterdir()):
        if low_path.suffix.lower() not in IMAGE_EXTS:
            continue
        high_path = high_dir / low_path.name
        if high_path.exists():
            pairs.append((low_path.stem, low_path, high_path))
    return pairs


def load_lol_pairs(split="eval", as_float=True, subset=None, seed=42):
    """Load matched LOL pairs as ``(name, low_rgb, high_rgb)`` tuples.

    If ``subset`` is given and smaller than the split, a reproducible random
    sample of that many pairs is loaded (seeded by ``seed``) — used to tune on a
    fast, fixed subset of the large train split.
    """
    pairs = list_lol_pairs(split)
    if subset is not None and subset < len(pairs):
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(pairs), size=subset, replace=False).tolist())
        pairs = [pairs[i] for i in idx]
    return [
        (name, load_image(lp, as_float), load_image(hp, as_float))
        for name, lp, hp in pairs
    ]
