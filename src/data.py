"""Data loading for paired low-light benchmarks (LOL, LOL-v2).

Layouts:
    data/lol/our485/{low,high}/    LOL-v1 training split (hyperparameter tuning)
    data/lol/eval15/{low,high}/    LOL-v1 test split (reporting)
    data/lolv2/{Low,high}/         LOL-v2 100-pair test split (cross-dataset
                                   generalization; no train split here)

Pair matching: LOL-v1 matches by identical stems across low/ and high/. LOL-v2
matches by stripping the role prefix ("low" / "normal") so that ``low00690.png``
and ``normal00690.png`` are recognized as the same scene.
"""
import re
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# Per-dataset layout descriptors.
# ``splits`` maps logical split name -> (low_dir, high_dir) relative to root.
# ``match`` extracts a join key from a stem given its role ("low" or "high").
_LOL_V1 = {
    "root": _PROJECT_ROOT / "data" / "lol",
    "splits": {
        "train": ("our485/low", "our485/high"),
        "eval":  ("eval15/low", "eval15/high"),
    },
    "match": lambda stem, role: stem,
}

_LOLV2_PREFIX = re.compile(r"^(low|normal)", re.IGNORECASE)
_LOL_V2 = {
    "root": _PROJECT_ROOT / "data" / "lolv2",
    "splits": {
        "eval": ("Low", "high"),
    },
    "match": lambda stem, role: _LOLV2_PREFIX.sub("", stem),
}

DATASETS = {"lol": _LOL_V1, "lolv2": _LOL_V2}


def _dataset(name):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {list(DATASETS)}")
    return DATASETS[name]


def split_dirs(split, dataset="lol"):
    """Return ``(low_dir, high_dir)`` for a named split of ``dataset``."""
    ds = _dataset(dataset)
    if split not in ds["splits"]:
        raise ValueError(
            f"dataset {dataset!r} does not define split {split!r}; "
            f"available: {list(ds['splits'])}"
        )
    low_rel, high_rel = ds["splits"][split]
    return ds["root"] / low_rel, ds["root"] / high_rel


def load_image(path, as_float=True):
    """Read an image as RGB (float32 in [0,1] when ``as_float``, else uint8)."""
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


def list_lol_pairs(split="eval", dataset="lol"):
    """Return a sorted list of ``(name, low_path, high_path)`` matched pairs."""
    ds = _dataset(dataset)
    low_dir, high_dir = split_dirs(split, dataset)
    if not low_dir.is_dir():
        raise FileNotFoundError(
            f"{dataset} '{split}' low/ directory not found: {low_dir}"
        )
    if not high_dir.is_dir():
        raise FileNotFoundError(
            f"{dataset} '{split}' high/ directory not found: {high_dir}"
        )

    match = ds["match"]
    high_index = {}
    for hp in sorted(high_dir.iterdir()):
        if hp.suffix.lower() not in IMAGE_EXTS:
            continue
        high_index[match(hp.stem, "high")] = hp

    pairs = []
    for lp in sorted(low_dir.iterdir()):
        if lp.suffix.lower() not in IMAGE_EXTS:
            continue
        key = match(lp.stem, "low")
        hp = high_index.get(key)
        if hp is not None:
            pairs.append((lp.stem, lp, hp))
    return pairs


def load_lol_pairs(split="eval", as_float=True, subset=None, seed=42, dataset="lol"):
    """Load matched pairs as ``(name, low_rgb, high_rgb)`` tuples.

    If ``subset`` is given and smaller than the split, a reproducible random
    sample of that many pairs is loaded (seeded by ``seed``).
    """
    pairs = list_lol_pairs(split, dataset=dataset)
    if subset is not None and subset < len(pairs):
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(pairs), size=subset, replace=False).tolist())
        pairs = [pairs[i] for i in idx]
    return [
        (name, load_image(lp, as_float), load_image(hp, as_float))
        for name, lp, hp in pairs
    ]
