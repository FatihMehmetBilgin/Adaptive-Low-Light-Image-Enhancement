"""Report-quality local-CPU runtime benchmark for Table II (proposal §V.B.4).

Honest per-frame timing of every method on eval15 frames (all 400x600):
  * time.perf_counter, enhancement op only (image file I/O excluded — frames are
    loaded into memory before timing),
  * per (method, image): 2 warm-up runs (discarded) + 20 timed repeats,
  * per-image mean over the 20 repeats, then mean +/- std across the images,
  * FPS = 1000 / mean_ms.

The proposed method uses the config selected on our485 (selected_config.json),
so this stays consistent with Table I / Table II. Environment info (CPU model,
cores, library versions) is printed for the report's "measured on local CPU" note.

Run from the repo root:  python experiments/benchmark_runtime.py
"""
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import scipy  # noqa: E402

from src import baselines, clahe_branch, data, pipeline, ssr_branch  # noqa: E402

WARMUP = 2
REPEATS = 20
N_IMAGES = 5
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def selected_config():
    """Return (clip, sigma) chosen on our485, or branch defaults if not tuned yet."""
    cfg_path = OUTPUTS / "selected_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            c = json.load(f)
        return float(c["clip_limit"]), float(c["sigma"])
    return clahe_branch.DEFAULT_CLIP_LIMIT, ssr_branch.DEFAULT_SIGMA


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment():
    return {
        "cpu": cpu_model(),
        "logical_cores": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
    }


def time_method(fn, images):
    """Per-image mean over REPEATS (after WARMUP), then mean +/- std across images."""
    per_image_means = []
    for img in images:
        for _ in range(WARMUP):
            fn(img)
        reps = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fn(img)
            reps.append((time.perf_counter() - t0) * 1000.0)
        per_image_means.append(float(np.mean(reps)))
    return float(np.mean(per_image_means)), float(np.std(per_image_means))


def main():
    OUTPUTS.mkdir(exist_ok=True)
    clip, sigma = selected_config()

    pairs = data.list_lol_pairs(split="eval")[:N_IMAGES]
    images = [data.load_image(lp) for _, lp, _ in pairs]  # I/O done before timing
    shapes = {img.shape for img in images}

    env = environment()
    print("=== Environment (local CPU benchmark) ===")
    print(f"  CPU            : {env['cpu']}")
    print(f"  Logical cores  : {env['logical_cores']}")
    print(f"  Python         : {env['python']}")
    print(f"  numpy / opencv / scipy : {env['numpy']} / {env['opencv']} / {env['scipy']}")
    print(f"  Frames         : {len(images)} eval15, shapes={shapes}")
    print(f"  Protocol       : {WARMUP} warm-up + {REPEATS} timed reps per (method, image); "
          f"proposed @ clip={clip}, sigma={sigma}\n")

    methods = {
        "Gamma Correction": baselines.gamma_correction,
        "GHE": baselines.ghe,
        "Sequential (CLAHE->SSR)":
            lambda x: baselines.sequential_clahe_ssr(x, clip, sigma),
        "Sequential (SSR->CLAHE)":
            lambda x: baselines.sequential_ssr_clahe(x, clip, sigma),
        "Proposed Parallel Fusion":
            lambda x: pipeline.enhance(x, clip, sigma),
    }

    print(f"{'Method':28s} {'mean(ms)':>9} {'std(ms)':>8} {'FPS':>7}")
    rows = []
    for name, fn in methods.items():
        mean_ms, std_ms = time_method(fn, images)
        fps = 1000.0 / mean_ms
        print(f"{name:28s} {mean_ms:9.2f} {std_ms:8.2f} {fps:7.1f}")
        rows.append({"method": name, "mean_ms": round(mean_ms, 2),
                     "std_ms": round(std_ms, 2), "fps": round(fps, 1)})

    with open(OUTPUTS / "runtime_final.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "mean_ms", "std_ms", "fps"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] saved outputs/runtime_final.csv  (measured on local CPU: {env['cpu']})")


if __name__ == "__main__":
    main()
