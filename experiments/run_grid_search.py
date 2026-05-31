"""Grid search (train) + baseline comparison (test) on the LOL dataset.

Methodology (avoids hyperparameter selection bias):
  * TUNE  on the our485 *train* split — pick the (clip, sigma) with the best mean
    SSIM over a fixed random subset (default 100 images, seed=42).
  * REPORT that single fixed config on the eval15 *test* split.

Produces under outputs/:
  * grid_search_train.csv      — full clip x sigma grid on the train subset
  * selected_config.json       — the chosen (clip, sigma) and how it was chosen
  * table1_psnr_ssim.csv       — Table I  (PSNR / SSIM) on eval15
  * table2_brisque_runtime.csv — Table II (BRISQUE / runtime) on eval15

The Table II runtime column prefers the robust benchmark in
outputs/runtime_final.csv (from benchmark_runtime.py) when present, so re-running
this grid search does not overwrite those report-quality timings with the lighter
in-script measurement; methods absent from that file fall back to light timing.

Baselines (GHE, gamma) and the sequential ablations are also reported on eval15;
the ablations use the *same* selected (clip, sigma) so the only variable versus
the proposed method is parallel-vs-sequential architecture.

Run from the repo root:  python experiments/run_grid_search.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import SEED, baselines, data, metrics, pipeline  # noqa: E402

CLIP_LIMITS = (1.5, 2.0, 3.0)
SIGMAS = (60, 80, 120)
TRAIN_SUBSET = 100
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def _mean(records, key):
    return float(np.mean([r[key] for r in records]))


def _robust_runtimes():
    """Load {method: mean_ms} from outputs/runtime_final.csv if present, else {}."""
    path = OUTPUTS / "runtime_final.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        return {row["method"]: float(row["mean_ms"]) for row in csv.DictReader(f)}


def _quality(reference, enhanced, full):
    q = {"psnr": metrics.psnr(reference, enhanced), "ssim": metrics.ssim(reference, enhanced)}
    if full:
        q["brisque"] = metrics.brisque(enhanced)
    return q


def evaluate_method(method_fn, pairs, full=True):
    """Mean metrics of a method (low_rgb -> enhanced_rgb) over pairs.

    ``full`` adds BRISQUE (no-reference, slow); tuning uses full=False so only
    the full-reference PSNR/SSIM are computed on the train subset.
    """
    per_image = [_quality(high, method_fn(low), full) for _, low, high in pairs]
    keys = ("psnr", "ssim", "brisque") if full else ("psnr", "ssim")
    return {k: _mean(per_image, k) for k in keys}


def run_grid(pairs, full=False):
    rows = []
    for clip in CLIP_LIMITS:
        for sigma in SIGMAS:
            res = evaluate_method(
                lambda low, c=clip, s=sigma: pipeline.enhance(low, clip_limit=c, sigma=s),
                pairs, full=full,
            )
            res["clip_limit"], res["sigma"] = clip, sigma
            rows.append(res)
            print(f"  clip={clip} sigma={sigma}:  PSNR={res['psnr']:.2f}  SSIM={res['ssim']:.3f}")
    return rows


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    np.random.seed(SEED)
    OUTPUTS.mkdir(exist_ok=True)

    # --- 1) TUNE on the our485 train subset (PSNR/SSIM only -> fast) ---
    train = data.load_lol_pairs(split="train", subset=TRAIN_SUBSET, seed=SEED)
    print(f"[tune] tuning on {len(train)} our485 images "
          f"(subset={TRAIN_SUBSET}, seed={SEED})\n[tune] proposed clip x sigma grid:")
    grid_rows = run_grid(train, full=False)
    _write_csv(OUTPUTS / "grid_search_train.csv",
               ["clip_limit", "sigma", "psnr", "ssim"], grid_rows)

    best = max(grid_rows, key=lambda r: r["ssim"])
    clip_star, sigma_star = best["clip_limit"], best["sigma"]
    print(f"[tune] selected by mean SSIM: clip={clip_star} sigma={sigma_star} "
          f"(train SSIM={best['ssim']:.3f}, PSNR={best['psnr']:.2f})")
    with open(OUTPUTS / "selected_config.json", "w") as f:
        json.dump({"clip_limit": clip_star, "sigma": sigma_star,
                   "selected_on": "our485", "train_subset": TRAIN_SUBSET,
                   "seed": SEED, "criterion": "mean SSIM"}, f, indent=2)

    # --- 2) REPORT the fixed config on the eval15 test split ---
    pairs = data.load_lol_pairs(split="eval")
    print(f"\n[eval] reporting on {len(pairs)} eval15 images @ "
          f"clip={clip_star}, sigma={sigma_star}")
    methods = {
        "Gamma Correction": baselines.gamma_correction,
        "GHE": baselines.ghe,
        "Sequential (CLAHE->SSR)":
            lambda x: baselines.sequential_clahe_ssr(x, clip_star, sigma_star),
        "Sequential (SSR->CLAHE)":
            lambda x: baselines.sequential_ssr_clahe(x, clip_star, sigma_star),
        "Proposed Parallel Fusion":
            lambda x: pipeline.enhance(x, clip_star, sigma_star),
    }
    quality = {name: evaluate_method(fn, pairs, full=True) for name, fn in methods.items()}

    # Runtime: prefer the robust benchmark (runtime_final.csv); fall back to a
    # light in-script timing for any method missing from it.
    rep = pairs[0][1]  # eval15 frames share one size, so timing is size-stable
    robust = _robust_runtimes()
    runtimes, rt_source = {}, {}
    for name, fn in methods.items():
        if name in robust:
            runtimes[name], rt_source[name] = robust[name], "robust"
        else:
            runtimes[name], rt_source[name] = metrics.runtime_ms(fn, rep, repeats=5), "light"
    if robust:
        n_light = sum(s == "light" for s in rt_source.values())
        print(f"[eval] runtime: from runtime_final.csv (robust)"
              + (f"; {n_light} method(s) fell back to light timing" if n_light else ""))
    else:
        print("[eval] runtime: own light timing (repeats=5); "
              "run benchmark_runtime.py for robust values")

    print(f"\n=== TABLE I — PSNR / SSIM (eval15, mean; proposed @ clip={clip_star}, "
          f"sigma={sigma_star} tuned on our485) ===")
    print(f"{'Method':28s} {'PSNR(dB)':>9} {'SSIM':>7}")
    t1 = []
    for name in methods:
        q = quality[name]
        print(f"{name:28s} {q['psnr']:9.2f} {q['ssim']:7.3f}")
        t1.append({"method": name, "psnr": round(q["psnr"], 2), "ssim": round(q["ssim"], 3)})

    print("\n=== TABLE II — BRISQUE / Runtime (eval15, mean) ===")
    print(f"{'Method':28s} {'BRISQUE':>8} {'Runtime(ms)':>12}")
    t2 = []
    for name in methods:
        b, rt = quality[name]["brisque"], runtimes[name]
        decimals = 2 if rt_source[name] == "robust" else 1
        print(f"{name:28s} {b:8.2f} {rt:12.2f}")
        t2.append({"method": name, "brisque": round(b, 2), "runtime_ms": round(rt, decimals)})

    _write_csv(OUTPUTS / "table1_psnr_ssim.csv", ["method", "psnr", "ssim"], t1)
    _write_csv(OUTPUTS / "table2_brisque_runtime.csv",
               ["method", "brisque", "runtime_ms"], t2)
    print("\n[done] saved grid_search_train.csv, selected_config.json, "
          "table1_psnr_ssim.csv, table2_brisque_runtime.csv to outputs/")
    print("[done] runtime measured on local CPU "
          "(robust values from runtime_final.csv when present).")


if __name__ == "__main__":
    main()
