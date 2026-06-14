"""Cross-dataset generalization on LOL-v2 (100 paired test images).

Methodology (no test-set selection bias):
  * Use the (clip, sigma) selected on the LOL-v1 our485 train subset
    (read from outputs/selected_config.json) WITHOUT retuning on LOL-v2.
  * Report PSNR / SSIM / BRISQUE for the proposed method and all baselines
    on the LOL-v2 100-image test split.

Runtime is NOT remeasured here: the algorithms and hardware are unchanged,
so the report-quality timings in outputs/runtime_final.csv (LOL-v1) carry
over.

Outputs under outputs/:
  * table3_lolv2_psnr_ssim.csv
  * table4_lolv2_brisque.csv

Run from the repo root:  python experiments/eval_lolv2.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import SEED, baselines, data, metrics, pipeline  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def _load_selected_config():
    cfg_path = OUTPUTS / "selected_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found — run experiments/run_grid_search.py first."
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    return float(cfg["clip_limit"]), float(cfg["sigma"])


def _mean(records, key):
    return float(np.mean([r[key] for r in records]))


def _per_image(method_fn, pairs):
    out = []
    for _, low, high in pairs:
        enh = method_fn(low)
        out.append({
            "psnr": metrics.psnr(high, enh),
            "ssim": metrics.ssim(high, enh),
            "brisque": metrics.brisque(enh),
        })
    return out


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    np.random.seed(SEED)
    OUTPUTS.mkdir(exist_ok=True)

    clip_star, sigma_star = _load_selected_config()
    print(f"[lolv2] using LOL-v1-selected config: clip={clip_star}, sigma={sigma_star}")

    pairs = data.load_lol_pairs(split="eval", dataset="lolv2")
    print(f"[lolv2] loaded {len(pairs)} LOL-v2 paired test images")

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

    print(f"\n=== TABLE III — PSNR / SSIM (LOL-v2, mean over {len(pairs)} images;"
          f" proposed @ clip={clip_star}, sigma={sigma_star}, no retuning) ===")
    print(f"{'Method':28s} {'PSNR(dB)':>9} {'SSIM':>7} {'BRISQUE':>8}")
    t_psnr_ssim, t_brisque = [], []
    for name, fn in methods.items():
        per = _per_image(fn, pairs)
        psnr_m = _mean(per, "psnr")
        ssim_m = _mean(per, "ssim")
        brisque_m = _mean(per, "brisque")
        print(f"{name:28s} {psnr_m:9.2f} {ssim_m:7.3f} {brisque_m:8.2f}")
        t_psnr_ssim.append({"method": name,
                            "psnr": round(psnr_m, 2),
                            "ssim": round(ssim_m, 3)})
        t_brisque.append({"method": name, "brisque": round(brisque_m, 2)})

    _write_csv(OUTPUTS / "table3_lolv2_psnr_ssim.csv",
               ["method", "psnr", "ssim"], t_psnr_ssim)
    _write_csv(OUTPUTS / "table4_lolv2_brisque.csv",
               ["method", "brisque"], t_brisque)
    print("\n[done] saved table3_lolv2_psnr_ssim.csv, table4_lolv2_brisque.csv to outputs/")


if __name__ == "__main__":
    main()
