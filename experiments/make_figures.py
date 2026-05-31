"""Qualitative figures for the report.

Saves under outputs/:
  * fig_method_comparison_<name>.png   — input | gamma | ghe | seq | proposed | GT
  * fig_branch_decomposition_<name>.png — V, V_CLAHE, V_SSR, V_new + output RGB
  * fig_grid_heatmap.png               — PSNR & SSIM over the train grid
                                         (reads grid_search_train.csv)

The proposed/sequential panels use the config selected on our485
(outputs/selected_config.json); without it they fall back to branch defaults.

Run from the repo root:  python experiments/make_figures.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import baselines, clahe_branch, data, pipeline, ssr_branch  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
N_FIGURES = 5  # number of eval15 images to render per-image figures for


def selected_config():
    """Return (clip, sigma) chosen on our485, or branch defaults if not tuned yet."""
    cfg_path = OUTPUTS / "selected_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            c = json.load(f)
        return float(c["clip_limit"]), float(c["sigma"])
    return clahe_branch.DEFAULT_CLIP_LIMIT, ssr_branch.DEFAULT_SIGMA


def method_comparison(name, low, high, clip, sigma):
    panels = [
        ("Input (low)", low),
        ("Gamma", baselines.gamma_correction(low)),
        ("GHE", baselines.ghe(low)),
        ("Seq CLAHE->SSR", baselines.sequential_clahe_ssr(low, clip, sigma)),
        ("Proposed", pipeline.enhance(low, clip, sigma)),
        ("Ground truth", high),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.3))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    path = OUTPUTS / f"fig_method_comparison_{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def branch_decomposition(name, low, clip, sigma):
    res = pipeline.enhance_detailed(low, clip_limit=clip, sigma=sigma)
    vmaps = [
        ("V (input)", res.v),
        ("V_CLAHE", res.v_clahe),
        ("V_SSR", res.v_ssr),
        (f"V_new (alpha={res.alpha:.2f})", res.v_new),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.3))
    for ax, (title, vmap) in zip(axes[:4], vmaps):
        ax.imshow(vmap, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    axes[4].imshow(np.clip(res.rgb, 0, 1))
    axes[4].set_title("Output RGB", fontsize=10)
    axes[4].axis("off")
    fig.tight_layout()
    path = OUTPUTS / f"fig_branch_decomposition_{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def grid_heatmap():
    csv_path = OUTPUTS / "grid_search_train.csv"
    if not csv_path.exists():
        print("  (skip heatmap: run experiments/run_grid_search.py first)")
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    clips = sorted({float(r["clip_limit"]) for r in rows})
    sigmas = sorted({float(r["sigma"]) for r in rows})

    def grid_of(key):
        M = np.zeros((len(clips), len(sigmas)))
        for r in rows:
            M[clips.index(float(r["clip_limit"])), sigmas.index(float(r["sigma"]))] = float(r[key])
        return M

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ("psnr", "ssim"), ("PSNR (dB) — our485", "SSIM — our485")):
        M = grid_of(key)
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(sigmas)), [f"{int(s)}" for s in sigmas])
        ax.set_yticks(range(len(clips)), [f"{c}" for c in clips])
        ax.set_xlabel("SSR sigma")
        ax.set_ylabel("CLAHE clip limit")
        ax.set_title(title)
        for i in range(len(clips)):
            for j in range(len(sigmas)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = OUTPUTS / "fig_grid_heatmap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUTS.mkdir(exist_ok=True)
    clip, sigma = selected_config()
    print(f"[figs] using config clip={clip}, sigma={sigma}")

    # Per-image qualitative figures for the first N eval15 (test) images only.
    pairs = data.list_lol_pairs(split="eval")[:N_FIGURES]
    generated = []
    for name, low_path, high_path in pairs:
        low, high = data.load_image(low_path), data.load_image(high_path)
        print("saved:", method_comparison(name, low, high, clip, sigma))
        print("saved:", branch_decomposition(name, low, clip, sigma))
        generated.append(name)

    # Single grid heatmap (unchanged).
    heatmap = grid_heatmap()
    if heatmap:
        print("saved:", heatmap)

    print(f"[figs] generated per-image figures for {len(generated)} "
          f"eval15 images: {generated}")


if __name__ == "__main__":
    main()
