# Adaptive Low-Light Image Enhancement Based on Hybrid CLAHE–Retinex with Value Channel Fusion in HSV Color Space

A purely mathematical, CPU-only low-light image enhancement pipeline that runs **Contrast Limited Adaptive Histogram Equalization (CLAHE)** and **Single-Scale Retinex (SSR)** *in parallel* on the HSV Value channel, then merges them with a per-image Shannon-entropy weight. Final-project submission for **COMP430 — Digital Image Processing**, Abdullah Gül University.

Full write-up: [`paper.pdf`](paper.pdf).

## Method at a glance

- Operates only on the **V (Value)** channel of HSV; Hue and Saturation are passed through untouched, preserving color fidelity.
- CLAHE and SSR branches read the **same** original V independently — neither feeds the other (the architectural distinction from the sequential chaining variants used here as ablations).
- Fusion weight α = H(V<sub>CLAHE</sub>) / (H(V<sub>CLAHE</sub>) + H(V<sub>SSR</sub>)) is computed per image from the Shannon entropy of each branch's output, so no static blend parameter must be grid-searched.
- Hyperparameters tuned on the LOL **our485** training split, evaluated **once** on **eval15** — no test-set selection bias.

## Key results (LOL eval15, mean over 15 paired images)

| Method                       | PSNR (dB) ↑ | SSIM ↑    | BRISQUE ↓ | Runtime (ms) ↓    | FPS ↑   |
|------------------------------|------------:|----------:|----------:|------------------:|--------:|
| Gamma Correction             | 11.91       | 0.501     | 22.14     | 4.67 ± 0.04       | 214.0   |
| GHE                          | 15.58       | 0.398     | 44.38     | 4.52 ± 0.10       | 221.1   |
| Sequential (CLAHE → SSR)     | 15.90       | 0.421     | 42.70     | 224.17 ± 0.74     | 4.5     |
| Sequential (SSR → CLAHE)     | 16.06       | 0.418     | 43.24     | 214.25 ± 3.03     | 4.7     |
| **Proposed Parallel Fusion** | **17.27**   | **0.526** | **34.42** | **214.61 ± 2.53** | **4.7** |

Tuned configuration: **clip = 1.5, σ = 120**. Runtime measured on Intel Core i5-11400H @ 2.70 GHz, excluding image file I/O.

The proposed method has the highest PSNR and SSIM of the five compared methods and the lowest BRISQUE among aggressive enhancement baselines (Gamma's lower BRISQUE is a documented quirk of the metric on smooth, under-enhanced dark images — see the paper). The per-frame runtime is within one standard deviation of the closer sequential ablation, so the architectural change carries **no runtime penalty**. At ≈4.7 FPS on 400×600 frames, the pipeline is positioned as a lightweight CPU method for thermally constrained edge platforms, not a real-time video processor.

### Cross-dataset generalization (LOL-v2, 100 paired test images, no retuning)

| Method                       | PSNR (dB) ↑ | SSIM ↑    | BRISQUE ↓ |
|------------------------------|------------:|----------:|----------:|
| Gamma Correction             | 14.80       | 0.537     | 30.06     |
| GHE                          | 15.13       | 0.395     | 48.87     |
| Sequential (CLAHE → SSR)     | 14.04       | 0.392     | 47.36     |
| Sequential (SSR → CLAHE)     | 14.15       | 0.388     | 49.55     |
| **Proposed Parallel Fusion** | **18.48**   | 0.526     | **41.25** |

On the larger LOL-v2 test set, applied with the LOL-tuned `(clip=1.5, σ=120)` config without any retuning, the proposed method retains the highest PSNR by a **wider +3.35 dB margin** over the next-best non-proposed method and is the best on every metric among the four aggressive enhancement baselines. Gamma's higher SSIM (0.537) / lower BRISQUE (30.06) is once again the documented under-enhancement quirk — its PSNR (14.80 dB vs. proposed 18.48 dB) shows it does not actually recover the scene.

## Repository structure

```
.
├── src/
│   ├── color.py            HSV decoupling + reconstruction
│   ├── clahe_branch.py     CLAHE on V
│   ├── ssr_branch.py       Single-Scale Retinex on V (log domain + percentile-clip normalization)
│   ├── fusion.py           Shannon-entropy adaptive fusion
│   ├── pipeline.py         End-to-end proposed method (decompose → branches → fuse → reconstruct)
│   ├── baselines.py        Gamma, GHE, Sequential CLAHE→SSR, Sequential SSR→CLAHE
│   ├── data.py             LOL loader (split = 'train' | 'eval', reproducible random subset)
│   └── metrics.py          PSNR, SSIM (skimage), BRISQUE (patched), runtime helper
├── experiments/
│   ├── run_grid_search.py      Tune (clip, σ) on our485, report on eval15
│   ├── benchmark_runtime.py    Report-quality local-CPU per-frame timing
│   └── make_figures.py         Qualitative figures + train-grid heatmap
├── main.py                 Sanity driver exercising every module
├── data/lol/               LOL dataset goes here (not redistributed; see below)
├── outputs/                Generated CSVs and figures
├── paper/                  IEEE-Conference paper (paper.tex, paper.pdf, references.bib, figures/)
└── requirements.txt
```

## Installation

The project is pinned for **Python 3.14**; pinned versions of `numpy` 2.4, `scipy` 1.17, `opencv-python` 4.13, `scikit-image` 0.26, `matplotlib` 3.10 carry pre-built wheels for that interpreter. The BRISQUE metric is provided by `brisque` 0.2.0 with `libsvm-official` 3.37.

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On other Python versions the pinned versions may need to be relaxed.

## Dataset

The LOL dataset is **not** redistributed in this repository. Download it from its source and place the splits as:

```
data/lol/our485/{low,high}/   # 485 training pairs (used only for hyperparameter tuning)
data/lol/eval15/{low,high}/   # 15 test pairs (used only for final reporting)
data/lolv2/{Low,high}/        # 100 LOL-v2 test pairs (cross-dataset generalization; no retuning)
```

For LOL and LOL-v1's `eval15` filenames must match across `low/` and `high/`. For LOL-v2 the loader pairs files by stripping the `low` / `normal` prefix, so `low00690.png` is paired with `normal00690.png`.

## Usage

### Sanity driver (exercises every module)

```bash
python main.py
```

Runs HSV round-trip, CLAHE / SSR / fusion statistics, end-to-end pipeline consistency vs. manual composition, and the four baselines; writes illustrative PNGs to `outputs/`.

### Full experiment (Table I + Table II)

```bash
python experiments/run_grid_search.py
```

Tunes (clip, σ) on a 100-image random subset of `our485` (seed 42), selects the cell with highest mean SSIM, then evaluates that single configuration plus all baselines on `eval15`. Writes `outputs/grid_search_train.csv`, `outputs/selected_config.json`, `outputs/table1_psnr_ssim.csv`, `outputs/table2_brisque_runtime.csv`.

When `outputs/runtime_final.csv` exists (next step), Table II's runtime column is sourced from it rather than from the in-script light timing.

### Cross-dataset generalization (LOL-v2)

```bash
python experiments/eval_lolv2.py
```

Reads `outputs/selected_config.json` (LOL-tuned `clip=1.5, σ=120`) and runs the proposed pipeline plus all four baselines on the 100-image LOL-v2 real-captured test split *without retuning*. Writes `outputs/table3_lolv2_psnr_ssim.csv` and `outputs/table4_lolv2_brisque.csv`. Runtime is not remeasured — the algorithms and hardware are unchanged, so the values in `outputs/runtime_final.csv` carry over.

### Report-quality runtime benchmark

```bash
python experiments/benchmark_runtime.py
```

For each method: 2 warm-up + 20 timed `time.perf_counter` repetitions per image, averaged over 5 representative `eval15` frames; mean ± std and FPS reported and saved to `outputs/runtime_final.csv`. Excludes image I/O. Environment info (CPU model, Python, library versions) is printed to the console.

### Figures

```bash
python experiments/make_figures.py
```

For the first 5 `eval15` images, produces side-by-side method comparison and per-channel branch decomposition figures, plus the train-grid PSNR/SSIM heatmap. All use the selected (clip, σ) from `selected_config.json` if present (else branch defaults).

## Outputs

After running the experiments above, `outputs/` contains:

- **CSVs:** `grid_search_train.csv`, `table1_psnr_ssim.csv`, `table2_brisque_runtime.csv`, `runtime_final.csv`.
- **JSON:** `selected_config.json` (chosen `(c, σ)`, criterion, subset size, seed).
- **Figures:** `fig_grid_heatmap.png`, `fig_method_comparison_<name>.png`, `fig_branch_decomposition_<name>.png` for each rendered eval15 image.

## Paper

Full IEEE-Conference write-up: [`paper/paper.pdf`](paper/paper.pdf). Sources in `paper/paper.tex` + `paper/references.bib`; compile from `paper/` with `latexmk -pdf paper.tex` (requires the IEEEtran LaTeX class).

## Reproducibility notes

- All randomness is seeded with `42` (exposed as `SEED` in `src/__init__.py`).
- The `brisque` 0.2.0 PyPI package has an upstream NumPy ≥ 2.0 bug in its feature fit (it divides by `array.shape`, a tuple, instead of the element count, so features come back wrapped in size-1 arrays). `src/metrics.py` applies a small, localized runtime patch that coerces these features via `np.ravel` without changing any numerical value. See the docstring of `src/metrics.py` for details.

## Author

**Fatih Mehmet Bilgin** — Department of Computer Engineering, Abdullah Gül University — `fatihmehmet.bilgin@agu.edu.tr`
