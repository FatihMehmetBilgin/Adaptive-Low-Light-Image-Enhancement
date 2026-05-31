"""Driver / sanity check.

Exercises the modules built so far:
  Module 1 — HSV decoupling round-trip (V isolation + lossless reconstruction).
  Module 2 — CLAHE branch on the V channel (local-contrast enhancement).

Run from the project root:  python main.py
"""
import numpy as np

from src import (
    SEED, baselines, clahe_branch, color, data, fusion, metrics, pipeline, ssr_branch,
)

OUTPUTS = data._PROJECT_ROOT / "outputs"


def _sample_image():
    """Return (name, rgb_float) from the LOL dataset if present, else synthetic."""
    try:
        pairs = data.list_lol_pairs()
    except FileNotFoundError:
        pairs = []
    if pairs:
        name, low_path, _ = pairs[0]
        return f"LOL/{name}", data.load_image(low_path)
    rng = np.random.default_rng(SEED)
    return "synthetic-random", rng.random((128, 128, 3), dtype=np.float32)


def check_module1(rgb):
    print("--- Module 1: HSV decoupling ---")
    h, s, v = color.decompose(rgb)
    print(
        f"[m1] decompose -> H[{h.min():.2f},{h.max():.2f}]  "
        f"S[{s.min():.3f},{s.max():.3f}]  V[{v.min():.3f},{v.max():.3f}]"
    )
    err = color.roundtrip_error(rgb)
    print(f"[m1] RGB->HSV->RGB max abs error: {err:.2e}  "
          f"[{'OK' if err < 1e-3 else 'FAIL'}]")
    h2, s2, _ = color.decompose(color.reconstruct(h, s, v))
    hs_drift = float(max(np.max(np.abs(h2 - h)), np.max(np.abs(s2 - s))))
    print(f"[m1] H/S drift after reconstruct: {hs_drift:.2e}")


def check_module2(rgb, name):
    print("--- Module 2: CLAHE branch ---")
    h, s, v = color.decompose(rgb)
    print(f"[m2] V before CLAHE: mean={v.mean():.3f}  std(contrast)={v.std():.3f}")
    for clip in (1.5, 2.0, 3.0):
        v_clahe = clahe_branch.enhance_value(v, clip_limit=clip)
        print(
            f"[m2] clip={clip}: V_CLAHE mean={v_clahe.mean():.3f}  "
            f"std={v_clahe.std():.3f}  range=[{v_clahe.min():.3f},{v_clahe.max():.3f}]"
        )

    # Save a visual (default clip) so the result can be eyeballed.
    v_clahe = clahe_branch.enhance_value(v, clip_limit=clahe_branch.DEFAULT_CLIP_LIMIT)
    rgb_clahe = color.reconstruct(h, s, v_clahe)
    safe = name.replace("/", "_")
    data.save_image(OUTPUTS / f"{safe}_orig.png", rgb)
    data.save_image(OUTPUTS / f"{safe}_clahe.png", rgb_clahe)
    print(f"[m2] saved -> outputs/{safe}_orig.png , outputs/{safe}_clahe.png")


def check_module3(rgb, name):
    print("--- Module 3: SSR branch ---")
    h, s, v = color.decompose(rgb)
    for sigma in (60, 80, 120):
        r_raw = ssr_branch.single_scale_retinex(v, sigma=sigma)
        v_ssr = ssr_branch.enhance_value(v, sigma=sigma)
        print(
            f"[m3] sigma={sigma}: raw R range=[{r_raw.min():.2f},{r_raw.max():.2f}]  "
            f"-> V_SSR mean={v_ssr.mean():.3f}  std={v_ssr.std():.3f}  "
            f"range=[{v_ssr.min():.3f},{v_ssr.max():.3f}]"
        )

    v_ssr = ssr_branch.enhance_value(v, sigma=ssr_branch.DEFAULT_SIGMA)
    rgb_ssr = color.reconstruct(h, s, v_ssr)
    safe = name.replace("/", "_")
    data.save_image(OUTPUTS / f"{safe}_ssr.png", rgb_ssr)
    print(f"[m3] saved -> outputs/{safe}_ssr.png")


def check_module4(rgb, name):
    print("--- Module 4: entropy-based adaptive fusion ---")
    h, s, v = color.decompose(rgb)
    v_clahe = clahe_branch.enhance_value(v, clip_limit=clahe_branch.DEFAULT_CLIP_LIMIT)
    v_ssr = ssr_branch.enhance_value(v, sigma=ssr_branch.DEFAULT_SIGMA)

    h_clahe = fusion.shannon_entropy(v_clahe)
    h_ssr = fusion.shannon_entropy(v_ssr)
    v_new, alpha = fusion.fuse(v_clahe, v_ssr)
    print(f"[m4] H(V_CLAHE)={h_clahe:.3f} bits  H(V_SSR)={h_ssr:.3f} bits")
    print(f"[m4] alpha = {alpha:.3f}  -> {'CLAHE-leaning' if alpha > 0.5 else 'SSR-leaning'}")
    print(f"[m4] V_new: mean={v_new.mean():.3f}  std={v_new.std():.3f}  "
          f"range=[{v_new.min():.3f},{v_new.max():.3f}]")

    # Blend must lie pixel-wise between the two branches (convex combination).
    lo = np.minimum(v_clahe, v_ssr)
    hi = np.maximum(v_clahe, v_ssr)
    within = bool(np.all(v_new >= lo - 1e-6) and np.all(v_new <= hi + 1e-6))
    print(f"[m4] V_new within [min,max] of branches: {within}")

    rgb_fused = color.reconstruct(h, s, v_new)
    safe = name.replace("/", "_")
    data.save_image(OUTPUTS / f"{safe}_fused.png", rgb_fused)
    print(f"[m4] saved -> outputs/{safe}_fused.png")


def check_module5(rgb, name):
    print("--- Module 5: full pipeline ---")
    res = pipeline.enhance_detailed(rgb)
    print(f"[m5] alpha={res.alpha:.3f}  V_new mean={res.v_new.mean():.3f} "
          f"std={res.v_new.std():.3f}  out RGB range=[{res.rgb.min():.3f},{res.rgb.max():.3f}]")

    # Consistency: pipeline output must equal manual composition of the modules.
    h, s, v = color.decompose(rgb)
    v_c = clahe_branch.enhance_value(v)
    v_r = ssr_branch.enhance_value(v)
    v_n, _ = fusion.fuse(v_c, v_r)
    manual = color.reconstruct(h, s, v_n)
    diff = float(np.max(np.abs(res.rgb - manual)))
    print(f"[m5] pipeline vs manual composition max diff: {diff:.2e}  "
          f"[{'OK' if diff < 1e-6 else 'FAIL'}]")

    safe = name.replace("/", "_")
    data.save_image(OUTPUTS / f"{safe}_pipeline.png", res.rgb)
    print(f"[m5] saved -> outputs/{safe}_pipeline.png")

    # Robustness: run end-to-end across a few dataset images if available.
    try:
        pairs = data.list_lol_pairs()
    except FileNotFoundError:
        pairs = []
    if pairs:
        print("[m5] multi-image run (first 3):")
        for nm, low_path, _ in pairs[:3]:
            out = pipeline.enhance_detailed(data.load_image(low_path))
            print(f"      {nm}: alpha={out.alpha:.3f}  out mean={out.rgb.mean():.3f}")


def check_module6(rgb, name):
    print("--- Module 6: baselines & ablations ---")
    safe = name.replace("/", "_")
    outputs = {
        "ghe": baselines.ghe(rgb),
        "gamma": baselines.gamma_correction(rgb),
        "seq_clahe_ssr": baselines.sequential_clahe_ssr(rgb),
        "seq_ssr_clahe": baselines.sequential_ssr_clahe(rgb),
    }
    for mname, out in outputs.items():
        print(f"[m6] {mname:14s}: RGB mean={out.mean():.3f}  std={out.std():.3f}  "
              f"range=[{out.min():.3f},{out.max():.3f}]")
        data.save_image(OUTPUTS / f"{safe}_{mname}.png", out)
    print(f"[m6] saved 4 baseline outputs to outputs/{safe}_*.png")


def check_module7(rgb, name):
    print("--- Module 7: metrics (PSNR / SSIM / BRISQUE / runtime) ---")
    try:
        pairs = data.list_lol_pairs()
    except FileNotFoundError:
        pairs = []
    if not pairs:
        print(f"[m7] no LOL pairs; no-reference + runtime only on '{name}':")
        print(f"[m7] BRISQUE={metrics.brisque(rgb):.2f}  "
              f"runtime={metrics.runtime_ms(pipeline.enhance, rgb, repeats=3):.1f} ms")
        return

    nm, low_path, high_path = pairs[0]
    low, high = data.load_image(low_path), data.load_image(high_path)
    candidates = {
        "low (input)": low,
        "gamma": baselines.gamma_correction(low),
        "ghe": baselines.ghe(low),
        "seq_clahe_ssr": baselines.sequential_clahe_ssr(low),
        "proposed": pipeline.enhance(low),
    }
    print(f"[m7] image {nm} — enhanced vs high reference:")
    print(f"      {'method':16s} {'PSNR(dB)':>9} {'SSIM':>7} {'BRISQUE':>8}")
    for mname, out in candidates.items():
        m = metrics.evaluate(high, out)
        print(f"      {mname:16s} {m['psnr']:9.2f} {m['ssim']:7.3f} {m['brisque']:8.2f}")

    rt = metrics.runtime_ms(pipeline.enhance, low, repeats=5)
    print(f"[m7] proposed pipeline runtime: {rt:.1f} ms/frame (local CPU; final on Colab)")


def main():
    np.random.seed(SEED)
    name, rgb = _sample_image()
    print(f"[sanity] sample image: {name}  shape={rgb.shape}  "
          f"dtype={rgb.dtype}  range=[{rgb.min():.3f},{rgb.max():.3f}]")
    check_module1(rgb)
    check_module2(rgb, name)
    check_module3(rgb, name)
    check_module4(rgb, name)
    check_module5(rgb, name)
    check_module6(rgb, name)
    check_module7(rgb, name)
    print("[sanity] checks complete.")


if __name__ == "__main__":
    main()
