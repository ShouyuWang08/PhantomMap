"""
Spatial atlas: aggregate phantom bboxes into (normalised) 2D KDE heatmaps.

Produces:
  * results/atlas_stats.json — per-model-per-benchmark summary statistics
  * report/figures/fig3_atlas.pdf — the 2x2 hero figure

Two datasets per panel:
  1. Phantom boxes  (pred=yes, gt=no) — the key finding.
  2. Honest boxes   (pred=yes, gt=yes) — control, shows where real objects
     typically are, for comparison.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import load_samples


# Pretty names for the paper.
MODEL_SHORT = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA-NeXT-7B",
}


def kde_grid(cx: np.ndarray, cy: np.ndarray, n: int = 200) -> np.ndarray:
    """Evaluate a 2D Gaussian KDE on the unit square.

    Uses an adaptive bandwidth that grows when the sample is small, so a
    low-n panel (e.g. LLaVA phantoms, n=71) produces a smooth continuous
    density rather than a discrete bump pattern. Falls back to a
    smoothed 2D histogram when the data covariance is rank-deficient
    (which happens when VLMs emit quantised bbox coordinates).
    """
    if len(cx) < 3:
        return np.zeros((n, n), dtype=np.float32)
    rng = np.random.default_rng(0)
    jitter = 1e-3
    cx_j = cx + rng.normal(0, jitter, size=cx.shape)
    cy_j = cy + rng.normal(0, jitter, size=cy.shape)
    pts = np.stack([cx_j, cy_j])
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.stack([X.ravel(), Y.ravel()])
    # Larger bandwidth for small samples so we display the population
    # structure rather than individual bumps.
    if len(cx) < 100:
        bw = 0.30
    elif len(cx) < 300:
        bw = 0.22
    else:
        bw = 0.15
    try:
        kde = gaussian_kde(pts, bw_method=bw)
        Z = kde(grid_pts).reshape(n, n)
    except np.linalg.LinAlgError:
        from scipy.ndimage import gaussian_filter
        H, _, _ = np.histogram2d(cx_j, cy_j, bins=n, range=[[0, 1], [0, 1]])
        Z = gaussian_filter(H.T, sigma=max(1.0, n * 0.02))
    return Z.astype(np.float32)


def _centers(samples) -> tuple[np.ndarray, np.ndarray]:
    cx = np.array([s.feats[0] for s in samples], dtype=np.float32)
    cy = np.array([s.feats[1] for s in samples], dtype=np.float32)
    return cx, cy


def center_bias(cx: np.ndarray, cy: np.ndarray) -> float:
    """Mean distance of bbox centers to the image center (0.5, 0.5)."""
    if len(cx) == 0:
        return float("nan")
    d = np.hypot(cx - 0.5, cy - 0.5)
    return float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out-fig", type=Path, default=Path("report/figures/fig3_atlas.pdf"))
    ap.add_argument("--out-stats", type=Path, default=Path("results/atlas_stats.json"))
    args = ap.parse_args()

    # Load all samples once, then split by (model, label).
    all_samples = load_samples(args.inputs, phantom_only=False)
    by_model: dict[str, dict[str, list]] = {}
    for s in all_samples:
        by_model.setdefault(s.model_id, {"phantom": [], "honest": []})
        if s.label == 1:
            by_model[s.model_id]["phantom"].append(s)
        else:
            by_model[s.model_id]["honest"].append(s)

    models = [m for m in MODEL_SHORT if m in by_model]
    if not models:
        raise SystemExit("No recognised models in inputs.")

    # Stats.
    stats = {}
    for m in models:
        for kind in ("phantom", "honest"):
            cx, cy = _centers(by_model[m][kind])
            stats[f"{MODEL_SHORT[m]}__{kind}"] = {
                "n": int(len(cx)),
                "mean_d_center": center_bias(cx, cy),
                "cx_mean": float(cx.mean()) if len(cx) else float("nan"),
                "cy_mean": float(cy.mean()) if len(cy) else float("nan"),
                "area_mean": float(np.mean([s.feats[4] for s in by_model[m][kind]]))
                if by_model[m][kind]
                else float("nan"),
            }
    args.out_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Figure: 2 rows (Qwen, LLaVA) x 2 cols (phantom, honest).
    # Sized to fit a single CVPR column (~3.3 in) at width=\columnwidth;
    # two-line titles prevent horizontal clash between adjacent panels.
    fig, axes = plt.subplots(
        nrows=len(models), ncols=2, figsize=(4.4, 2.0 * len(models)), squeeze=False
    )
    for row, m in enumerate(models):
        for col, kind in enumerate(("phantom", "honest")):
            ax = axes[row, col]
            cx, cy = _centers(by_model[m][kind])
            Z = kde_grid(cx, cy, n=150)
            ax.imshow(
                np.flipud(Z),
                extent=(0, 1, 0, 1),
                cmap="magma",
                aspect="equal",
            )
            ax.scatter(cx, 1 - cy, s=2, c="white", alpha=0.25, linewidths=0)
            mean_d = center_bias(cx, cy)
            ax.set_title(
                f"{MODEL_SHORT[m]} / {kind}\n"
                f"$n={len(cx)}$,  $\\bar{{d}}={mean_d:.3f}$",
                fontsize=9,
                pad=4,
            )
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_xlabel("normalised $x$", fontsize=8)
            ax.set_ylabel("normalised $y$", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.subplots_adjust(wspace=0.25, hspace=0.45)
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, bbox_inches="tight")
    print(f"wrote {args.out_fig}")
    print(f"wrote {args.out_stats}")


if __name__ == "__main__":
    main()
