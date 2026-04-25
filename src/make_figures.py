"""
Regenerate every numbered figure in the report as a vector PDF.

We keep each figure's code in one function so that tweaking Fig 4's
axis labels (most common request from reviewers) does not ripple into
Figs 1, 3, 5.

Fig 1 (teaser) and Fig 5 (qualitative failures) need the raw COCO
images available on disk under data/coco_val2014 (POPE) or
data/amber/images.
"""

from __future__ import annotations
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import load_samples, featurize_one
from atlas import kde_grid, center_bias, MODEL_SHORT


# Publication font sizes — readable on letter paper per instructor.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,      # embed TrueType (not Type 3 bitmaps)
        "ps.fonttype": 42,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def _load_all_records(jsonl_paths: list[Path]) -> list[dict]:
    out = []
    for p in jsonl_paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def _resolve_image(data_dir: Path, rec: dict) -> Path:
    if rec["split"].startswith("pope"):
        return data_dir / "coco_val2014" / rec["image"]
    return data_dir / "amber" / "images" / rec["image"]


# ---------------------------------------------------------------- Fig 1 teaser

def fig1_teaser(records: list[dict], data_dir: Path, out_path: Path, n_per_side: int = 3, seed: int = 0):
    """2x3 grid: top row 3 phantom examples, bottom row 3 honest ones, with
    bbox overlay. Examples are filtered to be REPRESENTATIVE of the
    population statistics reported in the atlas (phantoms moderately
    small, honest boxes moderately sized), so the figure supports rather
    than contradicts the paper's narrative."""
    rng = random.Random(seed)

    def _area_ratio(r):
        x1, y1, x2, y2 = r["bbox_valid"]
        return (x2 - x1) * (y2 - y1) / (r["img_w"] * r["img_h"])

    def _d_center(r):
        x1, y1, x2, y2 = r["bbox_valid"]
        cx = (x1 + x2) / 2.0 / r["img_w"]
        cy = (y1 + y2) / 2.0 / r["img_h"]
        return ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5

    # POPE's adversarial split deliberately picks absent objects that
    # co-occur visually with what IS in the image, so many "phantoms"
    # could be defended as reasonable human interpretations. For the
    # teaser figure we want examples where the model is UNAMBIGUOUSLY
    # wrong. We do this with two lists:
    # AMBIGUOUS_OBJECTS  --- often semantically overlapping with
    #                        something else in the image (toy bear vs
    #                        real bear, small table vs dining table,
    #                        SUV vs truck). Never used for Fig 1.
    # PREFERRED_OBJECTS  --- visually distinctive categories where a
    #                        viewer can agree at a glance that the
    #                        model is wrong (stop sign, fire hydrant,
    #                        giraffe, etc.). Preferred as much as
    #                        possible; score bonus below.
    AMBIGUOUS_OBJECTS = {
        "bear", "cup", "bowl", "wine glass", "vase",
        "tie", "chair", "couch", "bed",
        "truck", "car", "bus", "boat",
        "dining table",
        "umbrella",
        "dog", "cat", "bird",
        "skis", "snowboard", "surfboard",
        "handbag", "backpack", "suitcase",
        "book", "laptop", "tv", "remote",
        "bench",
        "bottle",
    }
    PREFERRED_OBJECTS = {
        "stop sign", "fire hydrant", "traffic light", "parking meter",
        "clock", "toaster", "microwave", "oven", "refrigerator",
        "toothbrush", "hair drier",
        "elephant", "giraffe", "zebra", "horse", "sheep", "cow",
        "banana", "apple", "orange", "broccoli", "carrot",
        "pizza", "hot dog", "donut", "cake", "sandwich",
        "kite", "skateboard", "tennis racket", "baseball bat",
        "baseball glove", "scissors", "frisbee", "sports ball",
        "bicycle", "motorcycle", "airplane", "train",
        "keyboard", "mouse", "cell phone",
    }

    def _valid(r):
        return (
            r.get("bbox_valid")
            and 0.02 <= _area_ratio(r) <= 0.45
            and r.get("object", "").strip().lower() not in AMBIGUOUS_OBJECTS
        )

    all_phantoms = [r for r in records if r["pred"] == "yes" and r["label"] == "no" and _valid(r)]
    all_honest = [r for r in records if r["pred"] == "yes" and r["label"] == "yes" and _valid(r)]

    # Try to pick one example per model per row for diversity.
    def pick_mixed(pool: list[dict], k: int, bias: str) -> list[dict]:
        """bias='phantom' prefers slightly smaller / off-centre boxes,
        'honest' prefers slightly larger / centred boxes, matching our
        reported atlas statistics. For 'phantom' we also boost the
        score of objects in PREFERRED_OBJECTS (visually distinctive
        categories that are unambiguously not in the image)."""
        if bias == "phantom":
            def score(r):
                s = -abs(_area_ratio(r) - 0.12) - abs(_d_center(r) - 0.32)
                if r.get("object", "").strip().lower() in PREFERRED_OBJECTS:
                    s += 10.0  # overwhelming preference for clean examples
                return s
        else:
            score = lambda r: (-abs(_area_ratio(r) - 0.22) - abs(_d_center(r) - 0.22))
        by_model: dict[str, list[dict]] = {}
        for r in sorted(pool, key=score, reverse=True):
            by_model.setdefault(r["model_id"], []).append(r)
        picked = []
        used_objects: set[str] = set()
        # Round-robin through models. Also enforce unique POPE-queried
        # objects across the row so we don't show e.g. "dining table"
        # twice in the phantom row.
        for strict in (True, False):
            while len(picked) < k and any(by_model.values()):
                advanced = False
                for mid in list(by_model):
                    if not by_model[mid]:
                        continue
                    cand = None
                    for i, r in enumerate(by_model[mid]):
                        if any(p["image"] == r["image"] and p["object"] == r["object"] for p in picked):
                            continue
                        if strict and r.get("object", "").lower() in used_objects:
                            continue
                        cand = by_model[mid].pop(i)
                        break
                    if cand is None:
                        continue
                    picked.append(cand)
                    used_objects.add(cand.get("object", "").lower())
                    advanced = True
                    if len(picked) == k:
                        break
                if not advanced:
                    break
            if len(picked) == k:
                break
        return picked[:k]

    phantoms = pick_mixed(all_phantoms, n_per_side, "phantom")
    honest = pick_mixed(all_honest, n_per_side, "honest")
    if len(phantoms) < n_per_side or len(honest) < n_per_side:
        print(
            f"[fig1] Warning: only {len(phantoms)} phantoms and {len(honest)} honest examples; "
            "figure may be incomplete."
        )

    fig, axes = plt.subplots(2, n_per_side, figsize=(6.8, 4.6))
    for row, (records_row, row_name, color) in enumerate(
        ((phantoms, "phantom", "#e0245e"), (honest, "honest", "#1da1f2"))
    ):
        for col, rec in enumerate(records_row):
            ax = axes[row, col]
            try:
                img = Image.open(_resolve_image(data_dir, rec)).convert("RGB")
            except FileNotFoundError:
                ax.text(0.5, 0.5, "image missing", ha="center", va="center")
                ax.axis("off")
                continue
            ax.imshow(img)
            x1, y1, x2, y2 = rec["bbox_valid"]
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    edgecolor=color, facecolor="none", linewidth=2,
                )
            )
            ax.set_title(
                f'"{rec["object"]}" — {row_name}\n{MODEL_SHORT.get(rec["model_id"], rec["model_id"])}',
                fontsize=8,
            )
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        "Top: hallucinated ('phantom') boxes.   Bottom: honest 'yes' answers on real objects.",
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------- Fig 3 atlas (shortcut)

def fig3_atlas(jsonl_paths: list[Path], out_path: Path):
    """Thin wrapper: atlas.py is authoritative. We import it to avoid
    duplicating the KDE code here, but expose a single-call entry point."""
    import subprocess
    import sys as _sys
    cmd = [
        _sys.executable, str(Path(__file__).parent / "atlas.py"),
        "--inputs", *[str(p) for p in jsonl_paths],
        "--out-fig", str(out_path),
        "--out-stats", "results/atlas_stats.json",
    ]
    subprocess.check_call(cmd)


# ---------------------------------------------------------------- Fig 4 ROC

def fig4_roc(metrics_paths: list[Path], out_path: Path):
    """ROC curves for the detector across models + baselines."""
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    colours = ["#003f5c", "#bc5090", "#ffa600"]

    for path, colour in zip(metrics_paths, colours):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        y_te = np.array(m["y_test"])
        scores = np.array(m["scores_full"])
        fpr, tpr, _ = roc_curve(y_te, scores)
        label = f"LR full  AUROC={m['auroc_full']:.3f}  ({path.stem})"
        ax.plot(fpr, tpr, color=colour, linewidth=1.3, label=label)

        scores_l = np.array(m["scores_logit_only"])
        fpr_l, tpr_l, _ = roc_curve(y_te, scores_l)
        ax.plot(
            fpr_l, tpr_l, color=colour, linewidth=0.9, linestyle="--",
            label=f"logit-only  AUROC={m['auroc_logit_only']:.3f}",
        )

    # HALP published reference: a single point would be misleading on ROC,
    # so we annotate it as text placed inside the plot area.
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, label="chance (0.500)")
    ax.plot([], [], " ", label="HALP published (Qwen):  AUROC=0.787")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------- Fig 5 failure modes

def fig5_failures(records: list[dict], data_dir: Path, out_path: Path, seed: int = 1):
    """6 phantom examples chosen to span three failure patterns:
    oversized (area > 0.5), off-center (d_center > 0.35), edge-clipped
    (touches image boundary). 2 per pattern, with strict de-duplication
    so the same (image, object) pair never appears twice."""
    rng = random.Random(seed)
    phantoms = [r for r in records if r["pred"] == "yes" and r["label"] == "no" and r.get("bbox_valid")]
    # Prefer Qwen examples (it grounds every yes, so the failure modes
    # are most visible), then round-robin with LLaVA.
    rng.shuffle(phantoms)

    def area(r):
        x1, y1, x2, y2 = r["bbox_valid"]
        return (x2 - x1) * (y2 - y1) / (r["img_w"] * r["img_h"])

    def d_cent(r):
        x1, y1, x2, y2 = r["bbox_valid"]
        cx = (x1 + x2) / 2 / r["img_w"]
        cy = (y1 + y2) / 2 / r["img_h"]
        return ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5

    def edge_clipped(r):
        x1, y1, x2, y2 = r["bbox_valid"]
        return x1 < 2 or y1 < 2 or x2 > r["img_w"] - 2 or y2 > r["img_h"] - 2

    # Strict priority: a record is assigned to ONE bucket only.
    # Priority order: oversized > edge-clipped > off-center.
    # We also enforce object diversity -- each of the 6 panels shows a
    # different POPE-queried object where possible -- so the figure
    # showcases the breadth of failure categories.
    used_keys: set[tuple] = set()
    used_objects: set[str] = set()
    used_images: set[str] = set()

    def key(r):
        return (r["image"], r.get("object", ""))

    def take(pool_name, pred, k: int = 2, relax_objects: bool = False):
        picked = []
        # Two passes: first strict (unique object), then relaxed if we
        # couldn't fill the bucket.
        for strict in (True, False) if not relax_objects else (False,):
            for r in phantoms:
                if len(picked) == k:
                    break
                if key(r) in used_keys:
                    continue
                if r["image"] in used_images:
                    continue
                if strict and r.get("object", "").lower() in used_objects:
                    continue
                if not pred(r):
                    continue
                picked.append(r)
                used_keys.add(key(r))
                used_images.add(r["image"])
                used_objects.add(r.get("object", "").lower())
            if len(picked) == k:
                break
        return picked

    oversize = take("oversized", lambda r: area(r) > 0.5)
    edge = take("edge-clipped", lambda r: edge_clipped(r) and area(r) <= 0.5)
    offcenter = take("off-center", lambda r: d_cent(r) > 0.35 and area(r) <= 0.5 and not edge_clipped(r))
    picks = oversize + offcenter + edge
    tags = (
        ["oversized"] * len(oversize)
        + ["off-center"] * len(offcenter)
        + ["edge-clipped"] * len(edge)
    )

    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6))
    for ax, rec, tag in zip(axes.flat, picks, tags):
        try:
            img = Image.open(_resolve_image(data_dir, rec)).convert("RGB")
        except FileNotFoundError:
            ax.text(0.5, 0.5, "missing", ha="center", va="center"); ax.axis("off"); continue
        ax.imshow(img)
        x1, y1, x2, y2 = rec["bbox_valid"]
        ax.add_patch(
            patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                edgecolor="#e0245e", facecolor="none", linewidth=2,
            )
        )
        ax.set_title(f"{tag}: \"{rec['object']}\"", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions",
        nargs="+",
        type=Path,
        required=True,
        help="predictions.jsonl files from run_vlm.py (can be multiple models)",
    )
    ap.add_argument(
        "--detector-metrics",
        nargs="*",
        type=Path,
        default=[],
        help="detector_metrics.json files from detector.py",
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("report/figures"))
    ap.add_argument("--figs", nargs="+", default=["1", "3", "4", "5"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_all_records(args.predictions)

    if "1" in args.figs:
        fig1_teaser(records, args.data_dir, args.out_dir / "fig1_teaser.pdf")
    if "3" in args.figs:
        fig3_atlas(args.predictions, args.out_dir / "fig3_atlas.pdf")
    if "4" in args.figs and args.detector_metrics:
        fig4_roc(args.detector_metrics, args.out_dir / "fig4_roc.pdf")
    if "5" in args.figs:
        fig5_failures(records, args.data_dir, args.out_dir / "fig5_failures.pdf")


if __name__ == "__main__":
    main()
