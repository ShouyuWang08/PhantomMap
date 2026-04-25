"""
Simple system-diagram generator for Fig 2 of the report.

We produce a matplotlib-based PDF so that the report builds out of the
box without requiring a PowerPoint / Inkscape round-trip. The team is
encouraged to replace fig2_method.pdf with a hand-crafted diagram
before submission, but the content below is already readable and
print-quality.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def draw_box(ax, x, y, w, h, text, color="#e8eefc", edge="#2a4d9b"):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=color,
            edgecolor=edge,
            linewidth=1.1,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.0),
    )


def build(out: Path):
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_box(ax, 0.1, 1.2, 1.6, 1.0, "Image +\ncandidate\nobject $o$", "#fde6d9", "#c44d19")
    draw_box(ax, 2.1, 1.2, 1.8, 1.0,
             'Elicitation\nprompt\n(\\S 3.1)', "#e8eefc", "#2a4d9b")
    draw_box(ax, 4.3, 1.2, 1.8, 1.0, "VLM\n(Qwen2.5-VL /\nLLaVA-NeXT)", "#e6f3e6", "#2c6f2c")
    draw_box(ax, 6.5, 2.15, 1.5, 0.6, 'answer "yes"/"no"', "#fff8c4", "#9c8a14")
    draw_box(ax, 6.5, 1.5, 1.5, 0.5, 'bbox (if yes)', "#fff8c4", "#9c8a14")
    draw_box(ax, 6.5, 0.85, 1.5, 0.5, 'coord logprobs', "#fff8c4", "#9c8a14")
    draw_box(ax, 8.4, 2.0, 1.5, 0.6, "Spatial atlas\n(KDE heatmap)", "#f2e0f7", "#7a2d8c")
    draw_box(ax, 8.4, 1.1, 1.5, 0.6, "Features +\ndetector (LR)", "#f2e0f7", "#7a2d8c")

    arrow(ax, 1.75, 1.7, 2.05, 1.7)
    arrow(ax, 3.95, 1.7, 4.25, 1.7)
    arrow(ax, 6.15, 1.9, 6.45, 2.45)
    arrow(ax, 6.15, 1.7, 6.45, 1.75)
    arrow(ax, 6.15, 1.5, 6.45, 1.1)
    arrow(ax, 8.05, 1.9, 8.35, 2.3)
    arrow(ax, 8.05, 1.5, 8.35, 1.4)
    arrow(ax, 8.05, 1.1, 8.35, 1.35)

    ax.text(5.0, 2.9, "PhantomMap pipeline", ha="center", fontsize=9, weight="bold")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("report/figures/fig2_method.pdf"))
    args = ap.parse_args()
    build(args.out)
