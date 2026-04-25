"""
Feature extraction from run_vlm.py output records.

Each record yields a fixed-length feature vector if the model emitted a
valid bbox, else None. The label (phantom vs. real) comes from the POPE
/ AMBER ground truth: phantom = (pred=yes, label=no).
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "cx",
    "cy",
    "w",
    "h",
    "area",
    "d_center",
    "aspect_ratio",
    "logp_mean",
    "logp_min",
    "logp_yes",
]


@dataclass
class Sample:
    feats: np.ndarray  # shape [len(FEATURE_NAMES)]
    label: int         # 1 = phantom (pred yes, gt no); 0 = honest (pred yes, gt yes)
    split: str
    image: str
    object: str
    model_id: str


def featurize_one(rec: dict) -> np.ndarray | None:
    """Return feature vector for a record that emitted a valid bbox, else None."""
    bbox = rec.get("bbox_valid")
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    W = float(rec["img_w"])
    H = float(rec["img_h"])
    # Normalise to [0,1].
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    area = w * h
    d_center = math.hypot(cx - 0.5, cy - 0.5)
    aspect = (x2 - x1) / max(1.0, (y2 - y1))
    lp_mean = _safe(rec.get("logp_mean"))
    lp_min = _safe(rec.get("logp_min"))
    lp_yes = _safe(rec.get("logp_yes"))
    return np.array(
        [cx, cy, w, h, area, d_center, aspect, lp_mean, lp_min, lp_yes], dtype=np.float32
    )


def _safe(x) -> float:
    """NaN → 0.0 as a simple imputation. Logistic regression with standardisation
    treats this as 'missing' = population mean, which is what we want."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return x


def load_samples(jsonl_paths: list[Path], phantom_only: bool = False) -> list[Sample]:
    """Load records from one or more jsonl files and return the features for
    every record where the model said YES and emitted a valid bbox.

    When phantom_only=True, drop records that are honest (pred=yes, gt=yes) —
    used when we just want the atlas of phantom boxes. For the detector we
    want both classes.
    """
    samples: list[Sample] = []
    for p in jsonl_paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("pred") != "yes":
                    continue
                feats = featurize_one(rec)
                if feats is None:
                    continue
                label = 1 if rec["label"] == "no" else 0
                if phantom_only and label != 1:
                    continue
                samples.append(
                    Sample(
                        feats=feats,
                        label=label,
                        split=rec["split"],
                        image=rec["image"],
                        object=rec["object"],
                        model_id=rec["model_id"],
                    )
                )
    return samples


def stack(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([s.feats for s in samples], axis=0)
    y = np.array([s.label for s in samples], dtype=np.int64)
    return X, y
