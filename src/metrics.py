"""
Metrics utilities: POPE-style yes/no accuracy, per-split hallucination
rate, and small helpers around sklearn AUROC / PR-AUC.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class POPEStats:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    yes_rate: float
    # "hallucination_rate" is the rate of yes-on-negatives, i.e. the
    # Phantom rate we care about in this project.
    hallucination_rate: float


def pope_stats(records: list[dict]) -> POPEStats:
    """Compute POPE-style stats from a list of records.

    Each record must have:
        label: "yes" or "no" (the ground truth)
        pred : "yes" or "no" (the model's parsed answer)
    """
    if not records:
        return POPEStats(0, 0, 0, 0, 0, 0, 0)
    y_true = np.array([1 if r["label"] == "yes" else 0 for r in records])
    y_pred = np.array([1 if r["pred"] == "yes" else 0 for r in records])
    n = len(records)
    acc = float((y_true == y_pred).mean())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    yes_rate = float(y_pred.mean())
    # Hallucinations: model said yes when label was no.
    n_neg = tn + fp
    hallu_rate = fp / n_neg if n_neg else 0.0
    return POPEStats(n, acc, precision, recall, f1, yes_rate, hallu_rate)


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Safe AUROC wrapper; returns NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(average_precision_score(y_true, y_score))


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray):
    """Return (fpr, tpr, thresholds) for plotting."""
    return roc_curve(y_true, y_score)


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray):
    return precision_recall_curve(y_true, y_score)
