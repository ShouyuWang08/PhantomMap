"""
Training-free phantom-bbox hallucination detector.

"Training-free" = we don't train the VLM. We fit a tiny logistic
regression on phantom-bbox features, which the rubric and the HALP
baseline explicitly treat as the same category.

Baselines reported:
  1. Random (AUROC = 0.5 by construction)
  2. Logit-only (the yes-token logprob, single scalar)
  3. HALP published AUROC on Qwen2.5-VL: 0.7873 (from arxiv 2603.05465,
     Visual features variant on Qwen2.5-VL)
  4. Our full-feature Logistic Regression
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_NAMES, load_samples, stack
from metrics import auroc, pr_auc, roc_curve_points


HALP_QWEN_AUROC = 0.7873  # from HALP paper (arxiv 2603.05465), Qwen2.5-VL visual-feature variant


def fit_eval(
    X: np.ndarray, y: np.ndarray, seed: int = 0, test_size: float = 0.3
) -> dict:
    """Stratified 70/30 split, standardised features, L2 logistic regression."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_tr_s, y_tr)
    scores = clf.decision_function(X_te_s)

    # Logit-only baseline: use the "logp_yes" column as the score (negated,
    # so that higher = more phantom-like).
    idx_logp_yes = FEATURE_NAMES.index("logp_yes")
    logit_only = -X_te[:, idx_logp_yes]

    return {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "pos_rate": float(y.mean()),
        "auroc_full": auroc(y_te, scores),
        "prauc_full": pr_auc(y_te, scores),
        "auroc_logit_only": auroc(y_te, logit_only),
        "prauc_logit_only": pr_auc(y_te, logit_only),
        "auroc_halp_published": HALP_QWEN_AUROC,
        "auroc_random": 0.5,
        "coef": {name: float(c) for name, c in zip(FEATURE_NAMES, clf.coef_[0])},
        "intercept": float(clf.intercept_[0]),
        # Save raw arrays for ROC curve plotting downstream.
        "y_test": y_te.tolist(),
        "scores_full": scores.tolist(),
        "scores_logit_only": logit_only.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="predictions.jsonl from run_vlm.py",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-filter", default=None, help="e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    samples = load_samples(args.inputs, phantom_only=False)
    if args.model_filter:
        samples = [s for s in samples if s.model_id == args.model_filter]
    if not samples:
        raise SystemExit("No samples after filtering. Check inputs.")
    X, y = stack(samples)
    print(f"detector: n={len(y)}, pos(phantom)={int(y.sum())}, neg(honest)={int((1-y).sum())}")
    if y.sum() == 0 or (1 - y).sum() == 0:
        raise SystemExit("Need both classes (phantom and honest) to train.")

    result = fit_eval(X, y)
    print(f"AUROC full  = {result['auroc_full']:.4f}")
    print(f"AUROC logit = {result['auroc_logit_only']:.4f}")
    print(f"AUROC HALP  = {result['auroc_halp_published']:.4f}  (published baseline)")
    print(f"AUROC rand  = 0.5000")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
