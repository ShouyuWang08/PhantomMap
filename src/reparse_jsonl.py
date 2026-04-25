"""
Re-parse an existing run_vlm.py jsonl in-place, applying the current
parse_bbox.py logic to the saved `raw` text. This lets us fix parser
bugs retroactively without re-running the VLMs.

Usage:
    python src/reparse_jsonl.py results/*.jsonl
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_bbox import parse, validate_bbox


def reparse_file(path: Path) -> dict:
    recs_in = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs_in.append(json.loads(line))
    n_recovered_bbox = 0
    n_changed_pred = 0
    recs_out = []
    for r in recs_in:
        parsed = parse(r["raw"])
        new_pred = parsed.answer
        new_bbox = (
            validate_bbox(parsed.bbox, r["img_w"], r["img_h"])
            if parsed.bbox is not None
            else None
        )
        if (r.get("bbox_valid") is None) and (new_bbox is not None):
            n_recovered_bbox += 1
        if r.get("pred") != new_pred:
            n_changed_pred += 1
        r = dict(r)  # copy
        r["pred"] = new_pred
        r["bbox"] = list(parsed.bbox) if parsed.bbox else None
        r["bbox_valid"] = list(new_bbox) if new_bbox else None
        recs_out.append(r)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs_out:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    return {
        "path": str(path),
        "n": len(recs_in),
        "bbox_recovered": n_recovered_bbox,
        "pred_changed": n_changed_pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()
    for p in args.paths:
        r = reparse_file(p)
        print(
            f"{r['path']}: n={r['n']}  bbox_recovered={r['bbox_recovered']}  "
            f"pred_changed={r['pred_changed']}"
        )


if __name__ == "__main__":
    main()
