"""
Download POPE (3 splits) + the COCO val2014 images referenced by
those splits. AMBER is intentionally NOT auto-downloaded here: its
image hosting is unreliable and POPE alone (9000 records across 3
splits) provides more than enough data for PhantomMap.

Two COCO-image modes:
  --coco-zip  (default): grab the official 6GB val2014.zip once and
              extract just the JPEGs referenced by POPE. Robust on
              Colab because it's a single large transfer.
  --coco-per-image: fall back to one-HTTP-request-per-image from
              images.cocodataset.org. Slower and more flaky on Colab,
              but uses less disk if you only want a subset.

Usage:
    python src/download_data.py --out data
"""

from __future__ import annotations
import argparse
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


POPE_SPLITS = {
    "pope_random":      "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json",
    "pope_popular":     "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_popular.json",
    "pope_adversarial": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json",
}

COCO_VAL2014_ZIP = "http://images.cocodataset.org/zips/val2014.zip"
COCO_VAL2014_BASE = "http://images.cocodataset.org/val2014"


def _download(url: str, dest: Path, chunk: int = 1 << 15) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as pbar:
            for data in r.iter_content(chunk):
                f.write(data)
                pbar.update(len(data))
    tmp.replace(dest)


def _coco_image_url(image_id: int) -> str:
    return f"{COCO_VAL2014_BASE}/COCO_val2014_{image_id:012d}.jpg"


def download_pope(out_dir: Path) -> list[dict]:
    merged: list[dict] = []
    for split, url in POPE_SPLITS.items():
        dest = out_dir / "pope" / f"{split}.json"
        _download(url, dest)
        with open(dest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["split"] = split
                merged.append(rec)
    print(f"POPE: {len(merged)} records across {len(POPE_SPLITS)} splits")
    return merged


def _needed_image_names(records: list[dict]) -> set[str]:
    """Collect the set of COCO_val2014_xxx.jpg basenames POPE references."""
    names: set[str] = set()
    for r in records:
        n = str(r.get("image") or "").strip()
        if n.startswith("COCO_val2014_") and n.endswith(".jpg"):
            names.add(n)
    return names


def download_coco_zip(records: list[dict], out_dir: Path, keep_zip: bool = False) -> None:
    """Download val2014.zip once, then extract only the JPEGs referenced
    by POPE into data/coco_val2014/ (~ 3000 files, a few hundred MB)."""
    needed = _needed_image_names(records)
    coco_dir = out_dir / "coco_val2014"
    coco_dir.mkdir(parents=True, exist_ok=True)
    # Short-circuit if everything is already extracted.
    present = {p.name for p in coco_dir.glob("COCO_val2014_*.jpg")}
    missing = needed - present
    if not missing:
        print(f"COCO val2014: already have all {len(needed)} referenced images")
        return
    print(
        f"COCO val2014: {len(missing)} of {len(needed)} referenced images missing; "
        f"downloading val2014.zip (~6 GB) to extract them."
    )
    zip_path = out_dir / "val2014.zip"
    _download(COCO_VAL2014_ZIP, zip_path)
    # Extract matching entries. zipfile is streaming so this doesn't
    # load the whole archive into memory.
    with zipfile.ZipFile(zip_path) as zf:
        # The zip's internal path is "val2014/COCO_val2014_xxx.jpg".
        for info in tqdm(zf.infolist(), desc="extract", unit="file"):
            if not info.filename.endswith(".jpg"):
                continue
            basename = Path(info.filename).name
            if basename not in missing:
                continue
            out_path = coco_dir / basename
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            with zf.open(info) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
    final_present = {p.name for p in coco_dir.glob("COCO_val2014_*.jpg")}
    still_missing = needed - final_present
    print(
        f"COCO val2014: extracted; present={len(final_present)}  "
        f"still-missing={len(still_missing)}"
    )
    if still_missing:
        print(
            "WARNING: these references were not found in val2014.zip "
            f"(first 5): {list(sorted(still_missing))[:5]}",
            file=sys.stderr,
        )


def download_coco_per_image(records: list[dict], out_dir: Path) -> None:
    """Fallback: one request per image. Slower on Colab."""
    needed = _needed_image_names(records)
    coco_dir = out_dir / "coco_val2014"
    coco_dir.mkdir(parents=True, exist_ok=True)
    present = {p.name for p in coco_dir.glob("COCO_val2014_*.jpg")}
    missing = sorted(needed - present)
    print(f"COCO val2014 (per-image): fetching {len(missing)} images")
    for name in tqdm(missing, desc="coco"):
        try:
            img_id = int(name.split("_")[-1].split(".")[0])
        except ValueError:
            continue
        dest = coco_dir / name
        try:
            _download(_coco_image_url(img_id), dest)
        except requests.HTTPError as e:
            print(f"  skip {name}: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument(
        "--coco-mode",
        choices=["zip", "per-image"],
        default="zip",
        help="how to fetch COCO val2014 images (default: bulk zip)",
    )
    ap.add_argument(
        "--keep-zip", action="store_true", help="keep val2014.zip after extraction"
    )
    ap.add_argument(
        "--skip-coco", action="store_true", help="only fetch POPE jsonl, no images"
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pope = download_pope(args.out)

    if args.skip_coco:
        return
    if args.coco_mode == "zip":
        download_coco_zip(pope, args.out, keep_zip=args.keep_zip)
    else:
        download_coco_per_image(pope, args.out)


if __name__ == "__main__":
    main()
