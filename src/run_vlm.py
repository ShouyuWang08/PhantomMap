"""
Inference driver for PhantomMap.

Given a POPE or AMBER split, run a VLM against every record, capture:
  * the parsed yes/no answer
  * the emitted bbox (if any)
  * per-token logprobs for the 4 bbox coordinate tokens
  * the raw generation

Writes JSONL, one line per record, idempotently (resumes from where the
output file left off).

Works on a free Colab T4 in bf16 for Qwen2.5-VL-7B and LLaVA-NeXT-7B.
Expected runtime: ~80 minutes per split on T4 at batch_size=1.

Usage:
    python src/run_vlm.py --model qwen --split pope_adversarial \
        --data-dir data --out results/qwen_pope_adv.jsonl
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image
from tqdm import tqdm

# Local imports.
sys.path.insert(0, str(Path(__file__).parent))
from prompts import build_prompt, qwen_messages
from parse_bbox import parse, validate_bbox


@dataclass
class Record:
    split: str
    image: str
    object: str
    label: str  # "yes" or "no" ground truth
    pred: str   # "yes" / "no" / "unknown"
    raw: str
    bbox: Optional[list]           # [x1,y1,x2,y2] in pixels, or None
    bbox_valid: Optional[list]     # clipped/validated, or None
    img_w: int
    img_h: int
    logp_mean: float               # mean logprob of the 4 coord tokens (nan if no bbox)
    logp_min: float                # min logprob among the 4 coord tokens (nan if no bbox)
    logp_yes: float                # logprob of the yes/no first token
    model_id: str


QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LLAVA_ID = "llava-hf/llava-v1.6-mistral-7b-hf"


# ------------------------------------------------------------------ IO


def load_records(data_dir: Path, split: str) -> list[dict]:
    """Load POPE or AMBER records for a single split."""
    if split.startswith("pope"):
        path = data_dir / "pope" / f"{split}.json"
        recs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        # Normalise keys: POPE fields are image, text, label, object_name.
        norm = []
        for r in recs:
            q = r.get("text") or r.get("question") or ""
            obj = r.get("object_name") or _extract_object_from_question(q) or ""
            norm.append(
                {
                    "image": r["image"],
                    "object": obj,
                    "label": r.get("label", "").lower(),
                    "split": split,
                }
            )
        return norm
    elif split == "amber":
        path = data_dir / "amber" / "query_discriminative.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        norm = []
        for r in data:
            norm.append(
                {
                    "image": r["image"],
                    "object": r["query"].replace("Is there a ", "").rstrip("?").strip(),
                    "label": "yes" if r["truth"] == "Yes" else "no",
                    "split": "amber",
                }
            )
        return norm
    else:
        raise ValueError(f"unknown split: {split}")


def _extract_object_from_question(q: str) -> Optional[str]:
    """POPE questions look like 'Is there a cat in the image?'. Extract 'cat'."""
    q = q.strip().rstrip("?").lower()
    for prefix in ("is there a ", "is there an ", "are there any "):
        if q.startswith(prefix):
            rest = q[len(prefix):]
            if rest.endswith(" in the image"):
                rest = rest[: -len(" in the image")]
            return rest.strip()
    return None


def resolve_image_path(data_dir: Path, name: str, split: str) -> Path:
    if split.startswith("pope"):
        return data_dir / "coco_val2014" / name
    elif split == "amber":
        return data_dir / "amber" / "images" / name
    raise ValueError(split)


def already_done(out_path: Path) -> set[str]:
    """Load primary keys from an existing output file so we can resume."""
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(f"{r['split']}::{r['image']}::{r['object']}")
    return done


# ------------------------------------------------------------------ Qwen


def load_qwen(device: str = "cuda", dtype=torch.bfloat16):
    """Lazy import so users without transformers installed can still read this file."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(QWEN_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID, torch_dtype=dtype, device_map=device
    )
    model.eval()
    return model, processor


def run_qwen_one(model, processor, image_path: Path, prompt: str, max_new_tokens: int = 96):
    """Single-sample Qwen2.5-VL forward. Returns (text, logprobs_list)."""
    from qwen_vl_utils import process_vision_info

    messages = qwen_messages(str(image_path), prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    prompt_len = inputs.input_ids.shape[-1]
    gen_ids = gen.sequences[0, prompt_len:]
    out_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Per-step logprob for the greedy-chosen token.
    logprobs = []
    for step_idx, score in enumerate(gen.scores):
        if step_idx >= len(gen_ids):
            break
        logp = torch.log_softmax(score[0].float(), dim=-1)
        tok_id = int(gen_ids[step_idx].item())
        logprobs.append(float(logp[tok_id].item()))

    # Also decode per-token so we can attribute logprobs to bbox digits.
    token_strs = [
        processor.tokenizer.decode([int(t)], skip_special_tokens=True)
        for t in gen_ids.tolist()
    ]
    return out_text, logprobs, token_strs


# ------------------------------------------------------------------ LLaVA


def load_llava(device: str = "cuda", dtype=torch.bfloat16):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained(LLAVA_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_ID, torch_dtype=dtype, device_map=device
    )
    model.eval()
    return model, processor


def run_llava_one(model, processor, image_path: Path, prompt: str, max_new_tokens: int = 96):
    img = Image.open(image_path).convert("RGB")
    conv = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(images=img, text=text, return_tensors="pt").to(
        model.device, model.dtype
    )
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    prompt_len = inputs.input_ids.shape[-1]
    gen_ids = gen.sequences[0, prompt_len:]
    out_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    logprobs = []
    for step_idx, score in enumerate(gen.scores):
        if step_idx >= len(gen_ids):
            break
        logp = torch.log_softmax(score[0].float(), dim=-1)
        tok_id = int(gen_ids[step_idx].item())
        logprobs.append(float(logp[tok_id].item()))
    token_strs = [
        processor.tokenizer.decode([int(t)], skip_special_tokens=True)
        for t in gen_ids.tolist()
    ]
    return out_text, logprobs, token_strs


# ------------------------------------------------------------------ Core loop


def _first_yes_no_logprob(tokens: list[str], logprobs: list[float]) -> float:
    """Return the logprob of the first explicit 'yes' or 'no' token.

    If the model skipped yes/no and jumped straight to a bbox (common for
    Qwen2.5-VL grounded prompts), fall back to the logprob of the first
    content-carrying token — we skip pure whitespace and markdown fences
    like ``` so the fallback captures the model's confidence in the
    actual assertion, not the opening of a code fence.
    """
    for t, lp in zip(tokens, logprobs):
        low = t.strip().lower()
        if low in ("yes", "no"):
            return lp
    for t, lp in zip(tokens, logprobs):
        stripped = t.strip()
        if not stripped:
            continue
        if stripped.startswith("```") or stripped == "json":
            continue
        return lp
    return float("nan")


def _bbox_coord_logprobs(tokens: list[str], logprobs: list[float]) -> list[float]:
    """Find the 4 integer tokens inside the first [...] span."""
    in_bbox = False
    picked: list[float] = []
    for t, lp in zip(tokens, logprobs):
        if "[" in t:
            in_bbox = True
            continue
        if "]" in t:
            break
        if in_bbox and any(c.isdigit() for c in t):
            picked.append(lp)
            if len(picked) == 4:
                break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "llava"], required=True)
    ap.add_argument(
        "--split",
        choices=["pope_random", "pope_popular", "pope_adversarial", "amber"],
        required=True,
    )
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="cap for debugging")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = load_records(args.data_dir, args.split)
    if args.limit:
        records = records[: args.limit]
    done = already_done(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "qwen":
        model, processor = load_qwen(device, dtype)
        run_one = lambda ip, pr: run_qwen_one(model, processor, ip, pr)
        model_id = QWEN_ID
    else:
        model, processor = load_llava(device, dtype)
        run_one = lambda ip, pr: run_llava_one(model, processor, ip, pr)
        model_id = LLAVA_ID

    missing_images = 0
    processed = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for r in tqdm(records, desc=f"{args.model}-{args.split}"):
            key = f"{r['split']}::{r['image']}::{r['object']}"
            if key in done:
                continue
            img_path = resolve_image_path(args.data_dir, r["image"], r["split"])
            if not img_path.exists():
                missing_images += 1
                if missing_images <= 3:
                    print(f"[skip] missing image: {img_path}", file=sys.stderr)
                continue
            processed += 1
            try:
                img_w, img_h = Image.open(img_path).size
                prompt = build_prompt(r["object"], with_bbox=True)
                text, logprobs, token_strs = run_one(img_path, prompt)
            except Exception as e:
                print(f"\n[err] {key}: {e}", file=sys.stderr)
                continue

            parsed = parse(text)
            valid = (
                validate_bbox(parsed.bbox, img_w, img_h)
                if parsed.bbox is not None
                else None
            )
            coord_lps = _bbox_coord_logprobs(token_strs, logprobs) if valid else []
            lp_mean = float(sum(coord_lps) / len(coord_lps)) if coord_lps else float("nan")
            lp_min = float(min(coord_lps)) if coord_lps else float("nan")

            rec = Record(
                split=r["split"],
                image=r["image"],
                object=r["object"],
                label=r["label"],
                pred=parsed.answer,
                raw=text,
                bbox=list(parsed.bbox) if parsed.bbox else None,
                bbox_valid=list(valid) if valid else None,
                img_w=img_w,
                img_h=img_h,
                logp_mean=lp_mean,
                logp_min=lp_min,
                logp_yes=_first_yes_no_logprob(token_strs, logprobs),
                model_id=model_id,
            )
            fout.write(json.dumps(asdict(rec)) + "\n")
            fout.flush()

    # Be loud at the end: if the vast majority of records were skipped
    # due to missing images, the downstream pipeline will silently
    # produce empty results, so we raise here.
    if processed == 0 and missing_images > 0:
        raise SystemExit(
            f"ERROR: all {missing_images} candidate records were skipped because "
            f"images were not found under {args.data_dir}. "
            f"Re-run: python src/download_data.py --out {args.data_dir}"
        )
    print(f"[done] processed={processed}  skipped(missing_image)={missing_images}")


if __name__ == "__main__":
    main()
