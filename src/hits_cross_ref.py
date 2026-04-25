"""
(Stretch goal, D7) Cross-reference phantom bbox vs EAZY's Hallucinatory
Image Tokens (HITs).

Given a hallucination event on Qwen2.5-VL (pred=yes, gt=no), the VLM
attends to some image tokens when it produces the hallucinated object
word. EAZY (Pseudoc18 et al., 2025) shows that a small number of
tokens drive hallucination. We extract the top-K high-attention image
tokens, map their patch indices back to pixel-space boxes, and compute
IoU with the VLM's self-reported phantom bbox.

Two outcomes are both publishable:
  * High mean IoU: phantom bbox is a downstream "readout" of HITs.
  * Low mean IoU: the VLM's self-reported location decouples from its
    internal attention hotspots.

Note: this module is heavier than the rest of the code because it needs
to (a) re-run the model with output_attentions=True (high memory), and
(b) know how Qwen2.5-VL maps image patches into its token stream. Only
run this after the atlas + detector are complete.
"""

from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from prompts import build_prompt, qwen_messages
from parse_bbox import parse, validate_bbox


QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# EAZY reports that middle layers (roughly 15-22 in Qwen/LLaVA 7B) carry
# the hallucination-inducing attention. Override via CLI if needed.
DEFAULT_LAYERS = tuple(range(15, 23))
TOP_K_HITS = 3


@dataclass
class CrossRef:
    split: str
    image: str
    object: str
    phantom_bbox: list             # [x1, y1, x2, y2] in pixels
    hits_boxes: list               # list of [x1, y1, x2, y2] top-K pixel boxes
    hits_union_box: list           # tight box around union of HITs
    iou_union: float               # IoU(phantom_bbox, hits_union_box)
    mean_iou_topk: float           # mean IoU across individual HITs vs phantom


def iou_xyxy(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _find_object_token_span(token_strs: list[str], obj: str) -> Optional[tuple[int, int]]:
    """Locate the contiguous span (start, end) of generated tokens whose
    joined text starts with the object name. Case-insensitive. Returns
    None if we cannot find it."""
    obj_low = obj.lower().strip()
    joined = ""
    starts = []
    for i, t in enumerate(token_strs):
        starts.append(len(joined))
        joined += t
    joined_low = joined.lower()
    pos = joined_low.find(obj_low)
    if pos < 0:
        return None
    start_tok = None
    end_tok = None
    for i, s in enumerate(starts):
        if start_tok is None and s >= pos:
            start_tok = max(0, i - 1)
        if s >= pos + len(obj_low):
            end_tok = i
            break
    if start_tok is None:
        start_tok = 0
    if end_tok is None:
        end_tok = len(token_strs)
    return start_tok, end_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="predictions.jsonl from run_vlm.py for Qwen only",
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=200, help="subsample for compute reasons")
    ap.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    args = ap.parse_args()

    # Lazy imports so the rest of the project runs without Qwen installed.
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    processor = AutoProcessor.from_pretrained(QWEN_ID)

    # Collect only phantom events that emitted a valid bbox.
    candidates = []
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["pred"] == "yes" and rec["label"] == "no" and rec.get("bbox_valid"):
                candidates.append(rec)
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"hits_cross_ref: {len(candidates)} candidate phantom events")

    results: list[dict] = []

    for rec in tqdm(candidates, desc="hits"):
        img_path = Path(args.data_dir) / (
            f"coco_val2014/{rec['image']}"
            if rec["split"].startswith("pope")
            else f"amber/images/{rec['image']}"
        )
        if not img_path.exists():
            continue
        prompt = build_prompt(rec["object"], with_bbox=True)
        messages = qwen_messages(str(img_path), prompt)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=True,
            )

        prompt_len = inputs.input_ids.shape[-1]
        gen_ids = gen.sequences[0, prompt_len:]
        token_strs = [
            processor.tokenizer.decode([int(t)], skip_special_tokens=True)
            for t in gen_ids.tolist()
        ]
        span = _find_object_token_span(token_strs, rec["object"])
        if span is None:
            continue
        start_tok, end_tok = span

        # Qwen2.5-VL encodes image as a sequence of patch tokens flattened
        # row-major. We need (a) the image-token index range in the full
        # input, and (b) the per-patch pixel region.
        # This mapping is model-specific; we rely on processor's grid_thw.
        grid_thw = inputs["image_grid_thw"][0].tolist()  # (T, H, W)
        t_patches, h_patches, w_patches = grid_thw
        # The attention tensor has shape:
        #     attentions[step][layer] -> (batch, heads, seq, seq)
        # For generated token i, the query is the last token; keys span the
        # full (prompt + previously-generated) sequence. We want keys that
        # correspond to image patches.
        # Qwen places image tokens contiguously inside the prompt; find the
        # index by searching for the image-pad token.
        image_token_id = getattr(model.config, "image_token_id", None) or processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        prompt_ids = inputs.input_ids[0].tolist()
        image_positions = [i for i, t in enumerate(prompt_ids) if t == image_token_id]
        if not image_positions:
            continue
        img_start = image_positions[0]
        img_end = image_positions[-1] + 1
        n_image_tokens = img_end - img_start
        # A single frame: expect t_patches * h_patches * w_patches == n_image_tokens.
        if t_patches * h_patches * w_patches != n_image_tokens:
            # Qwen uses spatial-merge of 2x2; account for it.
            merge = int(math.sqrt((t_patches * h_patches * w_patches) / max(1, n_image_tokens)))
            h_eff = h_patches // max(1, merge)
            w_eff = w_patches // max(1, merge)
        else:
            h_eff = h_patches
            w_eff = w_patches
        if h_eff * w_eff != n_image_tokens:
            # Fallback: skip this sample.
            continue

        # Sum attention from object-word query tokens to image-key tokens
        # across chosen layers and heads.
        attn_img = np.zeros(n_image_tokens, dtype=np.float32)
        n_steps_used = 0
        for step_idx in range(start_tok, end_tok):
            if step_idx >= len(gen.attentions):
                break
            layer_attn = gen.attentions[step_idx]
            for L in args.layers:
                if L >= len(layer_attn):
                    continue
                a = layer_attn[L]  # (batch, heads, 1, seq)
                # (1, heads, 1, seq) -> average heads -> (seq,)
                a_mean = a[0].mean(0).squeeze(0)
                a_mean = a_mean.to(torch.float32).cpu().numpy()
                attn_img += a_mean[img_start:img_end]
                n_steps_used += 1
        if n_steps_used == 0:
            continue
        attn_img /= n_steps_used

        # Top-K patch indices -> pixel boxes.
        top_idx = np.argsort(attn_img)[-TOP_K_HITS:][::-1]
        W, H = rec["img_w"], rec["img_h"]
        patch_w = W / w_eff
        patch_h = H / h_eff
        hits_boxes = []
        for k in top_idx:
            ry, rx = int(k // w_eff), int(k % w_eff)
            x1 = rx * patch_w; y1 = ry * patch_h
            x2 = x1 + patch_w; y2 = y1 + patch_h
            hits_boxes.append([float(x1), float(y1), float(x2), float(y2)])
        # Union box.
        xs1 = [b[0] for b in hits_boxes]; ys1 = [b[1] for b in hits_boxes]
        xs2 = [b[2] for b in hits_boxes]; ys2 = [b[3] for b in hits_boxes]
        union = (min(xs1), min(ys1), max(xs2), max(ys2))
        phantom = tuple(rec["bbox_valid"])
        iou_u = iou_xyxy(phantom, union)
        iou_topk = np.mean([iou_xyxy(phantom, tuple(b)) for b in hits_boxes])

        results.append(
            {
                "split": rec["split"],
                "image": rec["image"],
                "object": rec["object"],
                "phantom_bbox": rec["bbox_valid"],
                "hits_boxes": hits_boxes,
                "hits_union_box": list(union),
                "iou_union": float(iou_u),
                "mean_iou_topk": float(iou_topk),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    if results:
        ious_u = np.array([r["iou_union"] for r in results])
        print(
            f"mean IoU(phantom, HITs-union) = {ious_u.mean():.3f} ± {ious_u.std():.3f}"
            f"   median = {np.median(ious_u):.3f}"
            f"   (n={len(results)})"
        )


if __name__ == "__main__":
    main()
