"""
Tolerant parsing of VLM outputs into (answer, bbox, raw) tuples.

The elicitation prompt in prompts.py asks for:
    yes|no\n{"bbox_2d": [x1, y1, x2, y2]}
but real models emit noisier strings: trailing prose, markdown fences,
single quotes, slight key-name variants. This module absorbs that noise
while refusing anything that cannot be made into 4 integers in
image-pixel range.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Optional


# Any JSON-like object that looks like a bbox. We capture non-greedy
# to tolerate multiple objects in the same string.
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

# Fallback: 4 comma-separated numbers in square brackets.
_RAW_BBOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    r"\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


@dataclass
class ParsedOutput:
    """Result of parsing one VLM generation.

    answer: "yes" / "no" / "unknown"
    bbox:   (x1, y1, x2, y2) floats in image-pixel coordinates, or None
    raw:    the original string (kept for error analysis)
    """

    answer: str
    bbox: Optional[tuple[float, float, float, float]]
    raw: str


def _coerce_answer(text: str) -> str:
    """The model's first yes/no wins; default to 'unknown' if neither appears."""
    y = _YES_RE.search(text)
    n = _NO_RE.search(text)
    if y and n:
        return "yes" if y.start() < n.start() else "no"
    if y:
        return "yes"
    if n:
        return "no"
    return "unknown"


_BBOX_KEYS = ("bbox_2d", "bbox", "box", "bounding_box")


def _extract_bbox_from_obj(obj) -> Optional[tuple[float, float, float, float]]:
    """Pull a bbox out of either a dict or a list-of-dicts."""
    if isinstance(obj, dict):
        for key in _BBOX_KEYS:
            if key in obj and _is_bbox_list(obj[key]):
                return tuple(float(x) for x in obj[key])  # type: ignore[return-value]
    if isinstance(obj, list):
        for item in obj:
            b = _extract_bbox_from_obj(item)
            if b is not None:
                return b
    return None


def _try_json_bbox(s: str) -> Optional[tuple[float, float, float, float]]:
    """Search the string for any JSON object or JSON list whose elements
    contain a bbox. Qwen2.5-VL sometimes emits:
        {"bbox_2d": [x1,y1,x2,y2]}
        [{"bbox_2d": [x1,y1,x2,y2], "label": "cat"}]
        [{"bbox_2d": [x1,y1,x2,y2]}, {"bbox_2d": [...]}, ...]
    All of these must resolve to a single bbox (we take the first)."""

    # Strip markdown fences like ```json ... ``` first.
    stripped = re.sub(r"```(?:json)?\s*", "", s)
    stripped = stripped.replace("```", "")

    # Try to parse the entire stripped string (after trimming) as JSON.
    # Cheap win for well-formed outputs.
    for candidate in (stripped.strip(), s.strip()):
        try:
            obj = json.loads(candidate)
            b = _extract_bbox_from_obj(obj)
            if b is not None:
                return b
        except Exception:
            pass

    # Fallback: greedy pass over substrings that look like JSON lists
    # (`[...]`) or JSON dicts (`{...}`). We try lists first because when
    # both are present the list is usually the outer container.
    list_candidates = re.findall(r"\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\]", s)
    dict_candidates = re.findall(r"\{[^{}]*\}", s)
    for candidates in (list_candidates, dict_candidates):
        for chunk in candidates:
            chunk2 = chunk.replace("'", '"')
            try:
                obj = json.loads(chunk2)
            except Exception:
                continue
            b = _extract_bbox_from_obj(obj)
            if b is not None:
                return b
    return None


def _try_raw_bbox(s: str) -> Optional[tuple[float, float, float, float]]:
    m = _RAW_BBOX_RE.search(s)
    if not m:
        return None
    return tuple(float(g) for g in m.groups())  # type: ignore[return-value]


def _is_bbox_list(v) -> bool:
    if not isinstance(v, (list, tuple)):
        return False
    if len(v) != 4:
        return False
    try:
        [float(x) for x in v]
        return True
    except Exception:
        return False


def parse(text: str) -> ParsedOutput:
    """Main entry point: accept a raw VLM string, return structured output.

    If the model emits a bounding box without an explicit "yes"/"no",
    treat the bbox itself as an implicit "yes" — it's a first-party
    assertion that the object is present in the image. In practice
    Qwen2.5-VL frequently behaves this way on grounded prompts.
    """
    answer = _coerce_answer(text)
    bbox = _try_json_bbox(text) or _try_raw_bbox(text)
    if answer == "unknown" and bbox is not None:
        answer = "yes"
    return ParsedOutput(answer=answer, bbox=bbox, raw=text)


def validate_bbox(
    bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    min_side: int = 4,
) -> Optional[tuple[float, float, float, float]]:
    """Return the bbox clipped to image bounds if it is sane, else None.

    A bbox is sane when: 0 <= x1 < x2 <= W, 0 <= y1 < y2 <= H, and both
    sides are at least min_side pixels.

    Heuristic: if all four coordinates are in [0, 1.01], we assume the
    model emitted them in normalised [0,1] coordinates (LLaVA-NeXT
    behaves this way on grounded prompts) and scale them to pixels
    before clipping. Qwen2.5-VL always uses pixel coordinates, so
    this branch is a no-op for it.
    """
    x1, y1, x2, y2 = bbox
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_coord <= 1.01:
        x1 = x1 * img_w
        y1 = y1 * img_h
        x2 = x2 * img_w
        y2 = y2 * img_h
    # Some models emit (y1, x1, y2, x2). We detect and correct only the
    # obvious case where x > W but y <= H and swapping makes it sane.
    if x1 > img_w and y1 <= img_h and x2 > img_w and y2 <= img_h:
        x1, y1, x2, y2 = y1, x1, y2, x2
    # Clip.
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    # Enforce ordering.
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if (x2 - x1) < min_side or (y2 - y1) < min_side:
        return None
    return (x1, y1, x2, y2)


if __name__ == "__main__":
    # Smoke test.
    cases = [
        'yes\n{"bbox_2d": [10, 20, 100, 200]}',
        'YES.\n```json\n{"bbox_2d": [10, 20, 100, 200]}\n```',
        "no",
        "yes {'bbox': [1,2,3,4]}",
        'yes\n{"box": [5, 6, 50, 60]}',
        "garbage output",
    ]
    for c in cases:
        out = parse(c)
        print(f"IN: {c!r}\n -> answer={out.answer} bbox={out.bbox}\n")
