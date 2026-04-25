"""
Prompt templates for phantom-bbox elicitation.

The PhantomMap method rests on a single prompt design choice:
ask the VLM a yes/no existence question *and* require a bounding box
when the answer is "yes". Over POPE/AMBER negatives, every "yes"
is by construction a hallucination event, and the emitted bbox is
the VLM's first-party self-report of where it thinks the phantom is.

Keeping prompts in this module (rather than inlined) is intentional:
reviewers often ask for the verbatim prompt, and a single source of
truth prevents silent drift between runs.
"""

from __future__ import annotations
from typing import Optional


# Primary elicitation prompt used for every POPE / AMBER probe.
# Keep as a single f-string with one variable {object}. Do not add
# examples or chain-of-thought here: we want the raw, first-token
# behaviour of the VLM.
ELICIT_PROMPT = (
    "Look at the image. Is there a {object} in the image? "
    'Answer "yes" or "no". '
    "If yes, also output the bounding box of the {object} in JSON on a new line, "
    'exactly in the format: {{"bbox_2d": [x1, y1, x2, y2]}}, '
    "where x1,y1 is the top-left and x2,y2 is the bottom-right, in pixels. "
    "If no, output nothing else."
)


# Ablation prompt: no-bbox variant. Used to verify that the *addition*
# of bbox-output instructions does not change the yes/no rate
# (i.e. asking for a bbox does not itself induce more hallucinations).
ELICIT_PROMPT_NOBBOX = (
    "Look at the image. Is there a {object} in the image? "
    'Answer "yes" or "no".'
)


def build_prompt(obj: str, with_bbox: bool = True) -> str:
    """Fill the object name into the elicitation template.

    obj must be a bare noun phrase without articles (POPE queries use this form).
    """
    template = ELICIT_PROMPT if with_bbox else ELICIT_PROMPT_NOBBOX
    return template.format(object=obj.strip())


# Model-specific chat-template wrappers. Each returns the list of
# `messages` that transformers' apply_chat_template expects.
def qwen_messages(image_path: str, prompt: str) -> list[dict]:
    """Qwen2.5-VL messages. `image` key passes a local path that
    qwen_vl_utils.process_vision_info will resolve."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def llava_messages(image_path: str, prompt: str) -> list[dict]:
    """LLaVA-NeXT messages. The processor takes raw PIL elsewhere;
    this helper only formats the chat structure."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


if __name__ == "__main__":
    # Quick sanity print.
    for obj in ["cat", "bicycle", "stop sign"]:
        print(build_prompt(obj))
        print("---")
