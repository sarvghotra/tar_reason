"""Teacher-VLM reflections for ``--objective teacher_tree``.

The frozen Qwen3-VL judge process also writes one reflection per rendered image
(the "teacher"). It sees only the prompt and the image. Its reply is parsed
into the policy's own reflection format, measured on sampled SFT/RL policy
reflections: a leading space, ``<critique>\\nCorrection: <instruction>\\n`` and
then ``<im_start>`` for a refinement, or exactly ``LOOKS_GOOD_TEXT`` and then
EOS. The verdict uses the evaluation parser (``looks_good_verdict``).
"""

import re
from typing import Optional, Tuple

from llava.train.rl.rollout import CORRECTION_RE, looks_good_verdict

TEACHER_PROMPT_VERSION = "teacher_reflect_v1"
LOOKS_GOOD_TEXT = " no issues, it matches the prompt.\nCorrection: looks good."
TEACHER_INSTRUCTION = (
    "You are checking an image that was generated for this text prompt:\n"
    "\"{prompt}\"\n\n"
    "Compare the image with every part of the prompt: the objects, their counts, colors, "
    "positions and other attributes, and any requested text.\n\n"
    "If the image fully matches the prompt, reply with exactly these two lines:\n"
    "no issues, it matches the prompt.\n"
    "Correction: looks good.\n\n"
    "Otherwise reply with exactly two lines. The first line is one short sentence that describes "
    "the most important mismatch. The second line starts with \"Correction:\" followed by one short "
    "instruction that fixes it. For example:\n"
    "there are seven giraffes instead of six\n"
    "Correction: remove one giraffe from the right side of the image.\n"
    "or:\n"
    "the cup is blue but the prompt asks for a red cup\n"
    "Correction: change the color of the cup to red.\n\n"
    "Reply with the two lines only."
)
_SPECIAL = re.compile(r"<[^<>\s]*>")  # <im_start>, <I12>, <|im_end|>, ...
_SPACE = re.compile(r"\s+")
_LEAD = re.compile(r"^\s*(?:self[- ]reflect(?:ion)?\s*:\s*)", re.IGNORECASE)


def teacher_instruction(prompt: str) -> str:
    return TEACHER_INSTRUCTION.format(prompt=prompt.strip())


def parse_teacher_text(raw: Optional[str]) -> Optional[Tuple[str, bool]]:
    """Teacher reply -> (policy reflection text, verdict), or None if malformed.

    A "looks good" verdict always becomes the canonical ``LOOKS_GOOD_TEXT``.
    Otherwise the reply needs exactly one nonempty critique before a single
    "Correction:" clause with a nonempty instruction; each part is collapsed
    to one line. Text that would encode as special tokens is rejected.
    """
    if not raw or not raw.strip() or _SPECIAL.search(raw):
        return None
    text = _LEAD.sub("", raw.strip())
    if looks_good_verdict(text, "correction")[0]:
        return LOOKS_GOOD_TEXT, True
    clauses = list(CORRECTION_RE.finditer(text))
    if len(clauses) != 1:
        return None
    critique = _SPACE.sub(" ", text[:clauses[0].start()]).strip()
    instruction = _SPACE.sub(" ", text[clauses[0].end():]).strip()
    if not critique or not instruction:
        return None
    reflection = f" {critique}\nCorrection: {instruction}\n"
    if looks_good_verdict(reflection, "correction") != (False, True):
        return None
    return reflection, False
