"""GenEval2-style soft-TIFA reward computed *in latent space* by a frozen Tar
model answering the benchmark's VQA questions directly on discrete image
tokens (no pixel decoding).

Prompt layout follows Tar's understanding-mode training data
(``llava/model/llava_arch.py::prepare_inputs_labels_for_multimodal``): the user
turn is ``<S0>`` + the image tokens (as LM-vocab ids) + ``\\n`` + question.

Scoring mirrors ``GenEval2/evaluation_optz.py``: for every question the score
is the summed next-token probability of the accepted answer variants
(``Yes``/``yes``/`` Yes``/`` yes`` for yes-no questions; the number word,
its capitalised and space-prefixed forms and the digit for "How many"
questions). AM = mean over questions, GM = geometric mean.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import torch

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

SYSTEM_PROMPT = "You are a helpful assistant."
ANSWER_SUFFIXES = {
    "llava": "Answer the question using a single word or phrase.",
    "geneval2": "Answer in one word.",
    "none": "",
}


@dataclass
class RewardConfig:
    batch_size: int = 16
    answer_suffix: str = "llava"
    scale: int = 0
    alpha: float = 1.0          # reward = alpha * AM + (1 - alpha) * GM


def answer_variants(question: str, answer: str) -> List[str]:
    if question.startswith("How many"):
        digit = NUMBER_WORDS.get(answer.lower(), "other")
        return [answer, answer.capitalize(), " " + answer,
                " " + answer.capitalize(), digit, " " + digit]
    return ["Yes", "yes", " yes", " Yes"]


def geometric_mean(values: Sequence[float]) -> float:
    if any(v <= 0 for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


class TarLatentVQAReward:
    """Scores (image_codes, vqa_list) pairs with a frozen Tar LM.

    ``forward_hidden(input_ids, attention_mask) -> last_hidden_state`` and
    ``lm_head`` are supplied by the caller so the reward can share the policy's
    frozen base weights (LoRA adapters disabled) instead of loading a second
    7B model.
    """

    def __init__(self, tokenizer, forward_hidden: Callable, lm_head, device,
                 cfg: RewardConfig, image_start_id: int):
        self.tok = tokenizer
        self.forward_hidden = forward_hidden
        self.lm_head = lm_head
        self.device = device
        self.cfg = cfg
        self.image_start_id = image_start_id
        self.pad_id = tokenizer.pad_token_id

        suffix = ANSWER_SUFFIXES[cfg.answer_suffix]
        self._suffix_text = (" " + suffix) if suffix else ""
        prefix = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                  f"<|im_start|>user\n<S{cfg.scale}>")
        self.prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        assert self.prefix_ids[-1] == tokenizer.convert_tokens_to_ids(f"<S{cfg.scale}>")
        self._question_cache = {}
        self._space_id = tokenizer(" ", add_special_tokens=False).input_ids
        self._space_id = self._space_id[0] if len(self._space_id) == 1 else None

    # -- prompt construction -------------------------------------------------

    def _question_ids(self, question: str) -> List[int]:
        ids = self._question_cache.get(question)
        if ids is None:
            text = f"\n{question}{self._suffix_text}<|im_end|>\n<|im_start|>assistant\n"
            ids = self.tok(text, add_special_tokens=False).input_ids
            self._question_cache[question] = ids
        return ids

    def _answer_ids(self, question: str, answer: str) -> List[int]:
        ids = set()
        for variant in answer_variants(question, answer):
            enc = self.tok(variant, add_special_tokens=False).input_ids
            if not enc:
                continue
            # ' 3' tokenises to [space, '3']; crediting a bare space would be
            # a spurious match, so skip variants whose first piece is a space.
            if enc[0] == self._space_id:
                continue
            ids.add(enc[0])
        return sorted(ids)

    # -- scoring -------------------------------------------------------------

    @torch.no_grad()
    def score_images(self, codes: Sequence[Sequence[int]],
                     vqa_lists: Sequence[Sequence[Tuple[str, str]]]
                     ) -> Tuple[List[float], List[float], List[List[float]]]:
        """Returns (AM, GM, per_question_probs) per image."""
        assert len(codes) == len(vqa_lists)
        rows, owners, answer_ids = [], [], []
        for i, (code, vqa) in enumerate(zip(codes, vqa_lists)):
            img_ids = [int(c) + self.image_start_id for c in code]
            for question, answer in vqa:
                rows.append(self.prefix_ids + img_ids + self._question_ids(question))
                owners.append(i)
                answer_ids.append(self._answer_ids(question, answer))

        scores = [0.0] * len(rows)
        for start in range(0, len(rows), self.cfg.batch_size):
            chunk = rows[start:start + self.cfg.batch_size]
            width = max(len(r) for r in chunk)
            input_ids = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(chunk), width), dtype=torch.long)
            last = torch.zeros(len(chunk), dtype=torch.long)
            for j, r in enumerate(chunk):
                input_ids[j, :len(r)] = torch.tensor(r, dtype=torch.long)
                attention_mask[j, :len(r)] = 1
                last[j] = len(r) - 1
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            hidden = self.forward_hidden(input_ids, attention_mask)
            h_last = hidden[torch.arange(len(chunk), device=hidden.device), last.to(hidden.device)]
            probs = torch.softmax(self.lm_head(h_last).float(), dim=-1)
            for j in range(len(chunk)):
                ids = answer_ids[start + j]
                scores[start + j] = float(probs[j, ids].sum()) if ids else 0.0

        per_question: List[List[float]] = [[] for _ in codes]
        for owner, s in zip(owners, scores):
            per_question[owner].append(s)
        am = [sum(q) / len(q) for q in per_question]
        gm = [geometric_mean(q) for q in per_question]
        return am, gm, per_question

    def combine(self, am: float, gm: float) -> float:
        return self.cfg.alpha * am + (1.0 - self.cfg.alpha) * gm
