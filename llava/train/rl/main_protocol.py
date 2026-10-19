"""Main-branch evaluation generation protocol (opt-in ``--protocol main_adhoc``).

Port of ``generate_batch`` phases 1-3 from the main branch's
``eval/iterative_generation_adhoc.py`` (origin/main f289996), so benchmark
numbers can be compared across branches. It differs from the native
EOS-driven policy (``EpisodeRollout.run``) on purpose:

* exactly one reflection; at most two images (one forced refinement);
* image tokens: temperature 1, top-k 1200, top-p 0.95, no vocabulary mask,
  rows cut at the first stop id and codes padded with 0 / truncated to 729;
* reflection: greedy, ``reflect_tokens`` new tokens, stops on EOS,
  <|im_end|>, <|endoftext|>, <im_start> or pad;
* a reflection containing "looks good" (case-insensitive) skips phase 3 and
  keeps the draft as the final image;
* phase 3 conditions on the decoded, stripped reflection re-encoded as
  ``" " + text`` followed by ``"\\n<im_start><S0>"`` (full history);
* sampling defaults not passed explicitly come from the checkpoint's own
  ``generation_config`` exactly as in main (``model.generate`` kwargs only).

Rendering, judging and seeding stay with the evaluation scripts.
"""

import re
from typing import Dict, List

import torch

from llava.train.rl.grpo import KIND_IMG, KIND_NONE, KIND_TXT
from llava.train.rl.rollout import ITERATIVE_PROMPT_PREFIX, SYSTEM_PROMPT, RolloutBatch, Trajectory

PROTOCOL = "main_adhoc_v1"
SOURCE = "origin/main f289996 eval/iterative_generation_adhoc.py generate_batch"
SETTINGS = dict(protocol=PROTOCOL, source=SOURCE, max_refinements=1, image_temperature=1.0,
                image_top_k=1200, image_top_p=0.95, image_vocab_mask=False, reflection="greedy",
                looks_good_skips_refinement=True, reflection_reencoded=True, context="full_history")
LOOKS_GOOD = "looks good"


def trim_at(row, stop_ids):
    """Cut a generated row at the first stop token (main's ``trim_at``)."""
    for position, token_id in enumerate(row):
        if token_id in stop_ids:
            return row[:position]
    return row


class MainAdhocRollout:
    """``run(model, prompts, num_rollouts=1)`` -> one ``Trajectory`` per prompt."""

    def __init__(self, tokenizer, device, reflect_tokens: int = 128, gen_seq_len: int = 729,
                 scale: int = 0, temperature: float = 1.0, top_p: float = 0.95, top_k: int = 1200,
                 system_prompt: str = SYSTEM_PROMPT):
        self.tok, self.device = tokenizer, device
        self.reflect_tokens, self.gen_seq_len, self.scale = reflect_tokens, gen_seq_len, scale
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.system_prompt = system_prompt
        self.pad_id = tokenizer.pad_token_id
        ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"),
               tokenizer.convert_tokens_to_ids("<|endoftext|>"),
               tokenizer.convert_tokens_to_ids("<im_start>"), self.pad_id]
        self.text_stop_ids = list(set(ids) - {None})
        self.image_token_map = {}
        for token, token_id in tokenizer.get_vocab().items():
            match = re.fullmatch(r"<I(\d+)>", token)
            if match:
                self.image_token_map[token_id] = int(match.group(1))
        if not self.image_token_map:
            raise ValueError("Tokenizer contains no <I...> image tokens")

    def encode(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def make_prefix(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": ITERATIVE_PROMPT_PREFIX + prompt}]
        return self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True) + f"<im_start><S{self.scale}>"

    def _left_pad(self, rows):
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, width - len(row):] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, width - len(row):] = 1
        return input_ids.to(self.device), attention_mask.to(self.device)

    def _images(self, model, rows):
        input_ids, attention_mask = self._left_pad(rows)
        generated = model.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=self.gen_seq_len,
            do_sample=True, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            repetition_penalty=1.0, pad_token_id=self.pad_id,
        )[:, input_ids.shape[1]:].tolist()
        codes, new_ids = [], []
        stops = set(self.text_stop_ids)
        for row in generated:
            row = trim_at(row, stops)
            row_codes = [self.image_token_map[t] for t in row if t in self.image_token_map]
            codes.append((row_codes + [0] * self.gen_seq_len)[:self.gen_seq_len])
            new_ids.append(row)
        return codes, new_ids

    def _kinds(self, ids):
        return [KIND_IMG if t in self.image_token_map else KIND_TXT for t in ids]

    @torch.inference_mode()
    def run(self, model, prompts: List[Dict], num_rollouts: int = 1) -> RolloutBatch:
        if num_rollouts != 1:
            raise ValueError("The main-branch protocol generates one episode per prompt.")
        was_training = model.training
        model.eval()
        try:
            return self._run(model, prompts)
        finally:
            model.train(was_training)

    def _run(self, model, prompts):
        # Phase 1: first image.
        phase1 = [self.encode(self.make_prefix(p["prompt"])) for p in prompts]
        codes1, image1_ids = self._images(model, phase1)
        # Phase 2: greedy self-reflection.
        reflect_prefix = self.encode("<im_end>\nSelf-reflect:")
        phase2 = [row + image + reflect_prefix for row, image in zip(phase1, image1_ids)]
        input_ids, attention_mask = self._left_pad(phase2)
        reflect_out = model.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=self.reflect_tokens,
            do_sample=False, repetition_penalty=1.0, pad_token_id=self.pad_id,
            eos_token_id=self.text_stop_ids,
        )[:, input_ids.shape[1]:].tolist()
        stops = set(self.text_stop_ids)
        reflection_ids = [trim_at(row, stops) for row in reflect_out]
        reflections = [self.tok.decode(row, skip_special_tokens=True).strip() for row in reflection_ids]
        # Phase 3: one forced refinement unless the reflection says "looks good".
        image_start = self.encode(f"\n<im_start><S{self.scale}>")
        pending = [i for i, text in enumerate(reflections) if LOOKS_GOOD not in text.lower()]
        phase3 = {i: phase2[i] + self.encode(" " + reflections[i]) + image_start for i in pending}
        codes2 = {}
        if pending:
            codes, ids = self._images(model, [phase3[i] for i in pending])
            codes2 = {i: (c, r) for i, c, r in zip(pending, codes, ids)}
        trajectories = []
        for i, prompt in enumerate(prompts):
            prefix_len = len(phase1[i])
            if i in codes2:
                seq = phase3[i] + codes2[i][1]
                kinds = ([KIND_NONE] * prefix_len + self._kinds(image1_ids[i]) + [KIND_NONE] * len(reflect_prefix)
                         + [KIND_NONE] * (len(phase3[i]) - len(phase2[i])) + self._kinds(codes2[i][1]))
                images, stop = [codes1[i], codes2[i][0]], "forced_refinement"
            else:
                seq = phase2[i] + reflection_ids[i]
                kinds = ([KIND_NONE] * prefix_len + self._kinds(image1_ids[i]) + [KIND_NONE] * len(reflect_prefix)
                         + [KIND_TXT] * len(reflection_ids[i]))
                images, stop = [codes1[i]], "looks_good"
            assert len(seq) == len(kinds)
            trajectories.append(Trajectory(i, seq, kinds, images=images, reflections=[reflections[i]],
                                           stop_reason=stop))
        return RolloutBatch(prompts, trajectories)
