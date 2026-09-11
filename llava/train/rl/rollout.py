"""Independent image/reflection episodes with final-outcome credit.

Each episode starts with a forced image prefix. Images contain a fixed number
of sampled semantic codes. After each image, a forced self-reflection prefix
lets the policy sample text until either tokenizer EOS (episode completion) or
<im_start> (another image). Both sampled boundary tokens are retained and
trained. Safety limits truncate episodes without inventing EOS. The last
complete image is scored even for truncated episodes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

from llava.train.rl.grpo import KIND_IMG, KIND_NONE, KIND_TXT

ITERATIVE_PROMPT_PREFIX = "Generate an image iteratively by self-reflecting and correcting.\n"
SYSTEM_PROMPT = "You are a helpful assistant."
REFLECT_SPLICE = "<im_end>\nSelf-reflect:"
SCALE_SEQ_LEN = {0: 729, 1: 169, 2: 81}


@dataclass
class RolloutConfig:
    num_rollouts: int = 4
    max_refinements: int = 3
    max_seq_len: int = 4096
    scale: int = 0
    gen_seq_len: int = 729
    img_temperature: float = 1.0
    img_top_k: int = 0
    img_top_p: float = 1.0
    reflect_tokens: int = 128
    reflect_sample: bool = True
    reflect_temperature: float = 1.0
    reflect_top_k: int = 0
    reflect_top_p: float = 1.0
    gen_batch_size: int = 16

    def validate_for_training(self):
        """The PPO log-probs implement only a vocabulary-restricted softmax."""
        if self.num_rollouts < 2:
            raise ValueError("RL training requires at least two rollouts per prompt.")
        if self.max_refinements < 0:
            raise ValueError("max_refinements must be nonnegative.")
        if min(self.max_seq_len, self.gen_seq_len, self.reflect_tokens, self.gen_batch_size) <= 0:
            raise ValueError("Sequence and batch limits must be positive.")
        if not self.reflect_sample:
            raise ValueError("RL training requires sampled reflections; greedy decoding is evaluation-only.")
        for phase in ("img", "reflect"):
            for setting, required in (("temperature", 1.0), ("top_k", 0), ("top_p", 1.0)):
                name = f"{phase}_{setting}"
                if getattr(self, name) != required:
                    raise ValueError(f"RL training requires --{name}={required}; "
                                     "other values make sampling disagree with PPO log-probs.")


class ImageVocabOnly(LogitsProcessor):
    def __init__(self, start: int, end: int):
        self.start, self.end = start, end

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.start:self.end] = 0.0
        return scores + mask


class NoImageVocab(LogitsProcessor):
    def __init__(self, start: int, end: int):
        self.start, self.end = start, end

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        scores[:, self.start:self.end] = float("-inf")
        return scores


@dataclass
class Trajectory:
    prompt_idx: int
    seq: List[int]
    kinds: List[int]
    images: List[List[int]] = field(default_factory=list)  # unshifted semantic codes
    reflections: List[str] = field(default_factory=list)
    stop_reason: Optional[str] = None
    am: float = 0.0
    gm: float = 0.0
    reward: float = 0.0
    adv: float = 0.0
    final_image: object = field(default=None, repr=False)  # exact rendered reward input (PIL)
    question_scores: List[float] = field(default_factory=list)
    vqa_list: List = field(default_factory=list)

    @property
    def codes(self):
        return self.images[-1]


@dataclass
class RolloutBatch:
    prompts: List[Dict]
    trajectories: List[Trajectory]


class EpisodeRollout:
    def __init__(self, tokenizer, cfg: RolloutConfig, image_start_id: int,
                 num_image_tokens: int, device):
        self.tok = tokenizer
        self.cfg = cfg
        self.device = device
        self.img_start = image_start_id
        self.img_end = image_start_id + num_image_tokens
        self.pad_id = tokenizer.pad_token_id
        assert self.pad_id is not None

        self.eos_id = tokenizer.eos_token_id
        self.im_start_tok = tokenizer.convert_tokens_to_ids("<im_start>")
        if self.eos_id is None or self.eos_id == self.im_start_tok:
            raise ValueError("Episode EOS must be defined and distinct from <im_start>.")
        if self.img_start <= self.eos_id < self.img_end:
            raise ValueError("Episode EOS must be outside the image vocabulary.")
        self.text_stop_ids = sorted({self.eos_id, self.im_start_tok})
        self._check_special_tokens()

        self.reflect_splice = self._encode(REFLECT_SPLICE)
        # <im_start> is sampled by the policy; only the scale is forced here.
        self.image_splice = self._encode(f"<S{cfg.scale}>")
        self.image_proc = LogitsProcessorList([ImageVocabOnly(self.img_start, self.img_end)])
        self.text_proc = LogitsProcessorList([NoImageVocab(self.img_start, self.img_end)])
        if SCALE_SEQ_LEN[cfg.scale] != cfg.gen_seq_len:
            raise ValueError(f"scale {cfg.scale} implies {SCALE_SEQ_LEN[cfg.scale]} "
                             f"image tokens, got gen_seq_len={cfg.gen_seq_len}")

    # -- helpers -------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def _check_special_tokens(self):
        for t in ("<im_start>", "<im_end>", f"<S{self.cfg.scale}>", "<I0>",
                  "<|im_start|>", "<|im_end|>"):
            expected = self.tok.convert_tokens_to_ids(t)
            got = self._encode(t)
            if expected is None or expected == self.tok.unk_token_id or got != [expected]:
                raise RuntimeError(f"Special token {t!r} does not round-trip: {got} vs {expected}")

    def make_prefix(self, prompt: str) -> List[int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ITERATIVE_PROMPT_PREFIX + prompt},
        ]
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        return self._encode(text + f"<im_start><S{self.cfg.scale}>")

    def _left_pad(self, rows: Sequence[Sequence[int]]):
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, width - len(r):] = torch.tensor(r, dtype=torch.long)
            mask[i, width - len(r):] = 1
        return ids.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def _sample_images(self, model, rows: List[List[int]]) -> List[List[int]]:
        out = []
        for start in range(0, len(rows), self.cfg.gen_batch_size):
            chunk = rows[start:start + self.cfg.gen_batch_size]
            ids, mask = self._left_pad(chunk)
            gen = model.generate(
                # A checkpoint's decoding defaults must not add unmodelled
                # probability transforms (including Transformers' fallback).
                generation_config=GenerationConfig(), use_model_defaults=False,
                input_ids=ids, attention_mask=mask,
                min_new_tokens=self.cfg.gen_seq_len, max_new_tokens=self.cfg.gen_seq_len,
                do_sample=True, temperature=self.cfg.img_temperature,
                top_k=self.cfg.img_top_k, top_p=self.cfg.img_top_p,
                repetition_penalty=1.0, logits_processor=self.image_proc,
                pad_token_id=self.pad_id, eos_token_id=None, use_cache=True,
            )[:, ids.shape[1]:]
            if gen.shape[1] != self.cfg.gen_seq_len:
                raise RuntimeError(f"image phase produced {gen.shape[1]} tokens")
            if int(gen.min()) < self.img_start or int(gen.max()) >= self.img_end:
                raise RuntimeError("image phase sampled a non-image token")
            out.extend(gen.tolist())
        return out

    @torch.no_grad()
    def _sample_reflections(self, model, rows: List[List[int]], max_new_tokens=None) -> List[List[int]]:
        out = []
        for start in range(0, len(rows), self.cfg.gen_batch_size):
            chunk = rows[start:start + self.cfg.gen_batch_size]
            ids, mask = self._left_pad(chunk)
            kwargs = dict(do_sample=False)
            if self.cfg.reflect_sample:
                kwargs = dict(do_sample=True, temperature=self.cfg.reflect_temperature,
                              top_k=self.cfg.reflect_top_k, top_p=self.cfg.reflect_top_p)
            gen = model.generate(
                generation_config=GenerationConfig(), use_model_defaults=False,
                input_ids=ids, attention_mask=mask,
                max_new_tokens=max_new_tokens or self.cfg.reflect_tokens, repetition_penalty=1.0,
                logits_processor=self.text_proc, pad_token_id=self.pad_id,
                eos_token_id=self.text_stop_ids, use_cache=True, **kwargs,
            )[:, ids.shape[1]:].tolist()
            for row in gen:
                # Keep the sampled EOS/transition action, discard only batch padding.
                cut = len(row)
                for pos, t in enumerate(row):
                    if t in self.text_stop_ids:
                        cut = pos + 1
                        break
                out.append(row[:cut])
        return out

    # -- complete independent episodes -----------------------------------------

    def run(self, model, prompts: List[Dict], num_rollouts: Optional[int] = None) -> RolloutBatch:
        count = self.cfg.num_rollouts if num_rollouts is None else num_rollouts
        if count < 1:
            raise ValueError("num_rollouts must be positive.")
        # Respect the model's architectural context limit as well as our budget.
        limit = min(self.cfg.max_seq_len,
                    getattr(model.config, "max_position_embeddings", self.cfg.max_seq_len))
        was_training = model.training
        model.eval()
        try:
            return self._run(model, prompts, count, limit)
        finally:
            model.train(was_training)

    def _run(self, model, prompts, count, limit):
        trajectories = []
        for p_idx, prompt in enumerate(prompts):
            prefix = self.make_prefix(prompt["prompt"])
            if len(prefix) + self.cfg.gen_seq_len + len(self.reflect_splice) + 1 > limit:
                raise ValueError("Sequence budget cannot fit the prompt, draft, and an EOS decision.")
            for _ in range(count):
                trajectories.append(Trajectory(p_idx, list(prefix), [KIND_NONE] * len(prefix)))

        def append_images(pending):
            images = self._sample_images(model, [t.seq for t in pending])
            for trajectory, image in zip(pending, images):
                trajectory.seq.extend(image)
                trajectory.kinds.extend([KIND_IMG] * len(image))
                trajectory.images.append([token - self.img_start for token in image])

        if not trajectories:
            return RolloutBatch(prompts, trajectories)
        append_images(trajectories)
        active = trajectories
        while active:
            # Group by remaining reflection budget so left-padding cannot make
            # one row exceed its own sequence limit.
            groups = {}
            for trajectory in active:
                room = limit - len(trajectory.seq) - len(self.reflect_splice)
                if room <= 0:
                    trajectory.stop_reason = "max_seq_len"
                    continue
                trajectory.seq.extend(self.reflect_splice)
                trajectory.kinds.extend([KIND_NONE] * len(self.reflect_splice))
                groups.setdefault(min(room, self.cfg.reflect_tokens), []).append(trajectory)
            pending = []
            for budget, group in groups.items():
                reflections = self._sample_reflections(model, [t.seq for t in group], budget)
                for trajectory, tokens in zip(group, reflections):
                    trajectory.seq.extend(tokens)
                    trajectory.kinds.extend([KIND_TXT] * len(tokens))
                    trajectory.reflections.append(self.tok.decode(tokens, skip_special_tokens=True).strip())
                    if tokens and tokens[-1] == self.eos_id:
                        trajectory.stop_reason = "eos"
                    elif tokens and tokens[-1] == self.im_start_tok:
                        if len(trajectory.images) - 1 >= self.cfg.max_refinements:
                            trajectory.stop_reason = "max_refinements"
                        elif len(trajectory.seq) + len(self.image_splice) + self.cfg.gen_seq_len > limit:
                            trajectory.stop_reason = "max_seq_len"
                        else:
                            trajectory.seq.extend(self.image_splice)
                            trajectory.kinds.extend([KIND_NONE] * len(self.image_splice))
                            pending.append(trajectory)
                    else:
                        trajectory.stop_reason = ("max_seq_len" if len(trajectory.seq) >= limit
                                                  else "reflection_limit")
            if pending:
                append_images(pending)
            active = pending
        return RolloutBatch(prompts, trajectories)

    def build_training_rows(self, trajectories: List[Trajectory], reflect_token_weight: float):
        """Every sampled token receives its episode's final-outcome advantage.

        Prompt/splice/padding positions are excluded. Each row's token weights
        sum to one, so the batch loss averages trajectories rather than letting
        long episodes dominate. Sampled EOS and <im_start> are text actions.
        """
        if not trajectories or reflect_token_weight <= 0:
            raise ValueError("Need trajectories and a positive reflection token weight.")
        width = max(len(t.seq) for t in trajectories)
        shape = (len(trajectories), width)
        input_ids = torch.full(shape, self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(shape, dtype=torch.long)
        pos_kind = torch.zeros(shape, dtype=torch.long)
        adv = torch.zeros(shape, dtype=torch.float32)
        weight = torch.zeros(shape, dtype=torch.float32)
        for i, trajectory in enumerate(trajectories):
            length = len(trajectory.seq)
            input_ids[i, :length] = torch.tensor(trajectory.seq)
            attention_mask[i, :length] = 1
            pos_kind[i, :length] = torch.tensor(trajectory.kinds)
            sampled = pos_kind[i] != KIND_NONE
            if sampled[0] or not sampled.any():
                raise ValueError("An episode needs a prompt and at least one sampled action.")
            adv[i, sampled] = trajectory.adv
            weight[i, pos_kind[i] == KIND_IMG] = 1.0
            weight[i, pos_kind[i] == KIND_TXT] = reflect_token_weight
            weight[i] /= weight[i].sum()
        return dict(input_ids=input_ids, attention_mask=attention_mask,
                    train_mask=pos_kind != KIND_NONE, pos_kind=pos_kind, adv=adv, weight=weight)
