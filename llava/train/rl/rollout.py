"""Forced multi-round tree rollout for generate -> self-reflect -> refine.

The schedule mirrors ``eval/iterative_generation_adhoc.py`` so the RL policy
is trained on exactly the sequence it is evaluated with::

    <prefix><im_start><S0>                     round 0: draft image (729 codes)
    <im_end>\\nSelf-reflect:                    reflection text (sampled, stops on
                                               EOS / <|im_end|> / <im_start>)
    \\n<im_start><S0>                           round k image, unless the
                                               reflection says "looks good"

Rollouts form a tree: every prompt gets ``branch[0]`` drafts, every live node of
round k-1 gets ``branch[k]`` (reflection, image) children. A node whose
reflection contains "looks good" is a leaf (its image is its parent's image).

Each node owns one *segment* of sampled tokens (its image for round 0, its
reflection + image for later rounds). A leaf's training sequence contains the
segments of all its ancestors; to count every sampled token once, a leaf trains
an ancestor's segment only if it is that ancestor's first-born line.

With a ``critique_fn`` (``run(..., critique_fn=...)``) the reflection is not
sampled: the trainer supplies one critique text per live parent (e.g. the
oracle critique of the parent's failed VQA questions), it is spliced in as
``KIND_NONE`` exactly as the eval script splices decoded text (``" " + text``),
and every child of that parent shares it, so siblings differ only in their
image. A parent whose critique is ``None`` becomes a leaf.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from llava.train.rl.grpo import KIND_IMG, KIND_NONE, KIND_TXT

ITERATIVE_PROMPT_PREFIX = "Generate an image iteratively by self-reflecting and correcting.\n"
SYSTEM_PROMPT = "You are a helpful assistant."
REFLECT_SPLICE = "<im_end>\nSelf-reflect:"
LOOKS_GOOD_RE = re.compile(r"looks good", re.IGNORECASE)
SCALE_SEQ_LEN = {0: 729, 1: 169, 2: 81}


@dataclass
class RolloutConfig:
    branch: List[int] = field(default_factory=lambda: [4, 2])   # per-round fan-out
    scale: int = 0
    gen_seq_len: int = 729
    img_temperature: float = 1.0
    img_top_k: int = 1200
    img_top_p: float = 0.95
    reflect_tokens: int = 128
    reflect_sample: bool = True
    reflect_temperature: float = 1.0
    reflect_top_k: int = 0
    reflect_top_p: float = 0.95
    gen_batch_size: int = 16
    # Forced refinement: the first ``min_refines`` children of every parent
    # must refine. If such a child's reflection says "looks good" it is
    # re-sampled up to ``refine_resample_tries`` times; if it still says so it
    # is left as a "looks good" leaf. Sibling-relative advantages are
    # unaffected, but the sampled reflection is then drawn from the policy
    # conditioned on "not looks good", which the training log-prob ignores.
    min_refines: int = 0
    refine_resample_tries: int = 3

    @property
    def num_rounds(self) -> int:
        return len(self.branch) - 1


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


@dataclass(eq=False)   # identity semantics: nodes are tree vertices, never compared by value
class Node:
    prompt_idx: int
    round: int
    parent: Optional["Node"]
    first_born: bool
    seq: List[int]                       # full token sequence so far
    kinds: List[int]                     # per-token KIND_* aligned with seq
    seg_start: int = 0                   # where this node's own segment starts
    sibling_idx: int = 0                 # position among the parent's children
    codes: Optional[List[int]] = None    # image of this round (parent's if looks_good)
    looks_good: bool = False
    reflection: str = ""
    reflection_len: int = 0
    forced_refine: bool = False          # a "looks good" was re-sampled away
    teacher_forced: bool = False         # reflection supplied by critique_fn, not sampled
    critique_source: str = ""            # "sampled" | "oracle" | "policy" (frozen, via critique_fn)
    critique_fallback: bool = False      # hybrid: policy critique kept saying "looks good", used oracle
    children: List["Node"] = field(default_factory=list)
    # Filled in by the trainer. per_question is set once the image is scored
    # (critique_fn scores parents mid-rollout; score_tree does the rest).
    am: float = 0.0
    gm: float = 0.0
    per_question: Optional[List[float]] = None
    reward: float = 0.0
    adv: float = 0.0

    def ancestors(self) -> List["Node"]:
        chain, n = [], self
        while n is not None:
            chain.append(n)
            n = n.parent
        return chain[::-1]

    def trained_rounds(self) -> List[int]:
        """Rounds whose segment this leaf trains: its own, plus each ancestor
        reachable through an unbroken first-born line."""
        rounds = [self.round]
        n = self
        while n.first_born and n.parent is not None:
            n = n.parent
            rounds.append(n.round)
        return rounds


@dataclass
class RolloutBatch:
    prompts: List[Dict]
    roots: List[Node]            # round-0 nodes
    leaves: List[Node]
    nodes_by_round: List[List[Node]]

    def all_nodes(self) -> List[Node]:
        return [n for level in self.nodes_by_round for n in level]


class TreeRollout:
    def __init__(self, tokenizer, cfg: RolloutConfig, image_start_id: int,
                 num_image_tokens: int, device):
        self.tok = tokenizer
        self.cfg = cfg
        self.device = device
        self.img_start = image_start_id
        self.img_end = image_start_id + num_image_tokens
        self.pad_id = tokenizer.pad_token_id
        assert self.pad_id is not None

        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.im_start_tok = tokenizer.convert_tokens_to_ids("<im_start>")
        self.text_stop_ids = sorted({tokenizer.eos_token_id, im_end_id, eot_id,
                                     self.im_start_tok, self.pad_id} - {None})
        self._check_special_tokens()

        self.reflect_splice = self._encode(REFLECT_SPLICE)
        self.image_splice = self._encode(f"\n<im_start><S{cfg.scale}>")
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
                input_ids=ids, attention_mask=mask,
                min_new_tokens=self.cfg.gen_seq_len, max_new_tokens=self.cfg.gen_seq_len,
                do_sample=True, temperature=self.cfg.img_temperature,
                top_k=self.cfg.img_top_k, top_p=self.cfg.img_top_p,
                repetition_penalty=1.0, logits_processor=self.image_proc,
                pad_token_id=self.pad_id, use_cache=True,
            )[:, ids.shape[1]:]
            if gen.shape[1] != self.cfg.gen_seq_len:
                raise RuntimeError(f"image phase produced {gen.shape[1]} tokens")
            if int(gen.min()) < self.img_start or int(gen.max()) >= self.img_end:
                raise RuntimeError("image phase sampled a non-image token")
            out.extend(gen.tolist())
        return out

    @torch.no_grad()
    def _sample_reflections(self, model, rows: List[List[int]]) -> List[List[int]]:
        out = []
        for start in range(0, len(rows), self.cfg.gen_batch_size):
            chunk = rows[start:start + self.cfg.gen_batch_size]
            ids, mask = self._left_pad(chunk)
            kwargs = dict(do_sample=False)
            if self.cfg.reflect_sample:
                kwargs = dict(do_sample=True, temperature=self.cfg.reflect_temperature,
                              top_k=self.cfg.reflect_top_k, top_p=self.cfg.reflect_top_p)
            gen = model.generate(
                input_ids=ids, attention_mask=mask,
                max_new_tokens=self.cfg.reflect_tokens, repetition_penalty=1.0,
                logits_processor=self.text_proc, pad_token_id=self.pad_id,
                eos_token_id=self.text_stop_ids, use_cache=True, **kwargs,
            )[:, ids.shape[1]:].tolist()
            for row in gen:
                # Cut at the first stop token (the stop token itself is dropped,
                # exactly as the eval script does before splicing the next image).
                cut = len(row)
                for pos, t in enumerate(row):
                    if t in self.text_stop_ids:
                        cut = pos
                        break
                out.append(row[:cut])
        return out

    @torch.no_grad()
    def sample_critiques(self, model, parents: List[Node]) -> List[str]:
        """One policy reflection per parent, as decoded text (for critique_fn:
        the text is re-encoded and spliced untrained, like the eval script)."""
        rows = [n.seq + self.reflect_splice for n in parents]
        refl = self._sample_reflections(model, rows)
        return [self.tok.decode(r, skip_special_tokens=True).strip() for r in refl]

    # -- main entry ------------------------------------------------------------

    def run(self, model, prompts: List[Dict], branch: Optional[List[int]] = None,
            critique_fn: Optional[Callable[..., List[Optional[str]]]] = None
            ) -> RolloutBatch:
        """``critique_fn(parents, prompts, sample_fn) -> [text | None]`` replaces
        reflection sampling with a fixed critique per parent (see the module
        docstring). ``sample_fn(nodes) -> [text]`` draws one policy reflection
        per node, for critique sources that want the frozen policy's own text."""
        branch = list(branch or self.cfg.branch)
        was_training = model.training
        model.eval()
        try:
            return self._run(model, prompts, branch, critique_fn)
        finally:
            if was_training:
                model.train()

    def _run(self, model, prompts, branch, critique_fn=None) -> RolloutBatch:
        # Round 0: drafts.
        roots: List[Node] = []
        for p_idx, p in enumerate(prompts):
            prefix = self.make_prefix(p["prompt"])
            for g in range(branch[0]):
                roots.append(Node(prompt_idx=p_idx, round=0, parent=None, first_born=(g == 0),
                                  seq=list(prefix), kinds=[KIND_NONE] * len(prefix),
                                  seg_start=len(prefix)))
        images = self._sample_images(model, [n.seq for n in roots])
        for n, img in zip(roots, images):
            n.seq.extend(img)
            n.kinds.extend([KIND_IMG] * len(img))
            n.codes = [t - self.img_start for t in img]

        nodes_by_round = [roots]
        leaves: List[Node] = []
        level = roots
        for k in range(1, len(branch)):
            live = [n for n in level if not n.looks_good]
            leaves.extend(n for n in level if n.looks_good)
            texts: List[Optional[str]] = [None] * len(live)
            if critique_fn is not None and live:
                texts = list(critique_fn(live, prompts, lambda nodes: self.sample_critiques(model, nodes)))
                assert len(texts) == len(live), "critique_fn must return one text per parent"
                # No critique (nothing to fix): the parent's image is final.
                leaves.extend(n for n, t in zip(live, texts) if t is None)
                live = [n for n, t in zip(live, texts) if t is not None]
                texts = [t for t in texts if t is not None]
            children: List[Node] = []
            for n, text in zip(live, texts):
                splice = self.reflect_splice
                if text is not None:
                    # Spliced like the eval script splices decoded reflection
                    # text; untrained, so the child's segment is just its image.
                    splice = splice + self._encode(" " + text)
                for g in range(branch[k]):
                    seq = n.seq + splice
                    kinds = n.kinds + [KIND_NONE] * len(splice)
                    child = Node(prompt_idx=n.prompt_idx, round=k, parent=n, first_born=(g == 0),
                                 seq=seq, kinds=kinds, seg_start=len(seq), sibling_idx=g,
                                 teacher_forced=text is not None, reflection=text or "",
                                 critique_source=n.critique_source if text is not None else "",
                                 critique_fallback=n.critique_fallback if text is not None else False)
                    n.children.append(child)
                    children.append(child)
            if not children:
                # Every live node said "looks good" (or had nothing to fix):
                # keep an (empty) level so all ranks report the same set of
                # per-round metrics.
                nodes_by_round.append(children)
                level = children
                continue
            if critique_fn is None:
                reflections = self._sample_reflections(model, [c.seq for c in children])
                for c, refl in zip(children, reflections):
                    self._set_reflection(c, refl)
                # Forced refinement: re-sample "looks good" away for the first
                # min_refines children of each parent.
                forced = [c for c in children
                          if c.looks_good and c.sibling_idx < self.cfg.min_refines]
                for _ in range(self.cfg.refine_resample_tries):
                    if not forced:
                        break
                    reflections = self._sample_reflections(model, [c.seq[:c.seg_start] for c in forced])
                    for c, refl in zip(forced, reflections):
                        self._set_reflection(c, refl)
                        if not c.looks_good:
                            c.forced_refine = True
                    forced = [c for c in forced if c.looks_good]
            pending = [c for c in children if not c.looks_good]
            for c in pending:
                c.seq.extend(self.image_splice)
                c.kinds.extend([KIND_NONE] * len(self.image_splice))
            if pending:
                images = self._sample_images(model, [c.seq for c in pending])
                for c, img in zip(pending, images):
                    c.seq.extend(img)
                    c.kinds.extend([KIND_IMG] * len(img))
                    c.codes = [t - self.img_start for t in img]
            for c in children:
                if c.looks_good:
                    c.codes = c.parent.codes
            nodes_by_round.append(children)
            level = children
        leaves.extend(level)
        return RolloutBatch(prompts=prompts, roots=roots, leaves=leaves,
                            nodes_by_round=nodes_by_round)

    def _set_reflection(self, c: Node, refl: List[int]):
        """(Re)write ``c``'s own segment with the sampled reflection tokens."""
        del c.seq[c.seg_start:]
        del c.kinds[c.seg_start:]
        c.seq.extend(refl)
        c.kinds.extend([KIND_TXT] * len(refl))
        c.reflection_len = len(refl)
        c.reflection = self.tok.decode(refl, skip_special_tokens=True).strip()
        c.looks_good = LOOKS_GOOD_RE.search(c.reflection) is not None
        c.critique_source = "sampled"

    # -- tensors for training ---------------------------------------------------

    def build_training_rows(self, leaves: List[Node], reflect_token_weight: float):
        """Right-padded tensors for a list of leaves.

        Returns dict(input_ids, attention_mask, train_mask, pos_kind, adv, weight);
        adv / weight are per position (zero outside trained segments).
        """
        width = max(len(n.seq) for n in leaves)
        B = len(leaves)
        input_ids = torch.full((B, width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, width), dtype=torch.long)
        train_mask = torch.zeros((B, width), dtype=torch.bool)
        pos_kind = torch.zeros((B, width), dtype=torch.long)
        adv = torch.zeros((B, width), dtype=torch.float32)
        weight = torch.zeros((B, width), dtype=torch.float32)
        for i, leaf in enumerate(leaves):
            L = len(leaf.seq)
            input_ids[i, :L] = torch.tensor(leaf.seq, dtype=torch.long)
            attention_mask[i, :L] = 1
            pos_kind[i, :L] = torch.tensor(leaf.kinds, dtype=torch.long)
            trained = set(leaf.trained_rounds())
            chain = leaf.ancestors()
            for j, node in enumerate(chain):
                if node.round not in trained:
                    continue
                # A node's segment runs from its seg_start up to the child's
                # seg_start (the forced splice in between is KIND_NONE).
                end = chain[j + 1].seg_start if j + 1 < len(chain) else L
                for pos in range(node.seg_start, end):
                    kind = leaf.kinds[pos]
                    if kind == KIND_NONE:
                        continue
                    train_mask[i, pos] = True
                    adv[i, pos] = node.adv
                    weight[i, pos] = reflect_token_weight if kind == KIND_TXT else 1.0
        return dict(input_ids=input_ids, attention_mask=attention_mask,
                    train_mask=train_mask, pos_kind=pos_kind, adv=adv, weight=weight)
