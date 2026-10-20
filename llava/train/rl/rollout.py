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
sampled: the trainer supplies one ``Critique`` per live parent (e.g. the
oracle critique of the parent's failed VQA questions), its text is spliced in
as ``KIND_NONE`` exactly as the eval script splices decoded text
(``" " + text``), and every child of that parent shares it, so siblings differ
only in their image. A parent whose critique is ``None`` becomes a leaf.

With ``images_per_critique = M > 1`` (sampled critiques only) each parent gets
``branch[k]`` sampled critiques and every critique that refines gets ``M``
images: the first child of a critique (``critique_leader``) owns the sampled
reflection tokens (``KIND_TXT``), its ``M - 1`` followers carry the same text
as ``KIND_NONE`` and sample only an image. The critique is thus trained once,
with an advantage computed from the mean score of its ``M`` images against the
parent's other critiques (``Node.adv_txt``), while each image is trained
against the other images of the *same* critique (``Node.adv``). A "looks good"
critique is a single leaf (its followers would all be the parent's image).

``context_mode`` decides what a refinement round conditions on. ``"full"`` (the
default, as above) keeps the whole chain: prompt, draft, every critique and
every image so far. ``"mdp"`` rebuilds the context each round from the prompt
and the *latest* image only (``<prefix><latest image><im_end>\nSelf-reflect:``),
i.e. every round has exactly round 1's format, with the latest image standing in
for the draft; the critique of round k is also generated from that compact
context. Round 1 is identical in both modes. Training: with "full" a leaf's row
holds its ancestors' segments (trained along the first-born line); with "mdp"
every node's sequence is its own compact context, so each node gets its own row
training only its own segment (``build_training_rows(..., own_segment_only=True)``).

Chain mode (``chain_siblings = B > 0``, training rollouts only, i.e. ``run``
without an explicit ``branch``): instead of fanning out at every round, each
draft grows a single chain (one child per round, each under its own frozen
critique) up to a target round ``k*`` drawn uniformly from 1..D per draft
(D = ``len(branch) - 1``), and only at ``k*`` gets ``B`` siblings sharing the
chain's history and that round's critique; the siblings are leaves. Cost per
draft is (k* - 1) + B images instead of a full tree, and over many steps every
depth gets a clean sibling group. A chain that stops earlier (its image passes
every question, or a "looks good" critique) simply has nothing to fix there.
Chain nodes before ``k*`` are alone in their group, so their advantage is 0
(they keep the KL term, like a lone draft).
"""

import contextlib
import re
import warnings
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
    # Images sampled under each sampled critique (see the module docstring).
    # 1 = every child samples its own critique and image (t4). Ignored with a
    # critique_fn, where the children of a parent already share one critique.
    images_per_critique: int = 1
    # "full": refine on the whole chain so far; "mdp": prompt + latest image only
    # (see the module docstring).
    context_mode: str = "full"
    # Chain mode for training rollouts: >0 = B siblings at one random round per
    # draft, single children before it (see the module docstring); 0 = full tree.
    chain_siblings: int = 0

    @property
    def num_rounds(self) -> int:
        return len(self.branch) - 1


@dataclass
class Critique:
    """One frozen critique for a parent node, returned by ``critique_fn``.

    ``source`` / ``fallback`` are copied onto the parent's *children* (the
    nodes whose context the critique is), never onto the parent itself: a
    node's own ``critique_source`` must describe the critique it was refined
    under, which in deeper trees differs from the one chosen for its children.
    """
    text: str
    source: str = "oracle"     # "oracle" | "policy"
    fallback: bool = False     # hybrid: policy critique kept saying "looks good", used oracle


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
    critique_idx: int = 0                # which of the parent's critiques this node refines under
    critique_leader: bool = True         # owns the critique's sampled tokens (trains them)
    branch_round: int = 0                # chain mode: the round where this draft's chain branches
    children: List["Node"] = field(default_factory=list)
    # Filled in by the trainer. per_question is set once the image is scored
    # (critique_fn scores parents mid-rollout; score_tree does the rest).
    am: float = 0.0
    gm: float = 0.0
    per_question: Optional[List[float]] = None
    reward: float = 0.0                  # image-level reward (trainer's --break_weight)
    reward_txt: float = 0.0              # critique-level reward (--critique_break_weight)
    adv: float = 0.0                     # advantage of the image tokens
    adv_txt: float = 0.0                 # advantage of the reflection tokens (== adv unless M > 1)

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
        # Entered around every ``run`` (all policy sampling: drafts, sampled
        # critiques, refines). The trainer sets it to a MergedLoRA so sampling
        # runs on merged weights instead of peft's per-layer adapter matmuls.
        self.sampling_context = contextlib.nullcontext()

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
        if cfg.context_mode not in ("full", "mdp"):
            raise ValueError(f"context_mode must be 'full' or 'mdp', got {cfg.context_mode!r}")
        if SCALE_SEQ_LEN[cfg.scale] != cfg.gen_seq_len:
            raise ValueError(f"scale {cfg.scale} implies {SCALE_SEQ_LEN[cfg.scale]} "
                             f"image tokens, got gen_seq_len={cfg.gen_seq_len}")

    # -- helpers -------------------------------------------------------------

    def _state(self, n: Node):
        """(tokens, kinds) a refinement of ``n`` (or ``n``'s critique) conditions on.
        "full": ``n``'s whole sequence. "mdp": the prompt prefix + ``n``'s image as
        an untrained draft, the same layout as a round-0 node."""
        if self.cfg.context_mode == "full" or n.round == 0:
            return n.seq, n.kinds
        p = n.ancestors()[0].seg_start              # every sequence starts with the prompt prefix
        img = [self.img_start + c for c in n.codes]
        return n.seq[:p] + img, [KIND_NONE] * (p + len(img))

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
        if not self.cfg.reflect_sample and not getattr(self, "_warned_greedy_critiques", False):
            # Known limitation: this goes through _sample_reflections, which is
            # greedy when reflect_sample is False. A critique_fn that re-draws
            # "looks good" critiques (hybrid, see make_critique_fn) then gets
            # the identical text on every retry, wasting up to resample_tries
            # generate calls per parent before it falls back to the oracle,
            # and critique_fallback counts every policy-chosen parent whose
            # greedy critic says "looks good". Fix by sampling here regardless
            # of reflect_sample, or by short-circuiting the retries.
            warnings.warn("sample_critiques called with reflect_sample=False: critiques are "
                          "greedy, so re-sampling a 'looks good' critique cannot change it.")
            self._warned_greedy_critiques = True
        rows = [self._state(n)[0] + self.reflect_splice for n in parents]
        refl = self._sample_reflections(model, rows)
        return [self.tok.decode(r, skip_special_tokens=True).strip() for r in refl]

    # -- main entry ------------------------------------------------------------

    def run(self, model, prompts: List[Dict], branch: Optional[List[int]] = None,
            critique_fn: Optional[Callable[..., List[Optional[str]]]] = None
            ) -> RolloutBatch:
        """``critique_fn(parents, prompts, sample_fn) -> [Critique | None]``
        replaces reflection sampling with a fixed critique per parent (see the
        module docstring). ``sample_fn(nodes) -> [text]`` draws one policy
        reflection per node, for critique sources that want the frozen
        policy's own text."""
        chain = branch is None and self.cfg.chain_siblings > 0
        branch = list(branch or self.cfg.branch)
        was_training = model.training
        model.eval()
        try:
            with self.sampling_context:
                return self._run(model, prompts, branch, critique_fn, chain)
        finally:
            if was_training:
                model.train()

    def _run(self, model, prompts, branch, critique_fn=None, chain=False) -> RolloutBatch:
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
        depth = len(branch) - 1
        if chain and depth >= 1:
            # One target round per draft; the trainer seeds torch per (step, rank).
            for n, k in zip(roots, torch.randint(1, depth + 1, (len(roots),)).tolist()):
                n.branch_round = k

        nodes_by_round = [roots]
        leaves: List[Node] = []
        level = roots
        for k in range(1, len(branch)):
            live = [n for n in level if not n.looks_good]
            leaves.extend(n for n in level if n.looks_good)
            if chain:
                # Past its branch round a chain is finished: its nodes are leaves.
                leaves.extend(n for n in live if k > n.branch_round)
                live = [n for n in live if k <= n.branch_round]
            crits: List[Optional[Critique]] = [None] * len(live)
            if critique_fn is not None and live:
                crits = list(critique_fn(live, prompts, lambda nodes: self.sample_critiques(model, nodes)))
                assert len(crits) == len(live), "critique_fn must return one Critique per parent"
                assert all(c is None or isinstance(c, Critique) for c in crits), \
                    "critique_fn must return Critique objects (or None)"
                # No critique (nothing to fix): the parent's image is final.
                leaves.extend(n for n, c in zip(live, crits) if c is None)
                live = [n for n, c in zip(live, crits) if c is not None]
                crits = [c for c in crits if c is not None]
            children: List[Node] = []
            for n, crit in zip(live, crits):
                splice = self.reflect_splice
                if crit is not None:
                    # Spliced like the eval script splices decoded reflection
                    # text; untrained, so the child's segment is just its image.
                    splice = splice + self._encode(" " + crit.text)
                base_seq, base_kinds = self._state(n)
                n_kids = branch[k] if not chain else (self.cfg.chain_siblings if k == n.branch_round else 1)
                for g in range(n_kids):
                    seq = base_seq + splice
                    kinds = base_kinds + [KIND_NONE] * len(splice)
                    child = Node(prompt_idx=n.prompt_idx, round=k, parent=n, first_born=(g == 0),
                                 seq=seq, kinds=kinds, seg_start=len(seq), sibling_idx=g,
                                 critique_idx=g, teacher_forced=crit is not None,
                                 reflection=crit.text if crit is not None else "",
                                 critique_source=crit.source if crit is not None else "",
                                 critique_fallback=crit.fallback if crit is not None else False,
                                 branch_round=n.branch_round)
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
                children = self._add_critique_followers(children)
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

    def _add_critique_followers(self, leaders: List[Node]) -> List[Node]:
        """``images_per_critique - 1`` followers after every refining leader.

        A follower is the leader's sequence with the reflection re-marked as
        ``KIND_NONE`` (same ``seg_start``, so its segment is reflection + image
        like the leader's, but only the image is trained). The parents'
        ``children`` lists are rebuilt in the returned order.
        """
        M = self.cfg.images_per_critique
        if M <= 1 or not leaders:
            return leaders
        for lead in leaders:
            lead.parent.children = []
        out: List[Node] = []
        for lead in leaders:
            out.append(lead)
            lead.parent.children.append(lead)
            if lead.looks_good:
                continue
            for _ in range(M - 1):
                f = Node(prompt_idx=lead.prompt_idx, round=lead.round, parent=lead.parent,
                         first_born=False, seq=list(lead.seq),
                         kinds=lead.kinds[:lead.seg_start] + [KIND_NONE] * (len(lead.seq) - lead.seg_start),
                         seg_start=lead.seg_start, sibling_idx=lead.sibling_idx,
                         reflection=lead.reflection, reflection_len=lead.reflection_len,
                         forced_refine=lead.forced_refine, critique_source=lead.critique_source,
                         critique_idx=lead.critique_idx, critique_leader=False)
                out.append(f)
                lead.parent.children.append(f)
        return out

    # -- tensors for training ---------------------------------------------------

    def build_training_rows(self, leaves: List[Node], reflect_token_weight: float,
                            train_drafts: bool = True, train_images: bool = True,
                            image_adv_scale: float = 1.0, own_segment_only: bool = False):
        """Right-padded tensors for a list of leaves.

        Returns dict(input_ids, attention_mask, train_mask, pos_kind, adv, weight);
        adv / weight are per position (zero outside trained segments).
        ``train_drafts=False`` leaves the round-0 image out of the train mask:
        a draft with advantage 0 would otherwise still get the KL term, which
        pulls it back toward the reference. ``train_images=False`` leaves every
        image out (critic-only training: only ``KIND_TXT`` positions remain).
        ``image_adv_scale`` multiplies the image tokens' advantages; at 0 the
        images stay in the loss with the KL term only, i.e. they are anchored
        to the reference instead of dropped (dropping them lets the shared
        LoRA drift them freely: t10).
        ``own_segment_only=True`` (``context_mode="mdp"``): each element is a
        node, not a leaf; its sequence already holds the whole context it was
        generated from, so the row trains only the node's own segment.
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
            if own_segment_only:
                trained, chain = {leaf.round}, [leaf]
            for j, node in enumerate(chain):
                if node.round not in trained or (node.round == 0 and not train_drafts):
                    continue
                # A node's segment runs from its seg_start up to the child's
                # seg_start (the forced splice in between is KIND_NONE).
                end = chain[j + 1].seg_start if j + 1 < len(chain) else L
                for pos in range(node.seg_start, end):
                    kind = leaf.kinds[pos]
                    if kind == KIND_NONE or (kind == KIND_IMG and not train_images):
                        continue
                    train_mask[i, pos] = True
                    adv[i, pos] = node.adv_txt if kind == KIND_TXT else node.adv * image_adv_scale
                    weight[i, pos] = reflect_token_weight if kind == KIND_TXT else 1.0
        return dict(input_ids=input_ids, attention_mask=attention_mask,
                    train_mask=train_mask, pos_kind=pos_kind, adv=adv, weight=weight)
