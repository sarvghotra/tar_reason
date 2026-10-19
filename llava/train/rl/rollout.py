"""Independent image/reflection episodes with final-outcome credit.

Each episode starts with a forced image prefix. Images contain a fixed number
of sampled semantic codes. After each image, a forced self-reflection prefix
lets the policy sample text until either tokenizer EOS (episode completion) or
<im_start> (another image). Both sampled boundary tokens are retained and
trained. Safety limits truncate episodes without inventing EOS. The last
complete image is scored even for truncated episodes.

``run_tree`` (opt-in ``--tree_branch``) samples the same episodes as a tree:
after image k each live branch samples R_k reflections, and each reflection
that requests another image samples M_k images. Each leaf is still one complete
episode; ``TreeNode`` records which sampled actions the leaves share.

``run_one_iteration`` (opt-in ``--objective janus_one_iteration``) samples a
draft, one reflection and at most one refinement. The verdict is read from the
reflection text ("Correction: looks good" = Janus-Pro-R1's "Yes"); the draft,
the reflection and the final image are trained as three separate segments.

``run_teacher_tree`` (opt-in ``--objective teacher_tree``) is a tree whose
reflections are written by a teacher VLM from the rendered image: "looks good"
stops, a correction is followed by M policy images. Teacher tokens are spliced
(distilled, never policy-gradient trained); after every image the policy also
samples "shadow" reflections from the same context, which are not continued.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

from llava.train.rl.grpo import KIND_IMG, KIND_NONE, KIND_TXT

ITERATIVE_PROMPT_PREFIX = "Generate an image iteratively by self-reflecting and correcting.\n"
SYSTEM_PROMPT = "You are a helpful assistant."
REFLECT_SPLICE = "<im_end>\nSelf-reflect:"
SCALE_SEQ_LEN = {0: 729, 1: 169, 2: 81}
LOOKS_GOOD_RE = re.compile(r"looks good", re.IGNORECASE)
CORRECTION_RE = re.compile(r"correction\s*:", re.IGNORECASE)
SEGMENT_DRAFT, SEGMENT_REFLECTION, SEGMENT_FINAL = 1, 2, 3
SEGMENT_TEACHER = 4  # teacher_tree: spliced teacher reflection (distilled)


def looks_good_verdict(text: str, rule: str = "correction"):
    """(verdict, has_correction_clause). ``correction``: "looks good" after the
    last "Correction:" (falls back to the whole text when there is no clause);
    ``anywhere``: anywhere in the text (main's evaluation rule)."""
    if rule == "anywhere":
        return LOOKS_GOOD_RE.search(text) is not None, CORRECTION_RE.search(text) is not None
    if rule != "correction":
        raise ValueError("verdict rule must be correction or anywhere")
    clauses = list(CORRECTION_RE.finditer(text))
    if not clauses:
        return LOOKS_GOOD_RE.search(text) is not None, False
    return LOOKS_GOOD_RE.search(text[clauses[-1].end():]) is not None, True


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
    img_sample: bool = True          # False: greedy image tokens (evaluation only)
    reflect_sample: bool = True
    reflect_temperature: float = 1.0
    reflect_top_k: int = 0
    reflect_top_p: float = 1.0
    gen_batch_size: int = 16
    reset_context_per_turn: bool = False

    def validate_for_training(self):
        """The PPO log-probs implement only a vocabulary-restricted softmax."""
        if self.num_rollouts < 2:
            raise ValueError("RL training requires at least two rollouts per prompt.")
        if self.max_refinements < 0:
            raise ValueError("max_refinements must be nonnegative.")
        if min(self.max_seq_len, self.gen_seq_len, self.reflect_tokens, self.gen_batch_size) <= 0:
            raise ValueError("Sequence and batch limits must be positive.")
        if not (self.reflect_sample and self.img_sample):
            raise ValueError("RL training requires sampled images and reflections; greedy decoding is evaluation-only.")
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
class RolloutSegment:
    """Exact model context and actions for one independently forwarded segment."""
    seq: List[int]
    kinds: List[int]


@dataclass
class OneIteration:
    """Exact per-segment contexts and actions of a one-iteration episode."""
    draft_context: List[int]
    draft: List[int]
    reflect_context: List[int]
    reflection: List[int]
    text: str
    verdict: bool                          # True = "looks good" (stop; final = draft)
    has_correction_clause: bool
    verdict_disagreement: bool             # the correction and anywhere rules differ
    stop_token_agrees: bool                # sampled EOS/<im_start> matches the verdict
    mismatch: Optional[str] = None         # looks_good_im_start | correction_eos | cap
    refine_context: Optional[List[int]] = None
    refine: Optional[List[int]] = None
    packed: bool = False                   # refine_context extends reflect_context + reflection
    am_draft: float = 0.0
    gm_draft: float = 0.0
    question_scores_draft: List[float] = field(default_factory=list)
    reward_draft: float = 0.0
    reward_verdict: float = 0.0
    reward_final: float = 0.0
    adv_draft: float = 0.0
    adv_verdict: float = 0.0
    adv_final: float = 0.0


@dataclass(eq=False)
class ShadowReflection:
    """teacher_tree: a policy reflection sampled after an image and not continued.

    ``context`` is the exact sampling context (the same one the image's teacher
    reflection extends); ``tokens`` end with the sampled EOS / <im_start> unless
    the token cap was hit (``capped``).
    """
    context: List[int] = field(repr=False)
    tokens: List[int]
    text: str
    verdict: bool
    stop_token_agrees: bool
    capped: bool
    reward: float = 0.0
    adv: float = 0.0


@dataclass(eq=False)
class TreeNode:
    """One sampled action span of a rollout tree.

    ``context`` is the exact conditioning (including forced splices and, under
    reset context, the copied latest image); ``actions`` are the sampled tokens.
    A reflection's actions end with its sampled EOS / <im_start> when present.
    teacher_tree: a ``teacher`` reflection's actions were spliced from the
    teacher (not sampled); image nodes keep their exact rendered ``image``, the
    teacher's raw reply, their judge scores and their shadow reflections.
    """
    id: int
    prompt_idx: int
    kind: str                                  # "image" | "reflection"
    parent: Optional["TreeNode"] = field(default=None, repr=False)
    context: List[int] = field(default_factory=list, repr=False)
    actions: List[int] = field(default_factory=list, repr=False)
    action_kinds: List[int] = field(default_factory=list, repr=False)
    children: List["TreeNode"] = field(default_factory=list, repr=False)
    mass: float = 1.0      # draft = 1; children split their parent's mass equally
    value: float = 0.0     # mean final reward of the subtree
    adv: float = 0.0
    teacher: bool = False
    image: object = field(default=None, repr=False)
    teacher_reply: Optional[str] = field(default=None, repr=False)
    teacher_verdict: Optional[bool] = None
    shadows: List[ShadowReflection] = field(default_factory=list, repr=False)
    am: float = 0.0
    gm: float = 0.0
    question_scores: List[float] = field(default_factory=list, repr=False)


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
    segments: List[RolloutSegment] = field(default_factory=list)
    nodes: List[TreeNode] = field(default_factory=list, repr=False)  # tree path, root first
    mass: float = 1.0      # this leaf's share of its draft (1 without branching)
    one_iter: Optional[OneIteration] = field(default=None, repr=False)
    verdicts: List[bool] = field(default_factory=list)  # run_looks_good: per reflection
    scored_images: List = field(default_factory=list, repr=False)  # exact scored PILs, in order
    image_scores: List = field(default_factory=list, repr=False)   # (am, gm, per-question) per image

    @property
    def codes(self):
        return self.images[-1]

    @property
    def final_image_node(self) -> Optional[TreeNode]:
        return next((n for n in reversed(self.nodes) if n.kind == "image"), None)


@dataclass
class RolloutBatch:
    prompts: List[Dict]
    trajectories: List[Trajectory]
    nodes: List[TreeNode] = field(default_factory=list)  # tree mode only, creation order


def parse_tree_branch(spec) -> List[List[int]]:
    """``"R[xM],..."`` -> canonical ``[[R, M], ...]`` with trailing ``[1, 1]`` removed.

    Entry k applies after image k (image 0 is the draft): R reflections per
    image, and M images for each reflection that requests another image.
    """
    if isinstance(spec, (list, tuple)):
        parts = [list(e) for e in spec]
    else:
        spec = str(spec).strip()
        parts = []
        for part in spec.split(",") if spec else []:
            r, sep, m = part.strip().lower().partition("x")
            try:
                parts.append([int(r), int(m) if sep else 1])
            except ValueError:
                raise ValueError(f"Invalid tree branch entry {part!r}; use R or RxM, e.g. 2,2x2.")
    if any(len(e) != 2 or type(e[0]) is not int or type(e[1]) is not int
           or e[0] < 1 or e[1] < 1 for e in parts):
        raise ValueError(f"Tree branch factors must be positive integers, got {spec!r}.")
    while parts and parts[-1] == [1, 1]:
        parts.pop()
    return parts


def validate_tree_branch(branch: List[List[int]], max_refinements: int, num_rollouts: int,
                         max_leaves: int):
    if len(branch) > max_refinements:
        raise ValueError(f"tree_branch has {len(branch)} rounds but max_refinements={max_refinements}; "
                         "the reflection after the last allowed image can only stop.")
    worst = num_rollouts
    for r, m in branch:
        worst *= r * m
    if worst > max_leaves:
        raise ValueError(f"tree_branch allows up to {worst} leaves per prompt, above "
                         f"max_tree_leaves_per_prompt={max_leaves}.")


def assign_tree_masses(nodes: List[TreeNode], trajectories: List[Trajectory]):
    """Drafts have mass 1 and children split their parent's mass equally, so a
    node's weight does not depend on how many leaves its own action spawned."""
    for node in nodes:  # creation order: parents first
        node.mass = 1.0 if node.parent is None else node.parent.mass / len(node.parent.children)
    for trajectory in trajectories:
        trajectory.mass = trajectory.nodes[-1].mass


def order_tree_leaves(nodes: List[TreeNode], trajectories: List[Trajectory]) -> List[Trajectory]:
    """Leaves in depth-first order; every leaf ends on its own terminal node."""
    terminal = {}
    for trajectory in trajectories:
        if trajectory.nodes[-1] in terminal or trajectory.nodes[-1].children:
            raise RuntimeError("Each tree leaf must end on its own childless node.")
        terminal[trajectory.nodes[-1]] = trajectory
    ordered = []

    def visit(node):
        if node.children:
            for child in node.children:
                visit(child)
        elif node in terminal:
            ordered.append(terminal[node])
        else:
            raise RuntimeError("A childless tree node does not end any leaf.")

    for root in (n for n in nodes if n.parent is None):
        visit(root)
    if len(ordered) != len(trajectories):
        raise RuntimeError("Tree leaves and terminal nodes disagree.")
    return ordered


def _clone(t: Trajectory) -> Trajectory:
    """Independent copy at a branch point; the shared path keeps node identity."""
    c = Trajectory(t.prompt_idx, list(t.seq), list(t.kinds))
    c.images = [list(image) for image in t.images]
    c.reflections = list(t.reflections)
    c.segments = [RolloutSegment(list(s.seq), list(s.kinds)) for s in t.segments]
    c.nodes = list(t.nodes)
    return c


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
            kwargs = dict(do_sample=False)
            if self.cfg.img_sample:
                kwargs = dict(do_sample=True, temperature=self.cfg.img_temperature,
                              top_k=self.cfg.img_top_k, top_p=self.cfg.img_top_p)
            gen = model.generate(
                # A checkpoint's decoding defaults must not add unmodelled
                # probability transforms (including Transformers' fallback).
                generation_config=GenerationConfig(), use_model_defaults=False,
                input_ids=ids, attention_mask=mask,
                min_new_tokens=self.cfg.gen_seq_len, max_new_tokens=self.cfg.gen_seq_len,
                **kwargs, repetition_penalty=1.0, logits_processor=self.image_proc,
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
                if self.cfg.reset_context_per_turn:
                    trajectories[-1].segments.append(
                        RolloutSegment(list(prefix), [KIND_NONE] * len(prefix)))

        def context(t):
            return t.segments[-1].seq if self.cfg.reset_context_per_turn else t.seq

        def append(t, tokens, kind):
            # seq remains the logical episode for budgets, metrics and audit.
            # Segments separately preserve the exact conditioning used to sample.
            t.seq.extend(tokens)
            t.kinds.extend([kind] * len(tokens))
            if self.cfg.reset_context_per_turn:
                t.segments[-1].seq.extend(tokens)
                t.segments[-1].kinds.extend([kind] * len(tokens))

        def append_images(pending):
            images = self._sample_images(model, [context(t) for t in pending])
            for trajectory, image in zip(pending, images):
                append(trajectory, image, KIND_IMG)
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
                if self.cfg.reset_context_per_turn and len(trajectory.images) > 1:
                    prefix = self.make_prefix(prompts[trajectory.prompt_idx]["prompt"])
                    current = prefix + [code + self.img_start for code in trajectory.codes]
                    # Copied image/prefix are observations, never duplicate actions.
                    trajectory.segments.append(RolloutSegment(current, [KIND_NONE] * len(current)))
                append(trajectory, self.reflect_splice, KIND_NONE)
                groups.setdefault(min(room, self.cfg.reflect_tokens), []).append(trajectory)
            pending = []
            for budget, group in groups.items():
                reflections = self._sample_reflections(model, [context(t) for t in group], budget)
                for trajectory, tokens in zip(group, reflections):
                    append(trajectory, tokens, KIND_TXT)
                    trajectory.reflections.append(self.tok.decode(tokens, skip_special_tokens=True).strip())
                    if tokens and tokens[-1] == self.eos_id:
                        trajectory.stop_reason = "eos"
                    elif tokens and tokens[-1] == self.im_start_tok:
                        if len(trajectory.images) - 1 >= self.cfg.max_refinements:
                            trajectory.stop_reason = "max_refinements"
                        elif len(trajectory.seq) + len(self.image_splice) + self.cfg.gen_seq_len > limit:
                            trajectory.stop_reason = "max_seq_len"
                        else:
                            append(trajectory, self.image_splice, KIND_NONE)
                            pending.append(trajectory)
                    else:
                        trajectory.stop_reason = ("max_seq_len" if len(trajectory.seq) >= limit
                                                  else "reflection_limit")
            if pending:
                append_images(pending)
            active = pending
        return RolloutBatch(prompts, trajectories)

    # -- tree of episodes ------------------------------------------------------

    def run_tree(self, model, prompts: List[Dict], branch, num_rollouts: Optional[int] = None
                 ) -> RolloutBatch:
        """Like ``run``, but branch after image k into ``branch[k] = [R, M]``.

        With every factor 1 this issues the same generate calls as ``run`` and
        returns the same episodes. Clones are taken only after the budget room
        check (R) and after both cap checks (M), so each leaf ends on its own node.
        """
        branch = parse_tree_branch(branch)
        count = self.cfg.num_rollouts if num_rollouts is None else num_rollouts
        if count < 1:
            raise ValueError("num_rollouts must be positive.")
        limit = min(self.cfg.max_seq_len,
                    getattr(model.config, "max_position_embeddings", self.cfg.max_seq_len))
        was_training = model.training
        model.eval()
        try:
            return self._run_tree(model, prompts, count, limit, branch)
        finally:
            model.train(was_training)

    def _run_tree(self, model, prompts, count, limit, branch):
        trajectories, nodes = [], []
        for p_idx, prompt in enumerate(prompts):
            prefix = self.make_prefix(prompt["prompt"])
            if len(prefix) + self.cfg.gen_seq_len + len(self.reflect_splice) + 1 > limit:
                raise ValueError("Sequence budget cannot fit the prompt, draft, and an EOS decision.")
            for _ in range(count):
                trajectories.append(Trajectory(p_idx, list(prefix), [KIND_NONE] * len(prefix)))
                if self.cfg.reset_context_per_turn:
                    trajectories[-1].segments.append(
                        RolloutSegment(list(prefix), [KIND_NONE] * len(prefix)))

        def context(t):
            return t.segments[-1].seq if self.cfg.reset_context_per_turn else t.seq

        def append(t, tokens, kind):
            t.seq.extend(tokens)
            t.kinds.extend([kind] * len(tokens))
            if self.cfg.reset_context_per_turn:
                t.segments[-1].seq.extend(tokens)
                t.segments[-1].kinds.extend([kind] * len(tokens))

        def new_node(t, kind):
            parent = t.nodes[-1] if t.nodes else None
            node = TreeNode(len(nodes), t.prompt_idx, kind, parent, list(context(t)))
            if parent is not None:
                parent.children.append(node)
            nodes.append(node)
            t.nodes.append(node)
            return node

        def factors(t):
            k = len(t.images) - 1
            return branch[k] if k < len(branch) else [1, 1]

        def append_images(pending):
            for trajectory in pending:
                new_node(trajectory, "image")
            images = self._sample_images(model, [context(t) for t in pending])
            for trajectory, image in zip(pending, images):
                append(trajectory, image, KIND_IMG)
                trajectory.nodes[-1].actions = list(image)
                trajectory.nodes[-1].action_kinds = [KIND_IMG] * len(image)
                trajectory.images.append([token - self.img_start for token in image])

        if not trajectories:
            return RolloutBatch(prompts, trajectories)
        append_images(trajectories)
        active = list(trajectories)  # clones are appended to trajectories below
        while active:
            groups = {}
            for trajectory in active:
                room = limit - len(trajectory.seq) - len(self.reflect_splice)
                if room <= 0:
                    trajectory.stop_reason = "max_seq_len"
                    continue
                if self.cfg.reset_context_per_turn and len(trajectory.images) > 1:
                    prefix = self.make_prefix(prompts[trajectory.prompt_idx]["prompt"])
                    current = prefix + [code + self.img_start for code in trajectory.codes]
                    trajectory.segments.append(RolloutSegment(current, [KIND_NONE] * len(current)))
                append(trajectory, self.reflect_splice, KIND_NONE)
                copies = [trajectory] + [_clone(trajectory) for _ in range(factors(trajectory)[0] - 1)]
                trajectories.extend(copies[1:])
                for copy in copies:
                    new_node(copy, "reflection")
                groups.setdefault(min(room, self.cfg.reflect_tokens), []).extend(copies)
            pending = []
            for budget, group in groups.items():
                reflections = self._sample_reflections(model, [context(t) for t in group], budget)
                for trajectory, tokens in zip(group, reflections):
                    append(trajectory, tokens, KIND_TXT)
                    trajectory.nodes[-1].actions = list(tokens)
                    trajectory.nodes[-1].action_kinds = [KIND_TXT] * len(tokens)
                    trajectory.reflections.append(self.tok.decode(tokens, skip_special_tokens=True).strip())
                    if tokens and tokens[-1] == self.eos_id:
                        trajectory.stop_reason = "eos"
                    elif tokens and tokens[-1] == self.im_start_tok:
                        if len(trajectory.images) - 1 >= self.cfg.max_refinements:
                            trajectory.stop_reason = "max_refinements"
                        elif len(trajectory.seq) + len(self.image_splice) + self.cfg.gen_seq_len > limit:
                            trajectory.stop_reason = "max_seq_len"
                        else:
                            copies = [trajectory] + [_clone(trajectory)
                                                     for _ in range(factors(trajectory)[1] - 1)]
                            trajectories.extend(copies[1:])
                            for copy in copies:
                                append(copy, self.image_splice, KIND_NONE)
                                pending.append(copy)
                    else:
                        trajectory.stop_reason = ("max_seq_len" if len(trajectory.seq) >= limit
                                                  else "reflection_limit")
            if pending:
                append_images(pending)
            active = pending
        assign_tree_masses(nodes, trajectories)
        return RolloutBatch(prompts, order_tree_leaves(nodes, trajectories), nodes)

    # -- teacher-written reflections over a tree of images --------------------

    def run_teacher_tree(self, model, prompts: List[Dict], branch, teacher_fn, num_shadows: int = 1,
                         verdict_rule: str = "correction", num_rollouts: Optional[int] = None
                         ) -> RolloutBatch:
        """Images from the policy, reflections from a teacher (``--objective teacher_tree``).

        After each round of sampled images, ``teacher_fn(prompt_texts, codes, ask)``
        returns one ``(pil, reply)`` per image: the exact rendered pixels (kept on
        the image node, so the judge later scores what the teacher saw) and the
        teacher's raw reply, or None where ``ask`` is False (images at the
        refinement cap). A parsed reply becomes a ``teacher`` reflection node in
        the policy's own format: "looks good" ends the episode with EOS (stop
        ``teacher_looks_good``); a correction ends with <im_start> and is followed
        by ``branch[k] = [1, M]`` sampled images. Unparseable replies stop the
        episode (``teacher_malformed``). After every image the policy samples
        ``num_shadows`` reflections from the teacher reflection's context; they
        are recorded on the image node and never continued. Budgets follow
        ``run_tree`` (the logical episode must fit ``max_seq_len``).
        """
        from llava.train.rl.teacher import parse_teacher_text
        branch = parse_tree_branch(branch)
        if any(r != 1 for r, _ in branch):
            raise ValueError("teacher_tree branches only over images: every tree_branch entry needs R = 1.")
        if num_shadows < 0:
            raise ValueError("num_shadows must be nonnegative.")
        looks_good_verdict("", verdict_rule)  # validate the rule name
        count = self.cfg.num_rollouts if num_rollouts is None else num_rollouts
        if count < 1:
            raise ValueError("num_rollouts must be positive.")
        limit = min(self.cfg.max_seq_len,
                    getattr(model.config, "max_position_embeddings", self.cfg.max_seq_len))
        was_training = model.training
        model.eval()
        try:
            return self._run_teacher_tree(model, prompts, count, limit, branch, teacher_fn, num_shadows,
                                          verdict_rule, parse_teacher_text)
        finally:
            model.train(was_training)

    def _sample_shadows(self, model, image_nodes, contexts, num_shadows, limit, verdict_rule):
        groups = {}
        for node, context in zip(image_nodes, contexts):
            budget = min(self.cfg.reflect_tokens, limit - len(context))
            if budget > 0:
                groups.setdefault(budget, []).extend([(node, context)] * num_shadows)
        for budget, group in groups.items():
            samples = self._sample_reflections(model, [context for _, context in group], budget)
            for (node, context), tokens in zip(group, samples):
                text = self.tok.decode(tokens, skip_special_tokens=True).strip()
                verdict = looks_good_verdict(text, verdict_rule)[0]
                last = tokens[-1] if tokens else None
                node.shadows.append(ShadowReflection(
                    list(context), list(tokens), text, verdict,
                    stop_token_agrees=(last == self.eos_id) if verdict else (last == self.im_start_tok),
                    capped=last not in (self.eos_id, self.im_start_tok)))

    def _run_teacher_tree(self, model, prompts, count, limit, branch, teacher_fn, num_shadows,
                          verdict_rule, parse):
        reset = self.cfg.reset_context_per_turn
        trajectories, nodes = [], []
        for p_idx, prompt in enumerate(prompts):
            prefix = self.make_prefix(prompt["prompt"])
            if len(prefix) + self.cfg.gen_seq_len + len(self.reflect_splice) + 1 > limit:
                raise ValueError("Sequence budget cannot fit the prompt, draft, and an EOS decision.")
            for _ in range(count):
                trajectories.append(Trajectory(p_idx, list(prefix), [KIND_NONE] * len(prefix)))
                if reset:
                    trajectories[-1].segments.append(RolloutSegment(list(prefix), [KIND_NONE] * len(prefix)))

        def context(t):
            return t.segments[-1].seq if reset else t.seq

        def append(t, tokens, kind):
            t.seq.extend(tokens)
            t.kinds.extend([kind] * len(tokens))
            if reset:
                t.segments[-1].seq.extend(tokens)
                t.segments[-1].kinds.extend([kind] * len(tokens))

        def new_node(t, kind):
            parent = t.nodes[-1] if t.nodes else None
            node = TreeNode(len(nodes), t.prompt_idx, kind, parent, list(context(t)))
            if parent is not None:
                parent.children.append(node)
            nodes.append(node)
            t.nodes.append(node)
            return node

        def latest_image_context(t):
            # Reset: the prompt and the latest image only (copied, never an action).
            current = self.make_prefix(prompts[t.prompt_idx]["prompt"])
            return current + [code + self.img_start for code in t.codes]

        def reflection_context(t):
            base = latest_image_context(t) if reset and len(t.images) > 1 else context(t)
            return base + self.reflect_splice

        def append_images(pending):
            for trajectory in pending:
                new_node(trajectory, "image")
            images = self._sample_images(model, [context(t) for t in pending])
            for trajectory, image in zip(pending, images):
                append(trajectory, image, KIND_IMG)
                trajectory.nodes[-1].actions = list(image)
                trajectory.nodes[-1].action_kinds = [KIND_IMG] * len(image)
                trajectory.images.append([token - self.img_start for token in image])

        if not trajectories:
            return RolloutBatch(prompts, trajectories, nodes)
        append_images(trajectories)
        active = list(trajectories)  # clones are appended to trajectories below
        while active:
            ask = [len(t.images) - 1 < self.cfg.max_refinements for t in active]
            replies = teacher_fn([prompts[t.prompt_idx]["prompt"] for t in active],
                                 [t.codes for t in active], ask)
            if len(replies) != len(active):
                raise RuntimeError("teacher_fn must return one (image, reply) per image.")
            for t, asked, (image, reply) in zip(active, ask, replies):
                if image is None or (reply is None) == asked:
                    raise RuntimeError("teacher_fn must render every image and reply exactly where asked.")
                t.nodes[-1].image, t.nodes[-1].teacher_reply = image, reply
            if num_shadows:
                self._sample_shadows(model, [t.nodes[-1] for t in active],
                                     [reflection_context(t) for t in active], num_shadows, limit, verdict_rule)
            pending = []
            for t, asked in zip(active, ask):
                image_node = t.nodes[-1]
                if not asked:
                    t.stop_reason = "max_refinements"
                    continue
                parsed = parse(image_node.teacher_reply)
                tokens = None
                if parsed is not None:
                    text, verdict = parsed
                    tokens = self._encode(text) + [self.eos_id if verdict else self.im_start_tok]
                    if any(self.img_start <= x < self.img_end or x in self.text_stop_ids for x in tokens[:-1]):
                        tokens = None  # the text must encode as ordinary reflection tokens
                if tokens is None or len(tokens) > self.cfg.reflect_tokens:
                    t.stop_reason = "teacher_malformed"
                    continue
                image_node.teacher_verdict = verdict
                if len(t.seq) + len(self.reflect_splice) + len(tokens) > limit:
                    t.stop_reason = "max_seq_len"
                    continue
                if reset and len(t.images) > 1:
                    current = latest_image_context(t)
                    t.segments.append(RolloutSegment(current, [KIND_NONE] * len(current)))
                append(t, self.reflect_splice, KIND_NONE)
                reflection = new_node(t, "reflection")
                reflection.teacher = True
                reflection.actions, reflection.action_kinds = list(tokens), [KIND_TXT] * len(tokens)
                append(t, tokens, KIND_TXT)
                t.reflections.append(text.strip())
                if verdict:
                    t.stop_reason = "teacher_looks_good"
                    continue
                if len(t.seq) + len(self.image_splice) + self.cfg.gen_seq_len > limit:
                    t.stop_reason = "max_seq_len"
                    continue
                k = len(t.images) - 1
                images_per_reflection = branch[k][1] if k < len(branch) else 1
                copies = [t] + [_clone(t) for _ in range(images_per_reflection - 1)]
                trajectories.extend(copies[1:])
                for copy in copies:
                    append(copy, self.image_splice, KIND_NONE)
                    pending.append(copy)
            if pending:
                append_images(pending)
            active = pending
        assign_tree_masses(nodes, trajectories)
        return RolloutBatch(prompts, order_tree_leaves(nodes, trajectories), nodes)

    # -- one iteration with a text verdict (Janus-style) ----------------------

    def run_one_iteration(self, model, prompts: List[Dict], num_rollouts: Optional[int] = None,
                          verdict_rule: str = "correction") -> RolloutBatch:
        """Draft -> reflection -> one refinement unless the Correction says "looks good".

        Every sampled token keeps its exact sampling context. A refinement after
        a reflection that did not end in a sampled ``<im_start>`` (EOS or token
        cap) drops a trailing EOS and splices ``<im_start><S0>`` (newline-aware).
        """
        count = self.cfg.num_rollouts if num_rollouts is None else num_rollouts
        if count < 1:
            raise ValueError("num_rollouts must be positive.")
        looks_good_verdict("", verdict_rule)  # validate the rule name
        limit = min(self.cfg.max_seq_len,
                    getattr(model.config, "max_position_embeddings", self.cfg.max_seq_len))
        was_training = model.training
        model.eval()
        try:
            return self._run_one_iteration(model, prompts, count, limit, verdict_rule)
        finally:
            model.train(was_training)

    def _forced_image_splice(self, tokens: List[int]) -> List[int]:
        newline = bool(tokens) and self.tok.decode([tokens[-1]], skip_special_tokens=False).endswith("\n")
        splice = self._encode(("" if newline else "\n") + f"<im_start><S{self.cfg.scale}>")
        expected = ([] if newline else self._encode("\n")) + [self.im_start_tok] + self.image_splice
        if splice != expected:
            raise RuntimeError(f"Forced image splice encodes unexpectedly: {splice} vs {expected}")
        return splice

    def _run_one_iteration(self, model, prompts, count, limit, verdict_rule):
        trajectories = []
        for p_idx, prompt in enumerate(prompts):
            prefix = self.make_prefix(prompt["prompt"])
            need = (len(prefix) + 2 * self.cfg.gen_seq_len + len(self.reflect_splice)
                    + self.cfg.reflect_tokens + len(self.image_splice) + 2)
            if need > limit:
                raise ValueError("Sequence budget cannot fit a draft, a reflection and one refinement.")
            for _ in range(count):
                trajectories.append(Trajectory(p_idx, list(prefix), [KIND_NONE] * len(prefix)))
        if not trajectories:
            return RolloutBatch(prompts, trajectories)
        drafts = self._sample_images(model, [list(t.seq) for t in trajectories])
        for t, image in zip(trajectories, drafts):
            t.one_iter = OneIteration(draft_context=list(t.seq), draft=list(image), reflect_context=[],
                                      reflection=[], text="", verdict=False, has_correction_clause=False,
                                      verdict_disagreement=False, stop_token_agrees=False)
            t.seq.extend(image)
            t.kinds.extend([KIND_IMG] * len(image))
            t.images.append([token - self.img_start for token in image])
            t.seq.extend(self.reflect_splice)
            t.kinds.extend([KIND_NONE] * len(self.reflect_splice))
            t.one_iter.reflect_context = list(t.seq)
        reflections = self._sample_reflections(model, [t.one_iter.reflect_context for t in trajectories],
                                               self.cfg.reflect_tokens)
        pending = []
        for t, tokens in zip(trajectories, reflections):
            oi = t.one_iter
            oi.reflection = list(tokens)
            oi.text = self.tok.decode(tokens, skip_special_tokens=True).strip()
            t.reflections.append(oi.text)
            oi.verdict, oi.has_correction_clause = looks_good_verdict(oi.text, verdict_rule)
            other = looks_good_verdict(oi.text, "anywhere" if verdict_rule == "correction" else "correction")[0]
            oi.verdict_disagreement = other != oi.verdict
            last = tokens[-1] if tokens else None
            oi.stop_token_agrees = (last == self.eos_id) if oi.verdict else (last == self.im_start_tok)
            if last not in (self.eos_id, self.im_start_tok):
                oi.mismatch = "cap"
            elif not oi.stop_token_agrees:
                oi.mismatch = "looks_good_im_start" if oi.verdict else "correction_eos"
            t.seq.extend(tokens)
            t.kinds.extend([KIND_TXT] * len(tokens))
            if oi.verdict:
                t.stop_reason = "looks_good"
                continue
            t.stop_reason = "refined"
            if last == self.im_start_tok:
                oi.refine_context = oi.reflect_context + oi.reflection + self.image_splice
                oi.packed = True
            else:
                base = oi.reflect_context + (oi.reflection[:-1] if last == self.eos_id else oi.reflection)
                oi.refine_context = base + self._forced_image_splice(base)
            pending.append(t)
        if pending:
            images = self._sample_images(model, [t.one_iter.refine_context for t in pending])
            for t, image in zip(pending, images):
                t.one_iter.refine = list(image)
                tail = t.one_iter.refine_context[len(t.one_iter.reflect_context) + len(t.one_iter.reflection):] \
                    if t.one_iter.packed else self.image_splice
                t.seq.extend(tail)
                t.kinds.extend([KIND_NONE] * len(tail))
                t.seq.extend(image)
                t.kinds.extend([KIND_IMG] * len(image))
                t.images.append([token - self.img_start for token in image])
        return RolloutBatch(prompts, trajectories)

    def run_looks_good(self, model, prompts: List[Dict], num_rollouts: Optional[int] = None,
                       verdict_rule: str = "correction") -> RolloutBatch:
        """Evaluation only: a draft, then up to ``max_refinements`` reflect -> refine rounds.

        A round stops the episode when its reflection's verdict is "looks good"
        (``looks_good_verdict``); otherwise a refinement is forced exactly as in
        ``run_one_iteration`` (a sampled <im_start> keeps its context; a trailing
        EOS is dropped and ``<im_start><S0>`` spliced, newline-aware). EOS alone
        never stops. Reset context rebuilds every round from the prompt prefix and
        the latest image, i.e. the one-iteration draft slot. Each sampled context
        must fit ``max_seq_len``; the logical episode is not budgeted.
        """
        count = self.cfg.num_rollouts if num_rollouts is None else num_rollouts
        if count < 1:
            raise ValueError("num_rollouts must be positive.")
        if self.cfg.max_refinements < 0:
            raise ValueError("max_refinements must be nonnegative.")
        looks_good_verdict("", verdict_rule)  # validate the rule name
        limit = min(self.cfg.max_seq_len,
                    getattr(model.config, "max_position_embeddings", self.cfg.max_seq_len))
        was_training = model.training
        model.eval()
        try:
            return self._run_looks_good(model, prompts, count, limit, verdict_rule)
        finally:
            model.train(was_training)

    def _run_looks_good(self, model, prompts, count, limit, verdict_rule):
        reset = self.cfg.reset_context_per_turn
        trajectories = []
        for p_idx, prompt in enumerate(prompts):
            prefix = self.make_prefix(prompt["prompt"])
            if len(prefix) + 2 * self.cfg.gen_seq_len + len(self.reflect_splice) + self.cfg.reflect_tokens + 3 > limit:
                raise ValueError("Sequence budget cannot fit a draft, a reflection and one refinement.")
            for _ in range(count):
                trajectories.append(Trajectory(p_idx, list(prefix), [KIND_NONE] * len(prefix)))
                if reset:
                    trajectories[-1].segments.append(RolloutSegment(list(prefix), [KIND_NONE] * len(prefix)))

        def context(t):
            return t.segments[-1].seq if reset else t.seq

        def append(t, tokens, kind):
            t.seq.extend(tokens)
            t.kinds.extend([kind] * len(tokens))
            if reset:
                t.segments[-1].seq.extend(tokens)
                t.segments[-1].kinds.extend([kind] * len(tokens))

        def append_images(pending):
            for t, image in zip(pending, self._sample_images(model, [context(t) for t in pending])):
                append(t, image, KIND_IMG)
                t.images.append([token - self.img_start for token in image])

        if not trajectories:
            return RolloutBatch(prompts, trajectories)
        append_images(trajectories)
        active = trajectories
        while active:
            ready = []
            for t in active:
                if reset and len(t.images) > 1:
                    current = self.make_prefix(prompts[t.prompt_idx]["prompt"]) + [
                        code + self.img_start for code in t.codes]
                    t.segments.append(RolloutSegment(current, [KIND_NONE] * len(current)))
                if len(context(t)) + len(self.reflect_splice) + self.cfg.reflect_tokens > limit:
                    t.stop_reason = "max_seq_len"
                    continue
                append(t, self.reflect_splice, KIND_NONE)
                ready.append(t)
            pending = []
            if ready:
                reflections = self._sample_reflections(model, [context(t) for t in ready], self.cfg.reflect_tokens)
            for t, tokens in zip(ready, reflections if ready else []):
                text = self.tok.decode(tokens, skip_special_tokens=True).strip()
                t.reflections.append(text)
                verdict = looks_good_verdict(text, verdict_rule)[0]
                t.verdicts.append(verdict)
                last = tokens[-1] if tokens else None
                if verdict:
                    append(t, tokens, KIND_TXT)
                    t.stop_reason = "looks_good"
                    continue
                if len(t.images) - 1 >= self.cfg.max_refinements:
                    append(t, tokens, KIND_TXT)
                    t.stop_reason = "max_refinements"
                    continue
                if last == self.im_start_tok:
                    append(t, tokens, KIND_TXT)
                    splice = list(self.image_splice)
                else:
                    append(t, tokens[:-1] if last == self.eos_id else tokens, KIND_TXT)
                    splice = self._forced_image_splice(context(t))
                if len(context(t)) + len(splice) + self.cfg.gen_seq_len > limit:
                    t.stop_reason = "max_seq_len"
                    continue
                append(t, splice, KIND_NONE)
                pending.append(t)
            if pending:
                append_images(pending)
            active = pending
        return RolloutBatch(prompts, trajectories)

    def _pg_mask(self, input_ids, pos_kind, text_policy_gradient: str):
        """``transitions``: reflection prose keeps only its KL term; sampled EOS,
        <im_start> and image tokens keep their policy gradient."""
        if text_policy_gradient == "all":
            return None
        if text_policy_gradient != "transitions":
            raise ValueError("text_policy_gradient must be all or transitions")
        transition = (input_ids == self.eos_id) | (input_ids == self.im_start_tok)
        prose = (pos_kind == KIND_TXT) & ~transition
        return (~prose).float()

    def _check_row_args(self, reflect_token_weight, length_normalization):
        if reflect_token_weight <= 0:
            raise ValueError("Need a positive reflection token weight.")
        if length_normalization not in ("episode", "constant"):
            raise ValueError("length_normalization must be episode or constant")
        if length_normalization == "constant" and self.cfg.max_seq_len <= 0:
            raise ValueError("Constant length normalization requires positive max_seq_len.")

    def build_training_rows(self, trajectories: List[Trajectory], reflect_token_weight: float,
                            length_normalization: str = "episode",
                            text_policy_gradient: str = "all"):
        """Every sampled token receives its episode's final-outcome advantage.

        Prompt/splice/padding and copied image positions are excluded. Episode
        mode makes each episode's weights sum to one across all segments.
        Constant mode divides by configured
        max_seq_len regardless of sampled length (including capped episodes).
        Sampled EOS and <im_start> remain text actions in both modes.
        ``text_policy_gradient="transitions"`` adds a ``pg_mask`` row tensor.
        """
        if not trajectories:
            raise ValueError("Need trajectories and a positive reflection token weight.")
        self._check_row_args(reflect_token_weight, length_normalization)
        entries = []
        for trajectory in trajectories:
            if bool(trajectory.segments) != self.cfg.reset_context_per_turn:
                raise ValueError("Trajectory segments disagree with rollout context mode.")
            segments = trajectory.segments or [trajectory]
            normalizer = (sum(reflect_token_weight if k == KIND_TXT else 1.0
                              for k in trajectory.kinds if k != KIND_NONE)
                          if length_normalization == "episode" else self.cfg.max_seq_len)
            entries.extend((trajectory, segment, normalizer) for segment in segments)
        width = max(len(segment.seq) for _, segment, _ in entries)
        shape = (len(entries), width)
        input_ids = torch.full(shape, self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(shape, dtype=torch.long)
        pos_kind = torch.zeros(shape, dtype=torch.long)
        adv = torch.zeros(shape, dtype=torch.float32)
        weight = torch.zeros(shape, dtype=torch.float32)
        for i, (trajectory, segment, normalizer) in enumerate(entries):
            length = len(segment.seq)
            input_ids[i, :length] = torch.tensor(segment.seq)
            attention_mask[i, :length] = 1
            pos_kind[i, :length] = torch.tensor(segment.kinds)
            sampled = pos_kind[i] != KIND_NONE
            if sampled[0] or not sampled.any():
                raise ValueError("An episode needs a prompt and at least one sampled action.")
            adv[i, sampled] = trajectory.adv
            weight[i, pos_kind[i] == KIND_IMG] = 1.0
            weight[i, pos_kind[i] == KIND_TXT] = reflect_token_weight
            # Preserve the original floating-point normalization when unsplit.
            weight[i] /= (weight[i].sum() if length_normalization == "episode"
                          and not trajectory.segments else normalizer)
        rows = dict(input_ids=input_ids, attention_mask=attention_mask,
                    train_mask=pos_kind != KIND_NONE, pos_kind=pos_kind, adv=adv, weight=weight)
        pg_mask = self._pg_mask(input_ids, pos_kind, text_policy_gradient)
        if pg_mask is not None:
            rows["pg_mask"] = pg_mask
        return rows

    @staticmethod
    def _pack_tree_nodes(nodes: List[TreeNode]):
        """Rows ``(seq, kinds, owners)``: each node's context and actions, appended
        to its parent's row when it is the only child and its context extends that
        row. Sorted by length (stable: equal lengths keep creation order)."""
        rows, row_of = [], {}
        for node in nodes:  # creation order: parents first
            parent = node.parent
            r = row_of.get(parent) if parent is not None and len(parent.children) == 1 else None
            if r is not None and node.context[:len(rows[r][0])] == rows[r][0]:
                extra = node.context[len(rows[r][0]):]
            else:
                r, extra = len(rows), node.context
                rows.append(([], [], []))
            seq, kinds, owners = rows[r]
            seq.extend(extra)
            kinds.extend([KIND_NONE] * len(extra))
            owners.extend([None] * len(extra))
            seq.extend(node.actions)
            kinds.extend(node.action_kinds)
            owners.extend([node] * len(node.actions))
            row_of[node] = r
        rows.sort(key=lambda row: len(row[0]))
        return rows

    def build_teacher_tree_rows(self, batch: RolloutBatch, teacher_ce_coef: float, shadow_coef: float):
        """Training rows for ``--objective teacher_tree`` (loss divided by #drafts D).

        Tree rows (``_pack_tree_nodes``): image tokens get their node's tree
        advantage with the episode weights of ``build_tree_training_rows``, except
        that a leaf's normalizer counts only policy-sampled tokens. Teacher
        reflection tokens get no policy gradient and no KL (weight 0, pg_mask 0)
        and a cross-entropy weight that makes the CE term ``teacher_ce_coef`` x the
        mean over teacher reflections of their per-token NLL. Each shadow
        reflection gets its own row (context + tokens) with PG and KL weighted to
        ``shadow_coef`` x the mean over shadows of per-token means; a stop token
        that disagrees with the shadow's verdict keeps KL but no PG (as Janus).
        ``pos_segment``: 1 draft image, 2 shadow, 3 refined image, 4 teacher.
        """
        if not batch.nodes or not batch.trajectories:
            raise ValueError("Teacher-tree rows need a tree rollout batch.")
        if any(n.kind == "reflection" and not n.teacher for n in batch.nodes):
            raise ValueError("teacher_tree reflections must all come from the teacher.")
        if not (math.isfinite(teacher_ce_coef) and teacher_ce_coef >= 0
                and math.isfinite(shadow_coef) and shadow_coef >= 0):
            raise ValueError("teacher_ce_coef and shadow_coef must be finite and nonnegative.")
        drafts = sum(1 for n in batch.nodes if n.parent is None)
        teachers = sum(1 for n in batch.nodes if n.teacher)
        shadows = [s for n in batch.nodes for s in n.shadows if s.tokens]
        scale = {}
        for trajectory in batch.trajectories:
            normalizer = sum(len(n.actions) for n in trajectory.nodes if not n.teacher)
            for node in trajectory.nodes:
                if not node.teacher:
                    scale[node] = scale.get(node, 0.0) + trajectory.mass / normalizer
        entries = []  # (seq, kinds, adv, weight, ce_weight, pg_mask, segment)
        for seq, kinds, owners in self._pack_tree_nodes(batch.nodes):
            adv, weight, ce, pg, segment = [], [], [], [], []
            for owner in owners:
                if owner is None:
                    values = (0.0, 0.0, 0.0, 1.0, 0)
                elif owner.teacher:
                    values = (0.0, 0.0, teacher_ce_coef * drafts / (teachers * len(owner.actions)), 0.0,
                              SEGMENT_TEACHER)
                else:
                    values = (owner.adv, scale[owner], 0.0, 1.0,
                              SEGMENT_DRAFT if owner.parent is None else SEGMENT_FINAL)
                for column, value in zip((adv, weight, ce, pg, segment), values):
                    column.append(value)
            entries.append((seq, kinds, adv, weight, ce, pg, segment))
        for s in shadows:
            n = len(s.tokens)
            pg = [1.0] * n
            if not s.stop_token_agrees and not s.capped:
                pg[-1] = 0.0
            pad = [0.0] * len(s.context)
            entries.append((s.context + s.tokens, [KIND_NONE] * len(s.context) + [KIND_TXT] * n,
                            pad + [s.adv] * n, pad + [shadow_coef * drafts / (len(shadows) * n)] * n,
                            pad + [0.0] * n, [1.0] * len(s.context) + pg,
                            [0] * len(s.context) + [SEGMENT_REFLECTION] * n))
        entries.sort(key=lambda entry: len(entry[0]))  # stable
        shape = (len(entries), max(len(entry[0]) for entry in entries))
        out = dict(input_ids=torch.full(shape, self.pad_id, dtype=torch.long),
                   attention_mask=torch.zeros(shape, dtype=torch.long),
                   pos_kind=torch.zeros(shape, dtype=torch.long),
                   adv=torch.zeros(shape, dtype=torch.float32),
                   weight=torch.zeros(shape, dtype=torch.float32),
                   ce_weight=torch.zeros(shape, dtype=torch.float32),
                   pg_mask=torch.ones(shape, dtype=torch.float32),
                   pos_segment=torch.zeros(shape, dtype=torch.long))
        for i, (seq, kinds, adv, weight, ce, pg, segment) in enumerate(entries):
            if kinds[0] != KIND_NONE or all(k == KIND_NONE for k in kinds):
                raise ValueError("A teacher-tree row needs a context and at least one trained token.")
            n = len(seq)
            out["input_ids"][i, :n] = torch.tensor(seq)
            out["attention_mask"][i, :n] = 1
            out["pos_kind"][i, :n] = torch.tensor(kinds)
            out["adv"][i, :n] = torch.tensor(adv)
            out["weight"][i, :n] = torch.tensor(weight)
            out["ce_weight"][i, :n] = torch.tensor(ce)
            out["pg_mask"][i, :n] = torch.tensor(pg)
            out["pos_segment"][i, :n] = torch.tensor(segment)
        out["train_mask"] = out["pos_kind"] != KIND_NONE
        return out

    def build_tree_training_rows(self, batch: RolloutBatch, reflect_token_weight: float,
                                 length_normalization: str = "episode",
                                 text_policy_gradient: str = "all"):
        """One row per tree node (context + its actions), each action trained once.

        A node is appended to its parent's row when it is the parent's only
        child and its context extends that row, which reproduces the episode
        rows/segments when nothing branches. Token advantage is its node's.
        Episode weights: kind weight x sum over leaves L under the node of
        mass(L) / N_L, with N_L the kind-weight sum of L's sampled actions.
        Constant weights: kind weight x mass(node) / max_seq_len. Each draft's
        subtree therefore weighs like one episode, and the denominator is the
        number of drafts.
        """
        if not batch.nodes or not batch.trajectories:
            raise ValueError("Tree rows need a tree rollout batch.")
        self._check_row_args(reflect_token_weight, length_normalization)
        scale = {}
        if length_normalization == "episode":
            for trajectory in batch.trajectories:
                normalizer = sum(reflect_token_weight if k == KIND_TXT else 1.0
                                 for k in trajectory.kinds if k != KIND_NONE)
                for node in trajectory.nodes:
                    scale[node] = scale.get(node, 0.0) + trajectory.mass / normalizer
        else:
            scale = {node: node.mass / self.cfg.max_seq_len for node in batch.nodes}
        rows = self._pack_tree_nodes(batch.nodes)
        shape = (len(rows), max(len(row[0]) for row in rows))
        input_ids = torch.full(shape, self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(shape, dtype=torch.long)
        pos_kind = torch.zeros(shape, dtype=torch.long)
        adv = torch.zeros(shape, dtype=torch.float32)
        weight = torch.zeros(shape, dtype=torch.float32)
        for i, (seq, kinds, owners) in enumerate(rows):
            if kinds[0] != KIND_NONE or all(k == KIND_NONE for k in kinds):
                raise ValueError("A tree row needs a context and at least one sampled action.")
            input_ids[i, :len(seq)] = torch.tensor(seq)
            attention_mask[i, :len(seq)] = 1
            pos_kind[i, :len(seq)] = torch.tensor(kinds)
            for j, (kind, owner) in enumerate(zip(kinds, owners)):
                if owner is not None:
                    adv[i, j] = owner.adv
                    weight[i, j] = (reflect_token_weight if kind == KIND_TXT else 1.0) * scale[owner]
        out = dict(input_ids=input_ids, attention_mask=attention_mask,
                   train_mask=pos_kind != KIND_NONE, pos_kind=pos_kind, adv=adv, weight=weight)
        pg_mask = self._pg_mask(input_ids, pos_kind, text_policy_gradient)
        if pg_mask is not None:
            out["pg_mask"] = pg_mask
        return out

    def build_janus_rows(self, trajectories: List[Trajectory]):
        """Three-segment rows for one-iteration episodes (Janus-Pro-R1 loss).

        Segments: draft (adv_draft), reflection (adv_verdict), final (adv_final:
        the refined image, or the draft again when the verdict kept it). Each
        token weighs 1 / (#actions of its segment) / 3, so a batch loss summed
        and divided by #episodes is the mean over the three segment losses of
        per-sequence token means. Parts are packed into one row while each
        context extends the row. ``pg_mask`` drops the policy gradient of a
        stop token that disagrees with the verdict (its KL stays).
        """
        if not trajectories or any(t.one_iter is None for t in trajectories):
            raise ValueError("Janus rows need one-iteration trajectories.")
        rows = []
        for t in trajectories:
            oi = t.one_iter
            stop_pg = [1.0] * len(oi.reflection)
            if oi.reflection and not oi.stop_token_agrees and oi.mismatch != "cap":
                stop_pg[-1] = 0.0
            parts = [(oi.draft_context, oi.draft, KIND_IMG, oi.adv_draft, SEGMENT_DRAFT, None),
                     (oi.reflect_context, oi.reflection, KIND_TXT, oi.adv_verdict, SEGMENT_REFLECTION, stop_pg)]
            if oi.verdict:
                final = [(oi.draft_context, oi.draft, KIND_IMG, oi.adv_final, SEGMENT_FINAL, None)]
            else:
                final = [(oi.refine_context, oi.refine, KIND_IMG, oi.adv_final, SEGMENT_FINAL, None)]
            current = None
            for context, actions, kind, adv, segment, pg in parts + final:
                if not actions:
                    raise ValueError("Every segment needs at least one sampled action.")
                fresh = segment == SEGMENT_FINAL and oi.verdict
                if current is None or fresh or context[:len(current["seq"])] != current["seq"]:
                    current = dict(seq=[], kinds=[], adv=[], weight=[], segment=[], pg=[])
                    rows.append(current)
                extra = context[len(current["seq"]):]
                weight = 1.0 / len(actions) / 3.0
                current["seq"] += extra + list(actions)
                current["kinds"] += [KIND_NONE] * len(extra) + [kind] * len(actions)
                current["adv"] += [0.0] * len(extra) + [adv] * len(actions)
                current["weight"] += [0.0] * len(extra) + [weight] * len(actions)
                current["segment"] += [0] * len(extra) + [segment] * len(actions)
                current["pg"] += [1.0] * len(extra) + (pg or [1.0] * len(actions))
        rows.sort(key=lambda row: len(row["seq"]))  # stable
        shape = (len(rows), max(len(row["seq"]) for row in rows))
        out = dict(input_ids=torch.full(shape, self.pad_id, dtype=torch.long),
                   attention_mask=torch.zeros(shape, dtype=torch.long),
                   pos_kind=torch.zeros(shape, dtype=torch.long),
                   adv=torch.zeros(shape, dtype=torch.float32),
                   weight=torch.zeros(shape, dtype=torch.float32),
                   pos_segment=torch.zeros(shape, dtype=torch.long),
                   pg_mask=torch.ones(shape, dtype=torch.float32))
        for i, row in enumerate(rows):
            n = len(row["seq"])
            if row["kinds"][0] != KIND_NONE:
                raise ValueError("A Janus row needs a context before its first action.")
            out["input_ids"][i, :n] = torch.tensor(row["seq"])
            out["attention_mask"][i, :n] = 1
            out["pos_kind"][i, :n] = torch.tensor(row["kinds"])
            out["adv"][i, :n] = torch.tensor(row["adv"])
            out["weight"][i, :n] = torch.tensor(row["weight"])
            out["pos_segment"][i, :n] = torch.tensor(row["segment"])
            out["pg_mask"][i, :n] = torch.tensor(row["pg"])
        out["train_mask"] = out["pos_kind"] != KIND_NONE
        return out
