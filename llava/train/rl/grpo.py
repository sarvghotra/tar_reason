"""GRPO pieces: vocab-restricted log-probs, episode advantages, clipped loss.

Position kinds (shared with rollout.py): a sampled token was drawn either from
the image vocabulary (``KIND_IMG``: the contiguous ``<I0>..<I65535>`` range) or
from everything but the image vocabulary (``KIND_TXT``). The log-probs used for
training are normalised over the same restricted support the sampler used, so
the policy gradient is taken w.r.t. the distribution that was actually sampled.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

KIND_NONE = 0   # prompt / forced splice tokens: never trained
KIND_IMG = 1    # image token sampled from the image vocab
KIND_TXT = 2    # reflection token sampled from the non-image vocab

# Rows of hidden states pushed through lm_head at once; with the 217K vocab
# this bounds the fp32 logits buffer to ~0.9 GB.
LOGPROB_CHUNK = 1024


@dataclass
class GRPOConfig:
    clip_eps: float = 0.2
    kl_coef: float = 0.01
    adv_norm: str = "mean"       # "std": (r - mean) / (std + eps); "mean": r - mean
    adv_eps: float = 1e-4
    # "std" divides by max(std, adv_std_floor): a group whose rewards differ by
    # less than the floor (e.g. renderer noise) is not blown up to unit scale.
    adv_std_floor: float = 0.0
    reflect_token_weight: float = 1.0
    length_normalization: str = "episode"  # legacy episode mean | fixed max_seq_len
    kl_gradient_correction: bool = False

    def __post_init__(self):
        if self.length_normalization not in ("episode", "constant"):
            raise ValueError("length_normalization must be episode or constant")


# ---------------------------------------------------------------------------
# Log-probs
# ---------------------------------------------------------------------------

def _restricted_logprob(logits: torch.Tensor, targets: torch.Tensor,
                        kinds: torch.Tensor, img_start: int, img_end: int) -> torch.Tensor:
    """log p(target) under the per-row restricted support. logits: (N, V) fp32."""
    target_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lse_img = torch.logsumexp(logits[:, img_start:img_end], dim=-1)
    lse_before = torch.logsumexp(logits[:, :img_start], dim=-1)
    if img_end < logits.shape[-1]:
        lse_after = torch.logsumexp(logits[:, img_end:], dim=-1)
        lse_txt = torch.logaddexp(lse_before, lse_after)
    else:
        lse_txt = lse_before
    lse = torch.where(kinds == KIND_IMG, lse_img, lse_txt)
    return target_logit - lse


def _chunk_logprob(lm_head, hidden, targets, kinds, img_start, img_end):
    return _restricted_logprob(lm_head(hidden).float(), targets, kinds, img_start, img_end)


def masked_token_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          train_mask: torch.Tensor, pos_kind: torch.Tensor,
                          img_start: int, img_end: int) -> torch.Tensor:
    """Per-token log-probs of ``input_ids[t]`` for the positions where
    ``train_mask[t]`` is set, flattened in (row, position) order.

    ``model`` is the (peft-wrapped) Qwen2ForCausalLM; the decoder stack is run
    once and ``lm_head`` is applied only to the selected positions, in chunks
    under activation checkpointing, so no ``[B, T, V]`` logits tensor exists.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    hidden = base.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state

    # Position t-1 predicts token t.
    sel = train_mask[:, 1:]
    h = hidden[:, :-1][sel]
    targets = input_ids[:, 1:][sel]
    kinds = pos_kind[:, 1:][sel]
    if h.shape[0] == 0:
        return hidden.sum() * 0.0

    use_ckpt = torch.is_grad_enabled() and h.requires_grad
    pieces = []
    for hc, tc, kc in zip(h.split(LOGPROB_CHUNK), targets.split(LOGPROB_CHUNK),
                          kinds.split(LOGPROB_CHUNK)):
        if use_ckpt:
            pieces.append(torch.utils.checkpoint.checkpoint(
                _chunk_logprob, base.lm_head, hc, tc, kc, img_start, img_end,
                use_reentrant=False))
        else:
            pieces.append(_chunk_logprob(base.lm_head, hc, tc, kc, img_start, img_end))
    return torch.cat(pieces)


# ---------------------------------------------------------------------------
# Advantages
# ---------------------------------------------------------------------------

def normalize_groups(rewards: List[float], groups: List[int], cfg: GRPOConfig,
                     unbiased: bool = False) -> Tuple[List[float], int]:
    """Group-relative advantages. ``groups[i]`` is the group key of reward i.

    Returns (advantages, number_of_zero_variance_groups). A group whose
    rewards are all identical (or of size 1) gets advantage 0. ``unbiased``
    uses the sample std (as Janus-Pro-R1's ``Tensor.std()``) in std mode.
    """
    by_group: Dict[int, List[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    adv = [0.0] * len(rewards)
    zero_var = 0
    for members in by_group.values():
        vals = torch.tensor([rewards[i] for i in members], dtype=torch.float64)
        if len(members) < 2:
            zero_var += 1
            continue
        centered = vals - vals.mean()
        std = vals.std(unbiased=False)
        if float(std) < 1e-8:
            zero_var += 1
            continue
        if cfg.adv_norm == "std":
            scale = vals.std(unbiased=True) if unbiased else std
            if cfg.adv_std_floor > 0:
                scale = torch.clamp(scale, min=cfg.adv_std_floor)
            centered = centered / (scale + cfg.adv_eps)
        elif cfg.adv_norm != "mean":
            raise ValueError(f"Unknown adv_norm {cfg.adv_norm}")
        for i, a in zip(members, centered.tolist()):
            adv[i] = a
    return adv, zero_var


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

SEGMENT_NAMES = ((1, "draft"), (2, "reflection"), (3, "final"))


def grpo_loss(logp: torch.Tensor, old_logp: torch.Tensor, ref_logp: torch.Tensor,
              adv: torch.Tensor, weight: torch.Tensor, kinds: torch.Tensor,
              denom: float, cfg: GRPOConfig, pg_mask: Optional[torch.Tensor] = None,
              segments: Optional[torch.Tensor] = None, ce_weight: Optional[torch.Tensor] = None,
              segment_names=SEGMENT_NAMES):
    """Token-level clipped surrogate + KL(policy || ref) estimator.

    All inputs are flat (N,) over the trainable tokens of one micro-batch.
    ``denom`` is the number of episodes in the whole optimizer step. Token
    weights already include the selected episode or constant normalization.
    ``pg_mask`` (0/1) drops the policy-gradient term of masked tokens while
    keeping their KL term; None leaves every token's policy gradient.
    ``ce_weight`` (teacher_tree) adds ``ce_weight * -logp`` (supervised
    distillation of spliced teacher tokens, which should carry weight 0);
    None leaves the loss unchanged. ``segment_names`` labels ``segments``.
    """
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()
    pg1 = ratio * adv
    pg2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    pg = -torch.min(pg1, pg2)
    if pg_mask is not None:
        pg = pg * pg_mask

    d = ref_logp - logp
    kl = d.exp() - d - 1.0
    if cfg.kl_gradient_correction:
        # Do not detach or clip this ratio: its derivative is needed even when
        # the current policy equals the sampling policy and ratio == 1.
        kl = ratio * kl

    tok = weight * (pg + cfg.kl_coef * kl)
    teacher = None
    if ce_weight is not None:
        tok = tok + ce_weight * -logp
        teacher = ce_weight > 0
    loss = tok.sum() / denom

    with torch.no_grad():
        w = weight
        # A segment can contain only part of an episode's normalized weight.
        wsum = w.sum().clamp(min=torch.finfo(w.dtype).tiny)
        img = kinds == KIND_IMG
        txt = kinds == KIND_TXT
        if teacher is not None:
            txt = txt & ~teacher  # text phase metrics cover policy-sampled tokens only
        # Phase PG means cover only tokens that are in the policy-gradient loss.
        pg_img = img if pg_mask is None else img & (pg_mask > 0)
        pg_txt = txt if pg_mask is None else txt & (pg_mask > 0)
        metrics = {
            "pg_loss": float((w * pg).sum() / wsum),
            "kl": float((w * kl).sum() / wsum),
            "clip_frac": float((w * ((ratio - 1.0).abs() > cfg.clip_eps)).sum() / wsum),
            "approx_kl_old": float((w * (-log_ratio)).sum() / wsum),
            "pg_img": float((pg[pg_img]).mean()) if pg_img.any() else 0.0,
            "pg_txt": float((pg[pg_txt]).mean()) if pg_txt.any() else 0.0,
            "kl_txt": float((kl[txt]).mean()) if txt.any() else 0.0,
            "kl_img": float((kl[img]).mean()) if img.any() else 0.0,
        }
        if segments is not None:
            # Weighted shares of the objective's PG / KL per segment (sum = pg_loss / kl).
            for k, name in segment_names:
                sel = (segments == k).to(w.dtype)
                metrics[f"pg_{name}"] = float((w * pg * sel).sum() / wsum)
                metrics[f"kl_{name}"] = float((w * kl * sel).sum() / wsum)
        if teacher is not None:
            # Plain sums: train_on_rollouts adds these across micro-batches.
            metrics["sum_ce_loss"] = float((ce_weight * -logp).sum())
            metrics["sum_teacher_nll"] = float((-logp)[teacher].sum())
            metrics["sum_teacher_tokens"] = float(teacher.sum())
    return loss, metrics
