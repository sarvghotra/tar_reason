from dataclasses import dataclass

import torch
from transformers import LogitsProcessor


class ImageVocabLogitsProcessor(LogitsProcessor):
    """Restricts sampling to the contiguous range of image-token ids
    [image_start_id, image_start_id + num_image_tokens).
    """

    def __init__(self, image_start_id: int, num_image_tokens: int):
        super().__init__()
        self.start = int(image_start_id)
        self.end = int(image_start_id + num_image_tokens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.start : self.end] = 0.0
        return scores + mask


class NonImageLogitsProcessor(LogitsProcessor):
    """Inverse of ImageVocabLogitsProcessor: masks out the contiguous
    image-vocab range so free-text generation (the self-critique segment)
    can't accidentally sample an `<I...>` token that would corrupt the
    templated rollout structure.
    """

    def __init__(self, image_start_id: int, num_image_tokens: int):
        super().__init__()
        self.start = int(image_start_id)
        self.end = int(image_start_id + num_image_tokens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.zeros_like(scores)
        mask[:, self.start : self.end] = float("-inf")
        return scores + mask


def encode_splice(tokenizer, text: str, device: torch.device) -> torch.LongTensor:
    """Tokenize a fixed splice string (may contain registered special tokens)
    into a (S,) LongTensor on `device`. Used to inject deterministic structural
    tokens between rollout phases (e.g. `<im_end>\\nCritique: `).
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return torch.tensor(ids, dtype=torch.long, device=device)


def selective_log_softmax(logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
    """log p(targets) without materializing a full (B, L, V) log-softmax tensor.

    For Tar's V≈218K vocab, a naive `F.log_softmax(logits.float())` on a batch
    of B*G rollouts × ~800 tokens allocates multiple GB just to gather one
    index per position; even `torch.logsumexp` creates a `(B, L, V)` exp()
    intermediate.

    For high-precision inputs the batched path is cheap, so we take it.
    For bf16/fp16 we iterate row-by-row: peak memory is `(L, V)` instead of
    `(B, L, V)`, and we upcast to fp32 *inside* the loop so the sum keeps
    enough precision for PPO ratios to stay numerically equal to 1.0 on the
    first inner epoch (otherwise spurious clipping kicks in).
    """
    if logits.dtype in (torch.float32, torch.float64):
        target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return (target_logits - torch.logsumexp(logits, dim=-1)).float()

    rows = []
    for row_logits, row_targets in zip(logits, targets):
        lf = row_logits.float()
        lse = torch.logsumexp(lf, dim=-1)
        tl = lf.gather(-1, row_targets.unsqueeze(-1)).squeeze(-1)
        rows.append(tl - lse)
    return torch.stack(rows, dim=0)


def gather_token_log_probs(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    gen_start: int,
    gen_len: int,
) -> torch.Tensor:
    """Run `model` and return per-token log p(input_ids[t] | < t) for the
    contiguous generated span [gen_start, gen_start + gen_len).

    Sequences are expected to be left-padded (HF generate convention), so
    we pass explicit position_ids derived from the attention mask — the
    default arange would otherwise misalign RoPE positions for the padded
    prefix.
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = out.logits[:, gen_start - 1 : gen_start - 1 + gen_len, :]
    targets = input_ids[:, gen_start : gen_start + gen_len]
    return selective_log_softmax(logits, targets)


@dataclass
class GRPOConfig:
    clip_eps: float = 0.2
    kl_coef: float = 0.04
    advantage_eps: float = 1e-4


def group_relative_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    """rewards: (B*G,) — already arranged so that every contiguous block of
    `group_size` entries shares the same prompt. Returns (B*G,) normalized
    advantages with zero mean and unit std within each group.

    `group_size` must be >= 2 — torch.std of a single element is NaN with
    the default unbiased=True, and there's no within-group comparison to
    make for G=1 anyway.
    """
    assert group_size >= 2, f"group_size must be >= 2 for GRPO normalization (got {group_size})"
    N = rewards.shape[0]
    assert N % group_size == 0, f"reward count {N} not divisible by group size {group_size}"
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    adv = (grouped - mean) / (std + eps)
    return adv.view(N)


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    gen_mask: torch.Tensor,
    cfg: GRPOConfig,
):
    """All shapes (B*G, T) except advantages which is (B*G,)."""
    mask = gen_mask.float()
    mask_sum = mask.sum(dim=-1).clamp(min=1.0)

    ratio = (log_probs - old_log_probs).exp()
    adv = advantages.unsqueeze(-1)
    pg_1 = ratio * adv
    pg_2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    pg_tok = -torch.min(pg_1, pg_2)

    log_diff = ref_log_probs - log_probs
    kl_tok = torch.exp(log_diff) - log_diff - 1.0

    loss_tok = pg_tok + cfg.kl_coef * kl_tok
    per_sample = (loss_tok * mask).sum(dim=-1) / mask_sum
    loss = per_sample.mean()

    with torch.no_grad():
        clip_frac = (((ratio - 1.0).abs() > cfg.clip_eps).float() * mask).sum() / mask.sum().clamp(min=1.0)
        pg_loss_val = (pg_tok * mask).sum() / mask.sum().clamp(min=1.0)
        kl_val = (kl_tok * mask).sum() / mask.sum().clamp(min=1.0)

    metrics = {
        "pg_loss": pg_loss_val.detach(),
        "kl": kl_val.detach(),
        "clip_frac": clip_frac.detach(),
    }
    return loss, metrics


def build_t2i_prefix_ids(
    tokenizer,
    prompts,
    device: torch.device,
    scale: int = 0,
    system_prompt: str = "You are a helpful assistant.",
) -> tuple:
    """Tokenize each prompt into the canonical T2I generation prefix:

        <|im_start|>system\n{sys}<|im_end|>\n
        <|im_start|>user\n{prompt}<|im_end|>\n
        <|im_start|>assistant\n<im_start><S{scale}>

    Returns (input_ids (B, L), attention_mask (B, L), prompt_lens (B,)).
    Sequences are LEFT-padded (HF generate convention) so generation can
    continue from the same offset for every row in the batch.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    encoded = []
    for p in prompts:
        text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{p}<|im_end|>\n"
            f"<|im_start|>assistant\n<im_start><S{scale}>"
        )
        ids = tokenizer(text, add_special_tokens=False).input_ids
        encoded.append(torch.tensor(ids, dtype=torch.long, device=device))

    max_len = max(e.shape[0] for e in encoded)
    B = len(encoded)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    lens = torch.zeros(B, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        # Left-pad so the generation continues right after the assistant prefix
        # for every row in the batch.
        L = e.shape[0]
        input_ids[i, max_len - L :] = e
        attention_mask[i, max_len - L :] = 1
        lens[i] = L
    return input_ids, attention_mask, lens
