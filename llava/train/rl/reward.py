import re
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM

from llava.train.rl.grpo import selective_log_softmax


def _load_frozen(model_or_path, device, dtype, attn_impl):
    """Load (or accept) a Qwen2 model, move it to device, freeze it, eval()."""
    if isinstance(model_or_path, str):
        model = Qwen2ForCausalLM.from_pretrained(
            model_or_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        ).to(device)
    else:
        model = model_or_path
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class TarLatentReward(nn.Module):
    """Latent-space reward: log p_RM(prompt | generated_image_tokens)
    under a frozen Tar (Qwen2 + image vocab) model running in
    image-to-text understanding mode.

    The reward never decodes image tokens to pixels — it scores the
    likelihood the original caption is recovered directly from the
    generated discrete latent tokens.
    """

    def __init__(
        self,
        model_or_path,
        tokenizer,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        attn_impl: str = "sdpa",
        reward_question: str = "Describe the image shortly.",
        system_prompt: str = "You are a helpful assistant.",
        max_prompt_tokens: int = 128,
        normalize: bool = True,
    ):
        super().__init__()
        self.model = _load_frozen(model_or_path, device, dtype, attn_impl)

        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_prompt_tokens = max_prompt_tokens
        self.normalize = normalize

        i0 = tokenizer.convert_tokens_to_ids("<I0>")
        if i0 is None or i0 == tokenizer.unk_token_id:
            raise RuntimeError("Tokenizer is missing the <I0> image token.")
        self.image_start_token_id = i0

        prefix_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        suffix_text = (
            f"\n{reward_question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        self.prefix_ids = torch.tensor(
            tokenizer(prefix_text, add_special_tokens=False).input_ids,
            dtype=torch.long, device=device,
        )
        self.suffix_ids = torch.tensor(
            tokenizer(suffix_text, add_special_tokens=False).input_ids,
            dtype=torch.long, device=device,
        )
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.close_ids = torch.tensor([im_end_id], dtype=torch.long, device=device)

    @torch.no_grad()
    def compute_reward(
        self,
        prompts: List[str],
        image_token_ids: torch.LongTensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            prompts: list of N original text prompts.
            image_token_ids: (N, T_img) image-token indices in
                [0, num_image_tokens). These are *unshifted* indices, the
                same space the visual tokenizer produces.
            image_mask: (N, T_img) bool — True for valid image tokens.
                If None, all tokens are treated as valid.

        Returns:
            (N,) float reward — mean log p of the original prompt under the
            understanding-mode reward model.
        """
        N = len(prompts)
        assert image_token_ids.shape[0] == N
        device = self.device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        if image_mask is None:
            img_lens = torch.full((N,), image_token_ids.shape[1], dtype=torch.long, device=device)
        else:
            img_lens = image_mask.long().sum(dim=1)

        prompt_id_list = []
        for p in prompts:
            ids = self.tokenizer(p, add_special_tokens=False).input_ids[: self.max_prompt_tokens]
            prompt_id_list.append(torch.tensor(ids, dtype=torch.long, device=device))

        seqs = []
        prompt_starts = []
        prompt_lens = []
        prefix_len = self.prefix_ids.shape[0]
        suffix_len = self.suffix_ids.shape[0]
        close_len = self.close_ids.shape[0]
        for n in range(N):
            ln = int(img_lens[n].item())
            img_tok = image_token_ids[n, :ln].to(device).long() + self.image_start_token_id
            p_ids = prompt_id_list[n]
            seq = torch.cat([self.prefix_ids, img_tok, self.suffix_ids, p_ids, self.close_ids])
            seqs.append(seq)
            prompt_starts.append(prefix_len + ln + suffix_len)
            prompt_lens.append(p_ids.shape[0] + close_len)

        max_len = max(s.shape[0] for s in seqs)
        input_ids = torch.full((N, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((N, max_len), dtype=torch.long, device=device)
        for n, s in enumerate(seqs):
            input_ids[n, : s.shape[0]] = s
            attention_mask[n, : s.shape[0]] = 1

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        # Selective log-softmax — avoids materializing the full (N, L, V)
        # log-probs tensor (V ≈ 218K here, would cost several GB in fp32).
        # We shift left by 1 so position t's logit predicts input_ids[t].
        shifted_logits = out.logits[:, :-1, :]
        shifted_targets = input_ids[:, 1:]
        token_lp = selective_log_softmax(shifted_logits, shifted_targets)

        rewards = torch.empty(N, dtype=torch.float32, device=device)
        for n in range(N):
            # Token at position ps (the first prompt token) is predicted from
            # the logit at position ps-1; after the shift that maps to
            # token_lp[n, ps-1 : ps-1+pl].
            ps = prompt_starts[n]
            pl = prompt_lens[n]
            tok_lp = token_lp[n, ps - 1 : ps - 1 + pl]
            rewards[n] = tok_lp.mean() if self.normalize else tok_lp.sum()
        return rewards


def decompose_prompt(prompt: str, max_questions: int = 4) -> List[str]:
    """Heuristically split a T2I prompt into atomic yes/no probes.

    Returns a list of natural-language questions, each phrased so the
    expected answer for a faithful image is "yes". The first entry is
    always the holistic probe (the whole prompt); the rest are per-clause
    probes obtained by splitting on sentence / comma / semicolon and
    coordinating-conjunction ("and") boundaries.

    This is a crude, LLM-free decomposition — it concentrates the reward
    signal onto clause-level facts without an offline preprocessing pass.
    """
    prompt = prompt.strip()
    questions = [
        f'Does this image faithfully match the following description? "{prompt}"'
    ]
    # Split into clauses on sentence enders, commas, semicolons and " and ".
    parts = re.split(r"[.;,]|\band\b", prompt)
    seen = {prompt.lower()}
    for part in parts:
        clause = part.strip(" \"'")
        if len(clause.split()) < 2:  # drop bare articles / single words
            continue
        key = clause.lower()
        if key in seen:
            continue
        seen.add(key)
        questions.append(f'Does this image contain the following? "{clause}"')
        if len(questions) >= max_questions:
            break
    return questions


class TarVQAReward(nn.Module):
    """VQA-style reward.

    Instead of scoring the likelihood of recovering the whole caption, the
    frozen Tar model (image-to-text mode) answers a set of targeted yes/no
    probes derived from the prompt by `decompose_prompt`. The reward is the
    mean *calibrated* log-probability of "yes" across probes:

        score_q = log p(yes) - logsumexp(log p(yes), log p(no))
        reward  = mean_q score_q

    Reading the answer off the two yes/no tokens concentrates all reward
    variance onto the discriminative facts (counts, colors, relations),
    giving a much higher signal-to-noise ratio than a diffuse whole-caption
    likelihood — every other token's probability is ignored.

    Like `TarLatentReward`, the reward never decodes image tokens to pixels;
    the probes are scored directly on the discrete latent image tokens.
    """

    def __init__(
        self,
        model_or_path,
        tokenizer,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        attn_impl: str = "sdpa",
        system_prompt: str = "You are a helpful assistant.",
        max_prompt_tokens: int = 128,
        max_questions: int = 4,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.model = _load_frozen(model_or_path, device, dtype, attn_impl)

        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_prompt_tokens = max_prompt_tokens
        self.max_questions = max_questions
        self.chunk_size = chunk_size

        i0 = tokenizer.convert_tokens_to_ids("<I0>")
        if i0 is None or i0 == tokenizer.unk_token_id:
            raise RuntimeError("Tokenizer is missing the <I0> image token.")
        self.image_start_token_id = i0

        prefix_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        self.prefix_ids = torch.tensor(
            tokenizer(prefix_text, add_special_tokens=False).input_ids,
            dtype=torch.long, device=device,
        )

        # First-token ids of the answer words. The assistant answer is drawn
        # right after `assistant\n`, so the no-leading-space tokenization is
        # the one that matches generation. logsumexp over the casings keeps
        # the score robust to "Yes" vs "yes".
        def _first_ids(words):
            ids = set()
            for w in words:
                toks = tokenizer(w, add_special_tokens=False).input_ids
                if toks:
                    ids.add(toks[0])
            return sorted(ids)

        yes_ids = _first_ids(["Yes", "yes"])
        no_ids = _first_ids(["No", "no"])
        if not yes_ids or not no_ids:
            raise RuntimeError("Could not resolve yes/no answer token ids.")
        self.yes_ids = torch.tensor(yes_ids, dtype=torch.long, device=device)
        self.no_ids = torch.tensor(no_ids, dtype=torch.long, device=device)

    def _truncate(self, text: str) -> str:
        """Cap an over-long prompt so the probe text stays bounded."""
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) <= self.max_prompt_tokens:
            return text
        return self.tokenizer.decode(ids[: self.max_prompt_tokens])

    @torch.no_grad()
    def compute_reward(
        self,
        prompts: List[str],
        image_token_ids: torch.LongTensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            prompts: list of N original text prompts.
            image_token_ids: (N, T_img) unshifted image-token indices in
                [0, num_image_tokens).
            image_mask: (N, T_img) bool — True for valid image tokens.
                If None, all tokens are treated as valid.

        Returns:
            (N,) float reward — mean calibrated log p(yes) over each
            prompt's heuristic yes/no probes.
        """
        N = len(prompts)
        assert image_token_ids.shape[0] == N
        device = self.device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        if image_mask is None:
            img_lens = torch.full((N,), image_token_ids.shape[1], dtype=torch.long, device=device)
        else:
            img_lens = image_mask.long().sum(dim=1)

        # Build one (image, probe) row per question. `row_prompt[r]` maps
        # row r back to the prompt it scores.
        seqs: List[torch.Tensor] = []
        row_prompt: List[int] = []
        for n in range(N):
            ln = int(img_lens[n].item())
            img_tok = image_token_ids[n, :ln].to(device).long() + self.image_start_token_id
            for q in decompose_prompt(self._truncate(prompts[n]), self.max_questions):
                suffix_text = (
                    f"\n{q}\nAnswer with only \"yes\" or \"no\".<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                suffix_ids = torch.tensor(
                    self.tokenizer(suffix_text, add_special_tokens=False).input_ids,
                    dtype=torch.long, device=device,
                )
                seqs.append(torch.cat([self.prefix_ids, img_tok, suffix_ids]))
                row_prompt.append(n)

        R = len(seqs)
        yes_score = torch.empty(R, dtype=torch.float32, device=device)
        for start in range(0, R, self.chunk_size):
            chunk = seqs[start : start + self.chunk_size]
            B = len(chunk)
            max_len = max(s.shape[0] for s in chunk)
            input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
            last_idx = torch.empty(B, dtype=torch.long, device=device)
            for i, s in enumerate(chunk):
                input_ids[i, : s.shape[0]] = s
                attention_mask[i, : s.shape[0]] = 1
                last_idx[i] = s.shape[0] - 1  # right-padded: last real token

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            # The logit at the final real token predicts the assistant's
            # first answer token — exactly the yes/no decision.
            rows = torch.arange(B, device=device)
            ans_logits = out.logits[rows, last_idx].float()  # (B, V)
            yl = torch.logsumexp(ans_logits[:, self.yes_ids], dim=-1)
            nl = torch.logsumexp(ans_logits[:, self.no_ids], dim=-1)
            denom = torch.logsumexp(torch.stack([yl, nl], dim=-1), dim=-1)
            yes_score[start : start + B] = yl - denom  # log p(yes | {yes,no})

        # Mean calibrated log p(yes) over each prompt's probes.
        row_prompt_t = torch.tensor(row_prompt, dtype=torch.long, device=device)
        rewards = torch.zeros(N, dtype=torch.float32, device=device)
        counts = torch.zeros(N, dtype=torch.float32, device=device)
        rewards.index_add_(0, row_prompt_t, yes_score)
        counts.index_add_(0, row_prompt_t, torch.ones_like(yes_score))
        return rewards / counts.clamp(min=1.0)
