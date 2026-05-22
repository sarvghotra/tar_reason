"""GRPO fine-tuning of Tar on text-to-image generation.

The policy is an autoregressive Qwen2 model whose vocabulary has been
extended with discrete latent image tokens (the `<I...>` series). For
each prompt batch we sample `G` completions per prompt using the policy,
score each completion with a frozen Tar reward model that estimates
log p(prompt | generated_image_tokens) entirely in latent space (no
pixel decoding), normalize advantages within each group, snapshot
`old_log_probs` and `ref_log_probs`, and run a real PPO inner loop
(`--num_ppo_epochs` updates) over the cached rollout — so the clipped
surrogate and the KL-to-reference actually do work.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LogitsProcessorList, Qwen2ForCausalLM

from llava.train.rl.dataset import build_dataset_from_yaml, collate_prompts
from llava.train.rl.grpo import (
    GRPOConfig,
    ImageVocabLogitsProcessor,
    NonImageLogitsProcessor,
    build_t2i_prefix_ids,
    encode_splice,
    gather_token_log_probs,
    grpo_loss,
    group_relative_advantages,
)
from llava.train.rl.reward import TarLatentReward, TarVQAReward


def parse_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--reward_model_path", type=str, default=None,
                   help="Frozen reward model checkpoint. Defaults to --model_name_or_path.")
    p.add_argument("--num_image_tokens", type=int, default=65536)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--attn_implementation", type=str, default="sdpa")
    # Data
    p.add_argument("--data_path", type=str, required=True,
                   help="YAML in scripts/data_demo.yaml format. Only the user prompt is used.")
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    # Optim
    p.add_argument("--per_device_prompt_batch_size", type=int, default=1)
    p.add_argument("--group_size", type=int, default=4, help="GRPO group size G — completions per prompt. Must be >= 2.")
    p.add_argument("--num_ppo_epochs", type=int, default=4,
                   help="PPO inner-loop epochs per rollout. Each epoch is an optimizer step over the same cached rollout.")
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=500,
                   help="Maximum number of *rollouts* (outer steps). Each performs --num_ppo_epochs optimizer updates.")
    p.add_argument("--gradient_clipping", type=float, default=1.0)
    # Generation
    p.add_argument("--gen_scale", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--gen_seq_len", type=int, default=729, help="Number of image tokens per rollout.")
    p.add_argument("--gen_temperature", type=float, default=1.0)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--gen_top_k", type=int, default=1200)
    p.add_argument("--system_prompt", type=str, default="You are a helpful assistant.",
                   help="System prompt for the T2I policy rollouts.")
    p.add_argument("--reward_system_prompt", type=str, default=None,
                   help="System prompt for the reward model's I2T scoring. Defaults to --system_prompt.")
    # Self-critique / self-correction (templated two-pass rollout)
    p.add_argument("--enable_self_critique", action="store_true",
                   help="Use the templated draft->critique->final rollout instead of a single "
                        "image draw. Reward is scored on the final image; GRPO gradient flows "
                        "through the critique and final-image tokens only.")
    p.add_argument("--critique_seq_len", type=int, default=64,
                   help="Number of free-text tokens sampled for the self-critique segment.")
    p.add_argument("--critique_lead", type=str, default="<im_end>\nCritique: ",
                   help="Fixed tokens spliced after the draft image to close the image block "
                        "and prompt the critique segment.")
    p.add_argument("--final_lead", type=str, default="\nFinal image:\n",
                   help="Fixed text spliced after the critique. The image-block opener "
                        "<im_start><S{gen_scale}> is appended automatically.")
    # GRPO
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.04)
    p.add_argument("--advantage_eps", type=float, default=1e-4)
    # Reward
    p.add_argument("--reward_type", type=str, default="caption", choices=["caption", "vqa"],
                   help="caption: whole-caption likelihood log p(prompt | image). "
                        "vqa: mean calibrated log p(yes) over heuristic yes/no probes "
                        "decomposed from the prompt.")
    p.add_argument("--reward_question", type=str, default="Describe the image shortly.",
                   help="I2T question for --reward_type caption. Unused for vqa.")
    p.add_argument("--reward_max_prompt_tokens", type=int, default=128)
    p.add_argument("--reward_normalize", action=argparse.BooleanOptionalAction, default=True,
                   help="caption reward only: if set, reward = mean log p per token; else sum. "
                        "Disable with --no-reward-normalize.")
    p.add_argument("--vqa_max_questions", type=int, default=4,
                   help="--reward_type vqa: max probes per prompt (1 holistic + up to N-1 "
                        "clause probes).")
    # IO
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Auto-resume from latest DeepSpeed checkpoint in --output_dir if present.")
    # Misc
    p.add_argument("--deepspeed_config", type=str, required=True)
    p.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def rank0_print(*a, **kw):
    if is_main_process():
        print(*a, **kw, flush=True)


def load_ds_config(path: str, args) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    # Each PPO inner-epoch is its own optimizer step over a single rollout, so
    # gradient accumulation must be 1 — otherwise inner-epoch backwards would
    # accumulate without actually stepping.
    cfg["train_micro_batch_size_per_gpu"] = args.per_device_prompt_batch_size * args.group_size
    cfg["gradient_accumulation_steps"] = 1
    cfg["train_batch_size"] = (
        cfg["train_micro_batch_size_per_gpu"] * int(os.environ.get("WORLD_SIZE", 1))
    )
    cfg["gradient_clipping"] = args.gradient_clipping
    if "optimizer" in cfg and isinstance(cfg["optimizer"].get("params", {}), dict):
        cfg["optimizer"]["params"]["lr"] = args.learning_rate
        cfg["optimizer"]["params"]["weight_decay"] = args.weight_decay
    cfg.setdefault("bf16", {})["enabled"] = bool(args.bf16)
    if args.bf16 and "fp16" in cfg:
        cfg["fp16"]["enabled"] = False
    # Each outer step runs num_ppo_epochs optimizer updates, so the LR schedule
    # ticks num_ppo_epochs times per rollout.
    total_updates = max(args.max_steps * args.num_ppo_epochs, 1)
    warmup_updates = int(args.warmup_ratio * total_updates)
    if warmup_updates > 0:
        cfg["scheduler"] = {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": total_updates,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": warmup_updates,
                "warmup_type": "linear",
            },
        }
    return cfg


def assert_special_tokens(tokenizer, image_start_token_id: int):
    """Catch silent tokenizer regressions: the multi-character special-token
    strings we splice into prompts MUST round-trip to single registered IDs.
    Falls back to an exception with a clear message if any token gets split
    into multiple BPE pieces (would corrupt every rollout's prefix).
    """
    checks = {
        "<im_start>": tokenizer.convert_tokens_to_ids("<im_start>"),
        "<im_end>": tokenizer.convert_tokens_to_ids("<im_end>"),
        "<S0>": tokenizer.convert_tokens_to_ids("<S0>"),
        "<I0>": image_start_token_id,
        "<|im_start|>": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "<|im_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }
    for tok, expected in checks.items():
        if expected is None or expected == tokenizer.unk_token_id:
            raise RuntimeError(f"Special token {tok!r} is not registered in the tokenizer.")
        actual = tokenizer(tok, add_special_tokens=False).input_ids
        if actual != [expected]:
            raise RuntimeError(
                f"Special token {tok!r} tokenizes to {actual} (expected [{expected}]). "
                "The tokenizer is splitting it into BPE pieces — every rollout's prefix "
                "would be silently corrupted."
            )


def save_hf_model(engine, tokenizer, output_dir: str):
    """Save model weights in HF format (loadable by Qwen2ForCausalLM.from_pretrained)."""
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        engine.module.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
    if dist.is_initialized():
        dist.barrier()


def save_ds_checkpoint(engine, output_dir: str, step: int):
    """DeepSpeed checkpoint with optimizer/scheduler state — resumable."""
    tag = f"checkpoint-{step}"
    engine.save_checkpoint(output_dir, tag=tag, client_state={"step": step})


def load_ds_checkpoint(engine, output_dir: str) -> int:
    """Resume from the latest DeepSpeed checkpoint under output_dir.

    Returns the step to start from (0 if nothing to resume).
    """
    if not os.path.isdir(output_dir):
        return 0
    tags = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            try:
                tags.append((int(name.split("-")[1]), name))
            except ValueError:
                continue
    if not tags:
        return 0
    tags.sort()
    latest_tag = tags[-1][1]
    rank0_print(f"Resuming from DeepSpeed checkpoint {latest_tag}")
    _, client_state = engine.load_checkpoint(
        output_dir,
        tag=latest_tag,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    )
    if client_state is None:
        return tags[-1][0]
    return int(client_state.get("step", tags[-1][0]))


def prune_old_checkpoints(output_dir: str, save_total_limit: Optional[int]):
    if not save_total_limit or not is_main_process():
        return
    cks = []
    for n in os.listdir(output_dir):
        if n.startswith("checkpoint-"):
            try:
                cks.append((int(n.split("-")[1]), n))
            except ValueError:
                continue
    cks.sort()
    for _, name in cks[:-save_total_limit]:
        import shutil
        shutil.rmtree(os.path.join(output_dir, name), ignore_errors=True)


@dataclass
class StepMetrics:
    loss: float
    reward_mean: float
    reward_std: float
    pg_loss: float
    kl: float
    clip_frac: float
    adv_mean: float
    approx_kl_old: float


def all_reduce_mean(value: float) -> float:
    """Average a scalar across ranks for honest distributed logging."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=torch.cuda.current_device())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


@torch.no_grad()
def _sample_phase(engine, tokenizer, input_ids, attention_mask, n_tokens, args, logits_processor):
    """One fixed-length generation phase: appends exactly `n_tokens` sampled
    tokens to `input_ids`. min==max new tokens makes the length deterministic
    so every rollout in the batch stays rectangular.
    """
    return engine.module.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        min_new_tokens=n_tokens,
        max_new_tokens=n_tokens,
        do_sample=True,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        top_k=args.gen_top_k,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )


def _sample_critique_phase(
    engine, tokenizer, input_ids, attention_mask, n_tokens, args, logits_processor
):
    """Critique generation capped at `n_tokens` with early-EOS support.

    Unlike _sample_phase, the model may emit EOS before `n_tokens` — the
    output is always padded to `n_tokens` so downstream tensors stay
    rectangular. Returns (full_ids, full_attn_mask, crit_gen_mask):
        full_attn_mask  zeros out positions after the first EOS so Phase 3
                        does not attend to critique padding.
        crit_gen_mask   1 on tokens up to and including the first EOS (or all
                        n_tokens if no EOS), 0 on padding — used in gen_mask
                        to suppress gradient on pad tokens.
    """
    device = input_ids.device
    B = input_ids.shape[0]
    prefix_len = input_ids.shape[1]

    output = engine.module.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=n_tokens,
        do_sample=True,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        top_k=args.gen_top_k,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    # Pad to n_tokens if any sequence stopped at EOS before the budget.
    crit_tokens = output[:, prefix_len:]  # (B, actual_len <= n_tokens)
    actual_len = crit_tokens.shape[1]
    if actual_len < n_tokens:
        pad = torch.full(
            (B, n_tokens - actual_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        crit_tokens = torch.cat([crit_tokens, pad], dim=1)

    # Build mask: 1 up to and including first EOS, 0 on padding after it.
    # Works correctly when pad_token_id == eos_token_id because argmax finds
    # the *first* occurrence and has_eos guards the no-EOS case.
    eos_id = tokenizer.eos_token_id
    is_eos = crit_tokens == eos_id                               # (B, n_tokens)
    has_eos = is_eos.any(dim=1)                                  # (B,)
    first_eos_idx = is_eos.long().argmax(dim=1)                  # (B,)
    positions = torch.arange(n_tokens, device=device).unsqueeze(0).expand(B, n_tokens)
    cutoff = torch.where(has_eos, first_eos_idx + 1, torch.full_like(first_eos_idx, n_tokens))
    crit_mask = (positions < cutoff.unsqueeze(1)).long()         # (B, n_tokens)

    full_ids = torch.cat([input_ids, crit_tokens], dim=1)
    full_attn = torch.cat([attention_mask, crit_mask], dim=1)
    return full_ids, full_attn, crit_mask


@torch.no_grad()
def rollout(
    engine,
    tokenizer,
    prompts: List[str],
    args,
    image_start_token_id: int,
    image_proc,
    text_proc,
) -> tuple:
    """Sample G rollouts per prompt with the current policy.

    Default path: a single image draw of `--gen_seq_len` tokens.
    `--enable_self_critique`: a templated draft -> critique -> final rollout —
        Phase 1 draws a draft image (image-vocab masked),
        a fixed `--critique_lead` splice closes it and prompts a critique,
        Phase 2 draws `--critique_seq_len` free-text tokens (image-vocab
            masked OUT so the structure can't be corrupted),
        a fixed `--final_lead` splice opens the final image block,
        Phase 3 draws the final image (image-vocab masked).

    Returns (full_ids, full_mask, prefix_len, gen_len, gen_mask,
    final_image_ids, expanded_prompts):
        gen_mask        (B*G, gen_len) 1 on tokens that receive GRPO gradient —
                        every post-prefix token in the default path; only the
                        critique + final-image spans in the self-critique path
                        (draft image and splices are excluded).
        final_image_ids (B*G, D)       unshifted image-token indices of the
                        rewarded image (the final image in the critique path).
    """
    device = engine.device
    G = args.group_size
    expanded = [p for p in prompts for _ in range(G)]

    prefix_ids, prefix_mask, _ = build_t2i_prefix_ids(
        tokenizer, expanded, device, scale=args.gen_scale,
        system_prompt=args.system_prompt,
    )
    prefix_len = prefix_ids.shape[1]
    B = prefix_ids.shape[0]
    D = args.gen_seq_len

    def ones(width):
        return torch.ones(B, width, dtype=torch.long, device=device)

    was_training = engine.module.training
    engine.module.eval()
    try:
        if not args.enable_self_critique:
            full_ids = _sample_phase(engine, tokenizer, prefix_ids, prefix_mask, D, args, image_proc)
            gen_len = D
            gen_mask = ones(D)
            final_offset = 0
            full_mask = torch.cat([prefix_mask, ones(gen_len)], dim=1)
        else:
            splice1 = encode_splice(tokenizer, args.critique_lead, device)
            splice2 = encode_splice(
                tokenizer, f"{args.final_lead}<im_start><S{args.gen_scale}>", device,
            )
            S1, S2 = splice1.shape[0], splice2.shape[0]
            C = args.critique_seq_len
            splice1_b = splice1.unsqueeze(0).expand(B, S1)
            splice2_b = splice2.unsqueeze(0).expand(B, S2)

            # Phase 1: draft image.
            ids = _sample_phase(engine, tokenizer, prefix_ids, prefix_mask, D, args, image_proc)
            mask = torch.cat([prefix_mask, ones(D)], dim=1)
            # Splice 1 + Phase 2: free-text self-critique (EOS-aware).
            ids = torch.cat([ids, splice1_b], dim=1)
            mask = torch.cat([mask, ones(S1)], dim=1)
            ids, mask, crit_gen_mask = _sample_critique_phase(
                engine, tokenizer, ids, mask, C, args, text_proc,
            )
            # Splice 2 + Phase 3: final (corrected) image.
            ids = torch.cat([ids, splice2_b], dim=1)
            mask = torch.cat([mask, ones(S2)], dim=1)
            full_ids = _sample_phase(engine, tokenizer, ids, mask, D, args, image_proc)
            mask = torch.cat([mask, ones(D)], dim=1)

            gen_len = D + S1 + C + S2 + D
            # Gradient flows only through the critique span and the final
            # image — the draft image and the deterministic splices are masked
            # out (the splices are forced tokens, not policy decisions).
            # crit_gen_mask is 0 on any padding after an early EOS so filler
            # tokens do not receive gradient.
            gen_mask = torch.zeros(B, gen_len, dtype=torch.long, device=device)
            crit_start = D + S1
            final_offset = crit_start + C + S2
            gen_mask[:, crit_start : crit_start + C] = crit_gen_mask
            gen_mask[:, final_offset : final_offset + D] = 1
            # mask already encodes the correct attention for the full sequence
            # (0s at critique padding so Phase 3 does not attend to it).
            full_mask = mask
    finally:
        if was_training:
            engine.module.train()
    # The rewarded image: phase-3 tokens in the critique path, the whole
    # generation in the default path. Convert to unshifted vocab indices.
    fs = prefix_len + final_offset
    final_image_ids = full_ids[:, fs : fs + D] - image_start_token_id
    # Loud assertion (one CUDA sync per rollout — cheap) so a misconfigured
    # logits processor produces an error instead of silently corrupted training.
    mn = int(final_image_ids.min().item())
    mx = int(final_image_ids.max().item())
    if mn < 0 or mx >= args.num_image_tokens:
        raise RuntimeError(
            f"Rollout produced out-of-range image tokens (min={mn}, max={mx}, "
            f"num_image_tokens={args.num_image_tokens}). Logits processor likely misconfigured."
        )
    return full_ids, full_mask, prefix_len, gen_len, gen_mask, final_image_ids, expanded


def training_step(
    engine,
    ref_model,
    reward_model,
    tokenizer,
    prompts: List[str],
    args,
    image_start_token_id: int,
    image_proc,
    text_proc,
    grpo_cfg: GRPOConfig,
) -> StepMetrics:
    full_ids, full_mask, prefix_len, gen_len, gen_mask, final_image_ids, expanded_prompts = rollout(
        engine, tokenizer, prompts, args, image_start_token_id, image_proc, text_proc,
    )

    rewards = reward_model.compute_reward(expanded_prompts, final_image_ids)
    advantages = group_relative_advantages(rewards, args.group_size, eps=args.advantage_eps)

    # ref_log_probs is the KL anchor — frozen for the whole rollout.
    # old_log_probs is captured from epoch 0's forward (the policy hasn't
    # moved yet, so log_probs.detach() at that point IS the sampling
    # distribution). This saves one full forward pass per rollout vs. an
    # explicit no_grad snapshot, while staying exactly correct GRPO.
    with torch.no_grad():
        ref_log_probs = gather_token_log_probs(
            ref_model, full_ids, full_mask, prefix_len, gen_len,
        )

    gen_mask = gen_mask.float()
    old_log_probs = None
    last_loss = None
    last_metrics = None
    approx_kl_old = 0.0
    for _ in range(args.num_ppo_epochs):
        log_probs = gather_token_log_probs(
            engine.module, full_ids, full_mask, prefix_len, gen_len,
        )
        if old_log_probs is None:
            old_log_probs = log_probs.detach()
        loss, metrics = grpo_loss(
            log_probs, old_log_probs, ref_log_probs, advantages, gen_mask, grpo_cfg,
        )
        engine.backward(loss)
        engine.step()

        last_loss = float(loss.detach())
        last_metrics = metrics
        with torch.no_grad():
            approx_kl_old = float((old_log_probs - log_probs).mean())

    return StepMetrics(
        loss=last_loss,
        reward_mean=float(rewards.mean()),
        reward_std=float(rewards.std()) if rewards.numel() > 1 else 0.0,
        pg_loss=float(last_metrics["pg_loss"]),
        kl=float(last_metrics["kl"]),
        clip_frac=float(last_metrics["clip_frac"]),
        adv_mean=float(advantages.mean()),
        approx_kl_old=approx_kl_old,
    )


def main():
    args = parse_args()
    assert args.group_size >= 2, "--group_size must be >= 2 for GRPO normalization."
    assert args.num_ppo_epochs >= 1, "--num_ppo_epochs must be >= 1."

    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_start_token_id = tokenizer.convert_tokens_to_ids("<I0>")
    if image_start_token_id is None or image_start_token_id == tokenizer.unk_token_id:
        raise RuntimeError("<I0> token missing from tokenizer — model checkpoint must have image vocab.")
    assert_special_tokens(tokenizer, image_start_token_id)
    rank0_print(f"image_start_token_id={image_start_token_id}, num_image_tokens={args.num_image_tokens}")

    rank0_print(f"Loading policy from {args.model_name_or_path}")
    policy_dtype = torch.bfloat16 if args.bf16 else torch.float32
    policy = Qwen2ForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=policy_dtype,
        attn_implementation=args.attn_implementation,
    )

    ds_config = load_ds_config(args.deepspeed_config, args)
    engine, _, _, _ = deepspeed.initialize(
        model=policy,
        model_parameters=[p for p in policy.parameters() if p.requires_grad],
        config=ds_config,
    )

    rank0_print(f"Loading frozen reference policy from {args.model_name_or_path}")
    ref_model = Qwen2ForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=policy_dtype,
        attn_implementation=args.attn_implementation,
    ).to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    reward_path = args.reward_model_path or args.model_name_or_path
    share_ref = (reward_path == args.model_name_or_path)
    if share_ref:
        rank0_print("Sharing reference model as reward backbone (same checkpoint).")
        reward_backbone = ref_model
    else:
        rank0_print(f"Loading frozen reward model from {reward_path}")
        reward_backbone = reward_path
    if args.reward_type == "vqa":
        rank0_print(
            f"Reward: VQA-style — up to {args.vqa_max_questions} yes/no probes per prompt."
        )
        reward_model = TarVQAReward(
            model_or_path=reward_backbone,
            tokenizer=tokenizer,
            device=device,
            dtype=policy_dtype,
            attn_impl=args.attn_implementation,
            system_prompt=args.reward_system_prompt or args.system_prompt,
            max_prompt_tokens=args.reward_max_prompt_tokens,
            max_questions=args.vqa_max_questions,
        )
    else:
        rank0_print("Reward: whole-caption likelihood log p(prompt | image).")
        reward_model = TarLatentReward(
            model_or_path=reward_backbone,
            tokenizer=tokenizer,
            device=device,
            dtype=policy_dtype,
            attn_impl=args.attn_implementation,
            reward_question=args.reward_question,
            system_prompt=args.reward_system_prompt or args.system_prompt,
            max_prompt_tokens=args.reward_max_prompt_tokens,
            normalize=args.reward_normalize,
        )

    image_proc = LogitsProcessorList(
        [ImageVocabLogitsProcessor(image_start_token_id, args.num_image_tokens)]
    )
    text_proc = LogitsProcessorList(
        [NonImageLogitsProcessor(image_start_token_id, args.num_image_tokens)]
    )
    if args.enable_self_critique:
        s1 = tokenizer(args.critique_lead, add_special_tokens=False).input_ids
        s2 = tokenizer(
            f"{args.final_lead}<im_start><S{args.gen_scale}>", add_special_tokens=False,
        ).input_ids
        rank0_print(
            f"Self-critique rollout enabled: draft({args.gen_seq_len}) -> "
            f"splice1({len(s1)} tok) -> critique({args.critique_seq_len}) -> "
            f"splice2({len(s2)} tok) -> final({args.gen_seq_len})"
        )
    grpo_cfg = GRPOConfig(
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        advantage_eps=args.advantage_eps,
    )

    # Resume from latest DeepSpeed checkpoint if available.
    start_step = 0
    if args.resume:
        try:
            start_step = load_ds_checkpoint(engine, args.output_dir)
            if start_step > 0:
                rank0_print(f"Resumed at step {start_step}")
        except Exception as e:
            rank0_print(f"Resume failed ({e}); starting from step 0.")
            start_step = 0

    dataset = build_dataset_from_yaml(args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_prompt_batch_size,
        collate_fn=collate_prompts,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    use_wandb = (args.report_to == "wandb") and is_main_process()
    if use_wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "tar_reasoning"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=args.run_name or os.environ.get("WANDB_NAME"),
            config=vars(args),
            resume="allow",
        )

    rank0_print(
        f"Starting GRPO training for {args.max_steps} rollouts × "
        f"{args.num_ppo_epochs} PPO epochs (start_step={start_step})"
    )
    step = start_step
    t0 = time.time()
    for batch in dataloader:
        if step >= args.max_steps:
            break
        prompts = batch["prompts"]
        m = training_step(
            engine, ref_model, reward_model, tokenizer, prompts, args,
            image_start_token_id, image_proc, text_proc, grpo_cfg,
        )

        if step % args.logging_steps == 0:
            # Honest distributed metrics — averages across all ranks.
            loss = all_reduce_mean(m.loss)
            reward_mean = all_reduce_mean(m.reward_mean)
            reward_std = all_reduce_mean(m.reward_std)
            pg_loss = all_reduce_mean(m.pg_loss)
            kl = all_reduce_mean(m.kl)
            clip_frac = all_reduce_mean(m.clip_frac)
            adv_mean = all_reduce_mean(m.adv_mean)
            approx_kl_old = all_reduce_mean(m.approx_kl_old)

            if is_main_process():
                dt = time.time() - t0
                rank0_print(
                    f"step={step} loss={loss:.4f} "
                    f"reward={reward_mean:+.3f}±{reward_std:.3f} "
                    f"adv={adv_mean:+.3f} kl={kl:.4f} "
                    f"kl_old≈{approx_kl_old:+.4f} clip_frac={clip_frac:.3f} "
                    f"pg={pg_loss:.4f} ({dt:.1f}s)"
                )
                t0 = time.time()
                if use_wandb:
                    import wandb
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/reward_mean": reward_mean,
                            "train/reward_std": reward_std,
                            "train/pg_loss": pg_loss,
                            "train/kl": kl,
                            "train/approx_kl_old": approx_kl_old,
                            "train/clip_frac": clip_frac,
                            "train/adv_mean": adv_mean,
                        },
                        step=step,
                    )

        if args.save_steps > 0 and step > 0 and (step % args.save_steps == 0):
            save_ds_checkpoint(engine, args.output_dir, step)
            prune_old_checkpoints(args.output_dir, args.save_total_limit)

        step += 1

    # Final: write both a resumable DS checkpoint and an HF-format model.
    save_ds_checkpoint(engine, args.output_dir, step)
    save_hf_model(engine, tokenizer, os.path.join(args.output_dir, "final"))
    rank0_print(f"Done. Final HF checkpoint at {os.path.join(args.output_dir, 'final')}")


if __name__ == "__main__":
    main()
