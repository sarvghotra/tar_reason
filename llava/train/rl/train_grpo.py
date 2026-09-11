"""Final-image GRPO for independent, EOS-terminated image/reflection episodes.

Each prompt gets G independent trajectories. Only the last complete image of
an episode is rendered and scored by frozen Qwen3-VL soft-TIFA AM. Its prompt-relative
advantage trains every sampled action in that episode, including EOS. The loss
averages tokens within episodes, then episodes within the batch. Safety-limit
truncations are reported separately and score their last complete image.

Only LoRA trains. Parameters are synchronized before optimizer creation and
manual gradient all-reduce keeps ranks aligned. Launch: scripts/rl_ft/bash.sh.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from typing import Dict, List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Qwen2ForCausalLM, get_cosine_schedule_with_warmup

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llava.train.rl.dataset import GenEval2PromptDataset
from llava.train.rl.grpo import GRPOConfig, grpo_loss, masked_token_logprobs, normalize_groups
from llava.train.rl.pixel_reward import QwenPixelReward
from llava.train.rl.rollout import Trajectory, RolloutBatch, RolloutConfig, EpisodeRollout


RL_OBJECTIVE = "final_pixel_qwen3vl_soft_tifa_am_v1"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0,
                   help="Keep 0 so the training forward matches the sampling distribution.")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    # Data
    p.add_argument("--data_path", required=True, help="YAML listing train .jsonl files")
    p.add_argument("--eval_data_path", default=None, help="YAML listing val .jsonl files")
    p.add_argument("--max_atoms", type=int, default=None)
    p.add_argument("--prompts_per_gpu", type=int, default=2)
    # Rollout
    p.add_argument("--num_rollouts", type=int, default=4, help="Independent episodes per prompt.")
    p.add_argument("--max_refinements", type=int, default=3, help="Safety cap on image corrections; EOS can stop earlier.")
    p.add_argument("--max_seq_len", type=int, default=4096, help="Total prompt + episode token budget.")
    p.add_argument("--scale", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--gen_seq_len", type=int, default=729)
    p.add_argument("--img_temperature", type=float, default=1.0)
    p.add_argument("--img_top_k", type=int, default=0)
    p.add_argument("--img_top_p", type=float, default=1.0)
    p.add_argument("--reflect_tokens", type=int, default=128)
    p.add_argument("--reflect_temperature", type=float, default=1.0)
    p.add_argument("--reflect_top_k", type=int, default=0)
    p.add_argument("--reflect_top_p", type=float, default=1.0)
    p.add_argument("--gen_batch_size", type=int, default=16)
    # Reward
    p.add_argument("--reward_model_name_or_path", default="Qwen/Qwen3-VL-8B-Instruct",
                   help="Local/cached official GenEval2 Qwen3-VL-8B-Instruct judge.")
    p.add_argument("--reward_python", default=sys.executable,
                   help="Python interpreter with Qwen3-VL support; may be a separate venv.")
    p.add_argument("--reward_timeout", type=float, default=1800)
    # GRPO
    p.add_argument("--adv_norm", default="mean", choices=["std", "mean"],
                   help="Final reward minus prompt-group mean; optionally divide by group std.")
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--reflect_token_weight", type=float, default=1.0)
    p.add_argument("--num_ppo_epochs", type=int, default=1)
    p.add_argument("--train_micro_batch", type=int, default=2)
    # Optim
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=500)
    # Eval / IO
    p.add_argument("--output_dir", required=True)
    p.add_argument("--eval_output_dir", default=None,
                   help="Directory for validation metric JSON files; defaults to output_dir/evaluations.")
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--eval_on_start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_max_prompts", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    p.add_argument("--run_name", default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    # Exemplar logging is optional; visual de-tokenizer weights are required for reward.
    p.add_argument("--log_images", type=int, default=0)
    p.add_argument("--ar_path", default=None)
    p.add_argument("--encoder_path", default=None)
    p.add_argument("--decoder_path", default=None)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    args = p.parse_args()
    try:
        RolloutConfig(num_rollouts=args.num_rollouts, max_refinements=args.max_refinements,
                      max_seq_len=args.max_seq_len, gen_seq_len=args.gen_seq_len,
                      reflect_tokens=args.reflect_tokens, gen_batch_size=args.gen_batch_size,
                      **{f"{phase}_{setting}": getattr(args, f"{phase}_{setting}")
                         for phase in ("img", "reflect")
                         for setting in ("temperature", "top_k", "top_p")}).validate_for_training()
        if min(args.prompts_per_gpu, args.train_micro_batch, args.num_ppo_epochs) <= 0:
            raise ValueError("Batch sizes and PPO epochs must be positive.")
        if not math.isfinite(args.reflect_token_weight) or args.reflect_token_weight <= 0:
            raise ValueError("Reflection token weight must be finite and positive, including EOS credit.")
        if not math.isfinite(args.reward_timeout) or args.reward_timeout <= 0:
            raise ValueError("Reward worker timeout must be finite and positive.")
        if args.lora_dropout != 0.0:
            raise ValueError("RL training requires --lora_dropout=0 so training matches sampling.")
    except ValueError as exc:
        p.error(str(exc))
    return args


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def rank0_print(*a, **k):
    if is_main():
        print(*a, **k, flush=True)


def reduce_sum(values: Dict[str, float], device) -> Dict[str, float]:
    keys = sorted(values)
    t = torch.tensor([float(values[k]) for k in keys], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return dict(zip(keys, t.tolist()))


def reduce_mean(values: Dict[str, float], device) -> Dict[str, float]:
    out = reduce_sum(values, device)
    ws = dist.get_world_size() if dist.is_initialized() else 1
    return {k: v / ws for k, v in out.items()}


@torch.no_grad()
def synchronize_trainable_parameters(model):
    """Gradient averaging assumes identical parameters before the first update."""
    if dist.is_initialized():
        for p in model.parameters():
            if p.requires_grad:
                dist.broadcast(p, src=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def find_all_linear_names(model, exclude=("embed_tokens", "lm_head")):
    names = set()
    for name, module in model.named_modules():
        if any(kw in name for kw in exclude):
            continue
        if isinstance(module, torch.nn.Linear):
            names.add(name.split(".")[-1])
    return sorted(names)


def load_policy(args, device, adapter_dir=None):
    from peft import LoraConfig, PeftModel, get_peft_model

    # torchrun processes start with independent RNG states. Initialize the
    # policy identically; the training loop separately seeds per-rank rollouts.
    torch.manual_seed(args.seed)
    base = Qwen2ForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation)
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(False)
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    else:
        target = find_all_linear_names(base)
        rank0_print(f"LoRA r={args.lora_r} alpha={args.lora_alpha} on {target}")
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=args.lora_dropout, bias="none",
                         target_modules=target, task_type="CAUSAL_LM")
        model = get_peft_model(base, cfg)
    if base.config.attention_dropout != 0.0 or any(
            cfg.lora_dropout != 0.0 for cfg in model.peft_config.values()):
        raise ValueError("RL policy attention and LoRA dropout must be zero so training matches sampling.")
    model.to(device)
    # Also covers resumed adapters. Do this before constructing the optimizer.
    synchronize_trainable_parameters(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Trainable params: {n_train / 1e6:.1f}M")
    return model


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    cks = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-") and os.path.isfile(os.path.join(output_dir, name, "state.json")):
            try:
                cks.append((int(name.split("-")[1]), name))
            except ValueError:
                pass
    if not cks:
        return None
    return os.path.join(output_dir, max(cks)[1])


def save_checkpoint(model, optimizer, scheduler, step, args, device):
    ck = os.path.join(args.output_dir, f"checkpoint-{step}")
    if is_main():
        os.makedirs(ck, exist_ok=True)
        model.save_pretrained(ck)
        torch.save(optimizer.state_dict(), os.path.join(ck, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(ck, "scheduler.pt"))
        with open(os.path.join(ck, "state.json"), "w") as f:
            json.dump({"step": step, "objective": RL_OBJECTIVE}, f)
        # Prune.
        cks = sorted(
            (int(n.split("-")[1]), n) for n in os.listdir(args.output_dir)
            if n.startswith("checkpoint-") and os.path.isfile(os.path.join(args.output_dir, n, "state.json")))
        for _, name in cks[:-args.save_total_limit] if args.save_total_limit else []:
            shutil.rmtree(os.path.join(args.output_dir, name), ignore_errors=True)
        rank0_print(f"Saved {ck}")
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Final image rewards and prompt-group episode advantages
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_rollouts(batch: RolloutBatch, reward, decoder):
    trajectories = batch.trajectories
    # Render once. Reuse these exact pixels in validation logs, since rendering
    # again could produce a different image under the stochastic second AR model.
    for trajectory in trajectories:
        trajectory.final_image = decoder.decode_codes([trajectory.codes])[0]
        trajectory.vqa_list = batch.prompts[trajectory.prompt_idx]["vqa_list"]
    if torch.cuda.is_available():
        # The judge lives in another process on this GPU and cannot reuse our
        # allocator's cached rollout/rendering buffers.
        torch.cuda.empty_cache()
    am, gm, per_question = reward.score_images(
        [t.final_image for t in trajectories],
        [batch.prompts[t.prompt_idx]["vqa_list"] for t in trajectories])
    for trajectory, a, g, scores in zip(trajectories, am, gm, per_question):
        trajectory.am, trajectory.gm = a, g
        trajectory.reward = a
        trajectory.question_scores = scores


def assign_advantages(batch: RolloutBatch, gcfg: GRPOConfig):
    keys = [t.prompt_idx for t in batch.trajectories]
    advantages, zero_var = normalize_groups([t.reward for t in batch.trajectories], keys, gcfg)
    for trajectory, advantage in zip(batch.trajectories, advantages):
        trajectory.adv = advantage
    return zero_var, len(set(keys))


def rollout_stats(batch: RolloutBatch, prefix: str) -> Dict[str, float]:
    trajectories = batch.trajectories
    stats = {
        "n_final": len(trajectories),
        "am_final": sum(t.am for t in trajectories),
        "gm_final": sum(t.gm for t in trajectories),
        "reward_final": sum(t.reward for t in trajectories),
        "mean_refinements": sum(len(t.images) - 1 for t in trajectories),
        "mean_sampled_tokens": sum(sum(k != 0 for k in t.kinds) for t in trajectories),
        "truncated_rate": sum(t.stop_reason != "eos" for t in trajectories),
    }
    for reason in ("eos", "max_refinements", "max_seq_len", "reflection_limit"):
        stats[f"{reason}_rate"] = sum(t.stop_reason == reason for t in trajectories)
    return {prefix + k: v for k, v in stats.items()}


def finalize_stats(s: Dict[str, float], prefix: str) -> Dict[str, float]:
    count = max(s[f"{prefix}n_final"], 1.0)
    return {prefix + name: s[prefix + name] / count for name in (
        "am_final", "gm_final", "reward_final", "mean_refinements", "mean_sampled_tokens",
        "truncated_rate", "eos_rate", "max_refinements_rate", "max_seq_len_rate", "reflection_limit_rate")}


def validate_resume_checkpoint(path):
    with open(os.path.join(path, "state.json")) as f:
        state = json.load(f)
    if state.get("objective") != RL_OBJECTIVE:
        raise ValueError("Cannot resume a checkpoint from the old local-reward or semantic-reward objective. "
                         "Use a new output directory/run name for pixel Qwen soft-TIFA AM GRPO.")
    return state


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_on_rollouts(model, rollout: EpisodeRollout, batch: RolloutBatch, args, gcfg: GRPOConfig,
                  optimizer, scheduler, device) -> Dict[str, float]:
    rollout.cfg.validate_for_training()
    img_start, img_end = rollout.img_start, rollout.img_end
    trajectories = sorted(batch.trajectories, key=lambda t: len(t.seq))
    micro = [trajectories[i:i + args.train_micro_batch]
             for i in range(0, len(trajectories), args.train_micro_batch)]
    rows = [rollout.build_training_rows(m, args.reflect_token_weight) for m in micro]
    denom = max(len(trajectories), 1)  # each row has unit total token weight
    n_tokens = int(sum(r["train_mask"].sum() for r in rows))

    # Reference log-probs (adapters off) are fixed for the whole step.
    model.eval()
    ref_logps = []
    with torch.no_grad(), model.disable_adapter():
        for r in rows:
            ref_logps.append(masked_token_logprobs(
                model, r["input_ids"].to(device), r["attention_mask"].to(device),
                r["train_mask"].to(device), r["pos_kind"].to(device), img_start, img_end))

    model.train()
    old_logps = [None] * len(rows)
    agg: Dict[str, float] = {}
    for epoch in range(args.num_ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        agg = {}
        for i, r in enumerate(rows):
            ids = r["input_ids"].to(device)
            am = r["attention_mask"].to(device)
            tm = r["train_mask"].to(device)
            pk = r["pos_kind"].to(device)
            logp = masked_token_logprobs(model, ids, am, tm, pk, img_start, img_end)
            if old_logps[i] is None:
                old_logps[i] = logp.detach()
            sel = tm[:, 1:]
            adv = r["adv"].to(device)[:, 1:][sel]
            weight = r["weight"].to(device)[:, 1:][sel]
            kinds = pk[:, 1:][sel]
            loss, metrics = grpo_loss(logp, old_logps[i], ref_logps[i], adv, weight, kinds, denom, gcfg)
            loss.backward()
            w = float(weight.sum())
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v * w
            agg["loss"] = agg.get("loss", 0.0) + float(loss)
        # Average LoRA grads across ranks.
        params = [p for p in model.parameters() if p.requires_grad]
        if dist.is_initialized():
            ws = dist.get_world_size()
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(ws)
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        agg["grad_norm"] = float(gnorm)
    optimizer.zero_grad(set_to_none=True)
    for k in list(agg):
        if k not in ("loss", "grad_norm"):
            agg[k] /= denom
    agg["train_tokens"] = n_tokens
    return agg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(model, rollout: EpisodeRollout, reward, val_rows, args, device, step, decoder):
    if not val_rows:
        return {}
    torch.manual_seed(args.seed * 7919 + 17)
    sums: Dict[str, float] = {}
    samples = []
    bs = max(args.gen_batch_size, 1)
    for start in range(0, len(val_rows), bs):
        # Same EOS-driven stochastic policy as training; one episode per prompt.
        batch = rollout.run(model, val_rows[start:start + bs], num_rollouts=1)
        score_rollouts(batch, reward, decoder)
        for k, v in rollout_stats(batch, "val/").items():
            sums[k] = sums.get(k, 0.0) + v
        if is_main() and len(samples) < args.log_images:
            samples.extend((batch.prompts[t.prompt_idx]["prompt"], t)
                           for t in batch.trajectories[:args.log_images - len(samples)])
    sums = reduce_sum(sums, device)
    out = finalize_stats(sums, "val/")
    if is_main() and samples:
        eval_dir = args.eval_output_dir or os.path.join(args.output_dir, "evaluations")
        images = save_visual_evaluation(decoder, samples, eval_dir, step, args.seed, out)
        if args.report_to == "wandb":
            import wandb
            out["val/images"] = [wandb.Image(path, caption=caption) for path, caption in images]
    return out


def save_visual_evaluation(decoder, samples, eval_dir, step, seed, metrics):
    """Persist exemplars independently of W&B; keep renderer noise fixed across steps."""
    directory = os.path.join(eval_dir, f"step-{step}")
    os.makedirs(directory, exist_ok=True)
    records, logged_images = [], []
    for index, (prompt, trajectory) in enumerate(samples):
        devices = [decoder.device] if decoder.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed * 7919 + index)
            pils = decoder.decode_images(trajectory)
        from PIL import Image
        w, h = pils[0].size
        canvas = Image.new("RGB", (w * len(pils), h))
        image_files = []
        for image_index, im in enumerate(pils):
            name = f"example-{index:03d}-image-{image_index:02d}.png"
            im.save(os.path.join(directory, name))
            image_files.append(name)
            canvas.paste(im, (image_index * w, 0))
        strip = f"example-{index:03d}-trajectory.png"
        canvas.save(os.path.join(directory, strip))
        caption = (f"Step {step} | {prompt} | draft → refinements (final image at right) | "
                   f"Final reward={trajectory.reward:.4f}; AM={trajectory.am:.4f}; "
                   f"stop={trajectory.stop_reason} | " + " | ".join(trajectory.reflections))
        logged_images.append((os.path.join(directory, strip), caption))
        records.append(dict(prompt=prompt, reward=trajectory.reward, am=trajectory.am,
                            gm=trajectory.gm, stop_reason=trajectory.stop_reason,
                            reflections=trajectory.reflections, semantic_codes=trajectory.images,
                            question_scores=trajectory.question_scores,
                            vqa_list=trajectory.vqa_list,
                            images=image_files, trajectory_image=strip))
    with open(os.path.join(directory, "evaluation.json"), "w") as f:
        json.dump(dict(step=step, objective=RL_OBJECTIVE, metrics=metrics, examples=records), f, indent=2)
    return logged_images


class ImageDecoder:
    """Frozen second AR model and VQ decoder, used for reward and logging."""

    def __init__(self, args, device):
        from tok.mm_autoencoder import MMAutoEncoder
        self.tok = MMAutoEncoder(
            ar_path=args.ar_path, encoder_path=args.encoder_path, decoder_path=args.decoder_path,
            encoder_args={"input_type": "rec"}, decoder_args={},
        ).eval().to(dtype=torch.bfloat16, device=device)
        self.tok.ar_model.cls_token_num = args.gen_seq_len
        self.tok.encoder.pool_scale = args.scale + 1
        self.device = device
        self.cfg_scale = args.cfg_scale

    @torch.inference_mode()
    def decode_codes(self, image_codes):
        from PIL import Image
        codes = torch.tensor(image_codes, dtype=torch.long, device=self.device)
        imgs = self.tok.decode_from_encoder_indices(codes, {"cfg_scale": self.cfg_scale})
        return [Image.fromarray(im.cpu().numpy()) for im in imgs]

    def decode_images(self, trajectory: Trajectory):
        if trajectory.final_image is None:
            raise ValueError("An evaluation exemplar must retain its scored final image.")
        previous = self.decode_codes(trajectory.images[:-1]) if len(trajectory.images) > 1 else []
        return previous + [trajectory.final_image]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    image_start_id = tokenizer.convert_tokens_to_ids("<I0>")
    num_image_tokens = sum(1 for t in tokenizer.get_vocab() if t.startswith("<I") and t[2:-1].isdigit())
    rank0_print(f"image vocab: start={image_start_id} n={num_image_tokens}")

    resume_dir = latest_checkpoint(args.output_dir) if args.resume else None
    resume_state = validate_resume_checkpoint(resume_dir) if resume_dir else None
    model = load_policy(args, device, adapter_dir=resume_dir)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)
    total_updates = args.max_steps * args.num_ppo_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_updates), total_updates)
    start_step = 0
    if resume_dir is not None:
        optimizer.load_state_dict(torch.load(os.path.join(resume_dir, "optimizer.pt"), map_location=device))
        scheduler.load_state_dict(torch.load(os.path.join(resume_dir, "scheduler.pt")))
        start_step = int(resume_state["step"])
        rank0_print(f"Resumed from {resume_dir} at step {start_step}")

    rcfg = RolloutConfig(
        num_rollouts=args.num_rollouts, max_refinements=args.max_refinements,
        max_seq_len=args.max_seq_len, scale=args.scale, gen_seq_len=args.gen_seq_len,
        img_temperature=args.img_temperature, img_top_k=args.img_top_k, img_top_p=args.img_top_p,
        reflect_tokens=args.reflect_tokens, reflect_sample=True,
        reflect_temperature=args.reflect_temperature, reflect_top_k=args.reflect_top_k,
        reflect_top_p=args.reflect_top_p, gen_batch_size=args.gen_batch_size)
    rollout = EpisodeRollout(tokenizer, rcfg, image_start_id, num_image_tokens, device)

    if not (args.ar_path and args.encoder_path and args.decoder_path):
        raise ValueError("Pixel reward requires all three visual decoder checkpoint paths, even with log_images=0.")
    decoder = ImageDecoder(args, device)
    reward = QwenPixelReward(args.reward_python, args.reward_model_name_or_path,
                             device, args.reward_timeout)
    gcfg = GRPOConfig(clip_eps=args.clip_eps, kl_coef=args.kl_coef, adv_norm=args.adv_norm,
                      reflect_token_weight=args.reflect_token_weight)

    train_ds = GenEval2PromptDataset(args.data_path, seed=args.seed, rank=rank,
                                     world_size=world_size, max_atoms=args.max_atoms)
    val_rows = []
    if args.eval_data_path:
        val_ds = GenEval2PromptDataset(args.eval_data_path, seed=args.seed, rank=rank, world_size=world_size)
        val_rows = val_ds.all_rows()
        if args.eval_max_prompts:
            val_rows = val_rows[:max(1, args.eval_max_prompts // world_size)]
    rank0_print(f"train prompts: {len(train_ds)} (per rank/epoch {train_ds.per_epoch()}), "
                f"val prompts per rank: {len(val_rows)}")

    use_wandb = args.report_to == "wandb" and is_main()
    if use_wandb:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "tar_reasoning"),
                   entity=os.environ.get("WANDB_ENTITY"),
                   name=args.run_name or os.environ.get("WANDB_NAME"),
                   config=vars(args), resume="allow",
                   id=os.environ.get("WANDB_RUN_ID"))

    def log(metrics: Dict, step: int, validation: bool = False):
        if is_main():
            printable = {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in metrics.items() if not k.endswith("images")}
            print(f"step={step} {json.dumps(printable)}", flush=True)
            if validation:
                eval_dir = args.eval_output_dir or os.path.join(args.output_dir, "evaluations")
                os.makedirs(eval_dir, exist_ok=True)
                with open(os.path.join(eval_dir, f"step-{step}.json"), "w") as f:
                    json.dump({"step": step, **printable}, f, indent=2)
            if use_wandb:
                import wandb
                wandb.log(metrics, step=step)

    if args.eval_on_start and start_step == 0 and val_rows:
        log(run_validation(model, rollout, reward, val_rows, args, device, 0, decoder), 0, validation=True)

    prompt_iter = train_ds.iterate(args.prompts_per_gpu, skip_batches=start_step)
    for step in range(start_step + 1, args.max_steps + 1):
        prompts = next(prompt_iter)
        torch.manual_seed(args.seed * 1_000_003 + step * 7919 + rank)
        t0 = time.time()
        batch = rollout.run(model, prompts)
        t1 = time.time()
        model.eval()
        score_rollouts(batch, reward, decoder)
        zero_var, n_groups = assign_advantages(batch, gcfg)
        t2 = time.time()
        train_metrics = train_on_rollouts(model, rollout, batch, args, gcfg, optimizer, scheduler, device)
        t3 = time.time()

        if step % args.logging_steps == 0:
            sums = rollout_stats(batch, "train/")
            sums["train/zero_var_groups"] = zero_var
            sums["train/groups"] = n_groups
            sums = reduce_sum(sums, device)
            metrics = finalize_stats(sums, "train/")
            metrics["train/frac_zero_adv_groups"] = sums["train/zero_var_groups"] / max(sums["train/groups"], 1.0)
            metrics.update({f"train/{k}": v for k, v in reduce_mean(train_metrics, device).items()})
            metrics.update({"time/rollout": t1 - t0, "time/reward": t2 - t1, "time/train": t3 - t2,
                            "train/lr": scheduler.get_last_lr()[0], "train/epoch_frac":
                            step * args.prompts_per_gpu / max(train_ds.per_epoch(), 1)})
            log(metrics, step)

        if args.save_steps and step % args.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, step, args, device)
        if args.eval_steps and val_rows and step % args.eval_steps == 0:
            log(run_validation(model, rollout, reward, val_rows, args, device, step, decoder), step, validation=True)

    if args.max_steps % max(args.save_steps, 1) != 0:
        save_checkpoint(model, optimizer, scheduler, args.max_steps, args, device)
    reward.close()
    rank0_print("Done.")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
