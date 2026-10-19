"""Final-image GRPO for independent, EOS-terminated image/reflection episodes.

Each prompt gets G independent trajectories. Only the last complete image of
an episode is rendered and scored by frozen Qwen3-VL soft-TIFA AM. Its prompt-relative
advantage trains every sampled action in that episode, including EOS. The loss
defaults to averaging tokens within episodes, then episodes within the batch;
constant token normalization and corrected KL gradients are opt-in. Safety-limit
truncations are reported separately and score their last complete image.

Opt-in ``--tree_branch`` samples episodes as a tree that shares earlier actions.
Rewards are still final images only; a node's value is the mean final reward of
its subtree and its advantage is that value minus its siblings' mean (a chain of
only children shares its branch point's advantage). Opt-in
``--text_policy_gradient transitions`` keeps reflection prose out of the policy
gradient (KL only) while EOS, <im_start> and image tokens stay trainable.

Opt-in ``--objective janus_one_iteration`` (Janus-Pro-R1 style): one draft, one
reflection, at most one refinement; the verdict is "Correction: looks good".
Rewards AM(draft), 1 - |AM(draft) - verdict| and AM(final) train the draft, the
reflection and the final image as three segments, each normalized per prompt
group with the unbiased std, losses averaged over the segments.

Opt-in ``--objective teacher_tree``: a tree of policy images whose reflections
are written by the judge model acting as a teacher (image + prompt only); a
teacher "looks good" stops early. Image tokens get final-reward tree credit
(an intermediate image is credited with the final rewards it led to). Teacher
tokens are distilled with cross-entropy (no policy gradient, no KL). After every
image the policy samples shadow reflections that are not continued; their
verdict correctness against the judge's all_pass target is the reward (prompt
groups, unbiased std).

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
from llava.train.rl.grpo import (SEGMENT_NAMES, GRPOConfig, grpo_loss, masked_token_logprobs,
                                 normalize_groups)
from llava.train.rl.pixel_reward import QwenPixelReward
from llava.train.rl.rollout import (Trajectory, RolloutBatch, RolloutConfig, EpisodeRollout,
                                    parse_tree_branch, validate_tree_branch)
from llava.train.rl.teacher import TEACHER_PROMPT_VERSION, teacher_instruction

OBJECTIVES = ("final_episode", "janus_one_iteration", "teacher_tree")
VERDICT_OBJECTIVES = ("janus_one_iteration", "teacher_tree")
TEACHER_TREE_SEGMENTS = ((1, "draft"), (2, "shadow"), (3, "refined"))
from llava.train.rl.training_lifecycle import checkpoint_directory, stop_requested
from llava.train.rl.geneval2_eval_io import atomic_json


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
    p.add_argument("--prompts_per_gpu", type=int, default=4)
    # Rollout
    p.add_argument("--num_rollouts", type=int, default=4, help="Independent episodes per prompt.")
    p.add_argument("--max_refinements", type=int, default=3, help="Safety cap on image corrections; EOS can stop earlier.")
    p.add_argument("--max_seq_len", type=int, default=4096, help="Total prompt + episode token budget.")
    p.add_argument("--reset_context_per_turn", action=argparse.BooleanOptionalAction, default=False,
                   help="Each refinement sees only the prompt and latest image; train all segments on final reward.")
    p.add_argument("--objective", choices=OBJECTIVES, default="final_episode",
                   help="final_episode: final-image reward on every sampled token. "
                        "janus_one_iteration: draft/verdict/final rewards on three segments "
                        "(requires --max_refinements 1). teacher_tree: teacher-written reflections "
                        "drive a tree of policy images; teacher tokens distilled, shadow reflections "
                        "rewarded for verdict correctness.")
    p.add_argument("--verdict_rule", choices=["correction", "anywhere"], default="correction",
                   help="janus_one_iteration / teacher_tree shadows: 'looks good' inside the "
                        "Correction clause, or anywhere.")
    p.add_argument("--verdict_target", choices=["all_pass", "soft_am"], default="all_pass",
                   help="janus_one_iteration verdict reward. all_pass: 1[verdict == every draft "
                        "question >= --verdict_pass_threshold]. soft_am: 1 - |AM(draft) - verdict| "
                        "(Janus formula; collapses to 'looks good' when AM > 0.5 is common).")
    p.add_argument("--verdict_pass_threshold", type=float, default=0.5)
    p.add_argument("--shadow_reflections", type=int, default=1,
                   help="teacher_tree: policy reflections sampled (not continued) after every image.")
    p.add_argument("--shadow_coef", type=float, default=1.0,
                   help="teacher_tree: weight of the mean shadow-reflection loss.")
    p.add_argument("--teacher_ce_coef", type=float, default=0.1,
                   help="teacher_tree: weight of the mean per-token NLL of teacher reflections.")
    p.add_argument("--teacher_max_tokens", type=int, default=96,
                   help="teacher_tree: new-token cap of a teacher reply (judge tokenizer).")
    p.add_argument("--teacher_batch_size", type=int, default=16,
                   help="teacher_tree: images per teacher generate call (throughput only).")
    p.add_argument("--render_batch_size", type=int, default=1,
                   help="Images per second-AR renderer call (rows sample independently).")
    p.add_argument("--reward_question_batch_size", type=int, default=1,
                   help="(image, question) rows per judge forward; 1 = per-question generate.")
    p.add_argument("--tree_branch", default="",
                   help="Tree rollouts: comma list 'R[xM]' per refinement round k (after image k): "
                        "R reflections per image, M images per reflection that requests one. "
                        "Unlisted rounds do not branch. Empty: independent episodes.")
    p.add_argument("--max_tree_leaves_per_prompt", type=int, default=64,
                   help="Refuse tree_branch specs whose worst-case leaves per prompt exceed this.")
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
    p.add_argument("--adv_std_floor", type=float, default=0.0,
                   help="With --adv_norm std, divide by max(group std, this) (renderer-noise floor).")
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--reflect_token_weight", type=float, default=1.0)
    p.add_argument("--text_policy_gradient", choices=["all", "transitions"], default="all",
                   help="transitions: reflection prose gets no policy gradient (KL kept); "
                        "EOS, <im_start> and image tokens keep it.")
    p.add_argument("--length_normalization", choices=["episode", "constant"], default="episode",
                   help="Legacy episode mean, or divide token losses by fixed max_seq_len.")
    p.add_argument("--kl_gradient_correction", action=argparse.BooleanOptionalAction, default=False,
                   help="Multiply K3 KL by the differentiable, unclipped policy/old-policy ratio.")
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
    p.add_argument("--stop_at_unix", type=float, default=0,
                   help="Save and stop at a completed-update boundary after this UNIX time; 0 disables.")
    p.add_argument("--stop_file", default=None,
                   help="Save and stop at an update boundary when this file exists.")
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
        if not math.isfinite(args.stop_at_unix) or args.stop_at_unix < 0:
            raise ValueError("stop_at_unix must be finite and nonnegative.")
        args.tree_branch = parse_tree_branch(args.tree_branch)
        validate_tree_branch(args.tree_branch, args.max_refinements, args.num_rollouts,
                             args.max_tree_leaves_per_prompt)
        if min(args.render_batch_size, args.reward_question_batch_size) <= 0:
            raise ValueError("Render and judge batch sizes must be positive.")
        if not math.isfinite(args.verdict_pass_threshold):
            raise ValueError("verdict_pass_threshold must be finite.")
        if not math.isfinite(args.adv_std_floor) or args.adv_std_floor < 0:
            raise ValueError("adv_std_floor must be finite and nonnegative.")
        if args.objective == "janus_one_iteration":
            if args.max_refinements != 1 or args.tree_branch:
                raise ValueError("janus_one_iteration requires --max_refinements 1 and no --tree_branch.")
            if args.text_policy_gradient != "all":
                raise ValueError("janus_one_iteration trains the verdict text; use --text_policy_gradient all.")
            if args.reflect_token_weight != 1.0 or args.length_normalization != "episode":
                raise ValueError("janus_one_iteration has its own segment weights; keep "
                                 "--reflect_token_weight 1 and --length_normalization episode.")
        if args.objective == "teacher_tree":
            if any(r != 1 for r, _ in args.tree_branch):
                raise ValueError("teacher_tree branches only over images: every --tree_branch entry needs R = 1.")
            if args.max_refinements < 1:
                raise ValueError("teacher_tree needs --max_refinements >= 1.")
            if args.text_policy_gradient != "all" or args.reflect_token_weight != 1.0 \
                    or args.length_normalization != "episode":
                raise ValueError("teacher_tree has its own token weights; keep --text_policy_gradient all, "
                                 "--reflect_token_weight 1 and --length_normalization episode.")
            if args.verdict_target != "all_pass":
                raise ValueError("teacher_tree rewards shadow verdicts against --verdict_target all_pass.")
            if args.shadow_reflections < 0 or min(args.teacher_max_tokens, args.teacher_batch_size) <= 0:
                raise ValueError("teacher_tree needs shadow_reflections >= 0 and positive teacher limits.")
            for name in ("shadow_coef", "teacher_ce_coef"):
                value = getattr(args, name)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and nonnegative.")
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
        started = time.monotonic()
        with checkpoint_directory(args.output_dir, step) as temporary:
            model.save_pretrained(temporary)
            torch.save(optimizer.state_dict(), os.path.join(temporary, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(temporary, "scheduler.pt"))
            atomic_json(os.path.join(temporary, "state.json"),
                        {"step": step, "objective": RL_OBJECTIVE,
                         "rl_method": rl_method_settings(args),
                         # Throughput settings (not resume-guarded): sampling noise only.
                         "reward_batching": dict(
                             render_batch_size=getattr(args, "render_batch_size", 1),
                             reward_question_batch_size=getattr(args, "reward_question_batch_size", 1))})
        # Prune.
        cks = sorted(
            (int(n.split("-")[1]), n) for n in os.listdir(args.output_dir)
            if n.startswith("checkpoint-") and os.path.isfile(os.path.join(args.output_dir, n, "state.json")))
        for _, name in cks[:-args.save_total_limit] if args.save_total_limit else []:
            shutil.rmtree(os.path.join(args.output_dir, name), ignore_errors=True)
        rank0_print(f"Saved {ck} in {time.monotonic() - started:.3f} seconds")
    if dist.is_initialized():
        dist.barrier()


def coordinated_stop(args, device):
    """All workers stop at the same optimizer boundary, based on rank zero."""
    requested = stop_requested(args.stop_at_unix, args.stop_file) if is_main() else False
    flag = torch.tensor(int(requested), device=device)
    if dist.is_initialized():
        dist.broadcast(flag, src=0)
    return bool(flag.item())


# ---------------------------------------------------------------------------
# Final image rewards and prompt-group episode advantages
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_rollouts(batch: RolloutBatch, reward, decoder, render_batch_size: int = 1):
    trajectories = batch.trajectories
    # Render once. Reuse these exact pixels in validation logs, since rendering
    # again could produce a different image under the stochastic second AR model.
    # Tree leaves ending on the same image node (e.g. siblings that stopped
    # with EOS) share that one render and its score.
    pending, keys = {}, []
    for trajectory in trajectories:
        node = trajectory.final_image_node
        key = node if node is not None else id(trajectory)
        if key not in pending:
            pending[key] = trajectory
        keys.append(key)
    images = render_codes(decoder, [t.codes for t in pending.values()], render_batch_size)
    rendered = {k: (image, t.prompt_idx) for (k, t), image in zip(pending.items(), images)}
    for trajectory, key in zip(trajectories, keys):
        trajectory.final_image = rendered[key][0]
        trajectory.vqa_list = batch.prompts[trajectory.prompt_idx]["vqa_list"]
    if torch.cuda.is_available():
        # The judge lives in another process on this GPU and cannot reuse our
        # allocator's cached rollout/rendering buffers.
        torch.cuda.empty_cache()
    unique = list(rendered)
    am, gm, per_question = reward.score_images(
        [rendered[k][0] for k in unique],
        [batch.prompts[rendered[k][1]]["vqa_list"] for k in unique])
    scores = {k: (a, g, q) for k, a, g, q in zip(unique, am, gm, per_question)}
    for trajectory, key in zip(trajectories, keys):
        a, g, q = scores[key]
        trajectory.am, trajectory.gm = a, g
        trajectory.reward = a
        trajectory.question_scores = list(q)


def assign_advantages(batch: RolloutBatch, gcfg: GRPOConfig):
    keys = [t.prompt_idx for t in batch.trajectories]
    advantages, zero_var = normalize_groups([t.reward for t in batch.trajectories], keys, gcfg)
    for trajectory, advantage in zip(batch.trajectories, advantages):
        trajectory.adv = advantage
    return zero_var, len(set(keys))


def assign_tree_advantages(batch: RolloutBatch, gcfg: GRPOConfig):
    """Final-reward credit over a rollout tree.

    value(leaf node) = its episode's final reward; value(node) = mean value of
    its children. adv(node) = value(node) - mean value of its siblings; a node
    that is its parent's only child uses the siblings of its nearest ancestor
    that has some (drafts' siblings are the prompt's drafts). Without branching
    this equals final reward minus the prompt mean. With adv_norm=std a prompt's
    advantages are divided by the mass-weighted std of its leaf rewards; a
    prompt whose leaf rewards are all equal gets zero advantages.
    """
    leaf_of = {t.nodes[-1]: t for t in batch.trajectories}
    for node in reversed(batch.nodes):  # children were created after parents
        node.value = (sum(c.value for c in node.children) / len(node.children)
                      if node.children else leaf_of[node].reward)
    drafts: Dict[int, List] = {}
    for node in batch.nodes:
        if node.parent is None:
            drafts.setdefault(node.prompt_idx, []).append(node)
    for node in batch.nodes:
        anchor = node
        while anchor.parent is not None and len(anchor.parent.children) == 1:
            anchor = anchor.parent
        siblings = drafts[anchor.prompt_idx] if anchor.parent is None else anchor.parent.children
        node.adv = (node.value - sum(s.value for s in siblings) / len(siblings)
                    if len(siblings) > 1 else 0.0)
    leaves: Dict[int, List[Trajectory]] = {}
    for trajectory in batch.trajectories:
        leaves.setdefault(trajectory.prompt_idx, []).append(trajectory)
    zero_var, factor = 0, {}
    for prompt_idx, group in leaves.items():
        total = sum(t.mass for t in group)
        mean = sum(t.mass * t.reward for t in group) / total
        std = math.sqrt(sum(t.mass * (t.reward - mean) ** 2 for t in group) / total)
        if len(group) < 2 or std < 1e-8:
            zero_var += 1
            factor[prompt_idx] = 0.0
        elif gcfg.adv_norm == "std":
            factor[prompt_idx] = 1.0 / (std + gcfg.adv_eps)
        elif gcfg.adv_norm == "mean":
            factor[prompt_idx] = 1.0
        else:
            raise ValueError(f"Unknown adv_norm {gcfg.adv_norm}")
    for node in batch.nodes:
        node.adv *= factor[node.prompt_idx]
    for trajectory in batch.trajectories:
        trajectory.adv = trajectory.nodes[-1].adv
    return zero_var, len(leaves)


def render_codes(decoder, codes_list, batch_size: int = 1):
    """Render in order, ``batch_size`` images per renderer call (rows sample independently)."""
    images = []
    for start in range(0, len(codes_list), batch_size):
        chunk = decoder.decode_codes(codes_list[start:start + batch_size])
        if len(chunk) != len(codes_list[start:start + batch_size]):
            raise RuntimeError("Renderer returned the wrong number of images.")
        images.extend(chunk)
    return images


def verdict_target(question_scores, threshold: float = 0.5) -> bool:
    """A draft "passes" when every question scores at least ``threshold``."""
    return all(q >= threshold for q in question_scores)


@torch.no_grad()
def score_one_iteration(batch: RolloutBatch, reward, decoder, render_batch_size: int = 1,
                        target: str = "soft_am", threshold: float = 0.5):
    """Render every draft and every refined image once and judge them in one call.
    A kept draft is the final image: the same pixels and the same score."""
    if target not in ("soft_am", "all_pass"):
        raise ValueError("verdict target must be soft_am or all_pass")
    codes, owners = [], []
    for t in batch.trajectories:
        t.vqa_list = batch.prompts[t.prompt_idx]["vqa_list"]
        codes.extend(t.images)
        owners.extend(t for _ in t.images)
    images = render_codes(decoder, codes, render_batch_size)
    position = 0
    for t in batch.trajectories:
        t.scored_images = images[position:position + len(t.images)]
        position += len(t.images)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    am, gm, per_question = reward.score_images(
        images, [batch.prompts[t.prompt_idx]["vqa_list"] for t in owners])
    position = 0
    for t in batch.trajectories:
        n = len(t.images)
        a, g, q = am[position:position + n], gm[position:position + n], per_question[position:position + n]
        position += n
        oi = t.one_iter
        oi.am_draft, oi.gm_draft, oi.question_scores_draft = a[0], g[0], list(q[0])
        t.am, t.gm, t.question_scores = a[-1], g[-1], list(q[-1])
        t.reward = t.am
        t.final_image = t.scored_images[-1]
        oi.reward_draft = oi.am_draft
        if target == "all_pass":
            oi.reward_verdict = float(oi.verdict == verdict_target(oi.question_scores_draft, threshold))
        else:
            oi.reward_verdict = 1.0 - abs(oi.am_draft - float(oi.verdict))
        oi.reward_final = t.am


def assign_janus_advantages(batch: RolloutBatch, gcfg: GRPOConfig) -> Dict[str, int]:
    """Each reward is normalized within its prompt group (unbiased std, as Janus)."""
    keys = [t.prompt_idx for t in batch.trajectories]
    counts = {"groups": len(set(keys))}
    for name in ("draft", "verdict", "final"):
        rewards = [getattr(t.one_iter, f"reward_{name}") for t in batch.trajectories]
        advantages, zero_var = normalize_groups(rewards, keys, gcfg, unbiased=True)
        for t, a in zip(batch.trajectories, advantages):
            setattr(t.one_iter, f"adv_{name}", a)
        counts[f"zero_var_{name}"] = zero_var
    for t in batch.trajectories:
        t.adv = t.one_iter.adv_final
    return counts


JANUS_COUNTS = ("n", "looks_good", "refined", "draft_pass", "looks_good_on_pass", "looks_good_on_fail",
                "verdict_correct",
                "mismatch_looks_good_im_start", "mismatch_correction_eos", "mismatch_cap",
                "rule_disagreement", "no_correction_clause", "improved", "degraded")
JANUS_SUMS = ("am_draft", "am_final", "gm_final", "reward_verdict", "delta_am_refined",
              "reward_draft_sq", "reward_verdict_sq", "reward_final_sq", "reflection_tokens")


def janus_stats(batch: RolloutBatch, prefix: str, fail_threshold: float = 0.5) -> Dict[str, float]:
    """``fail_threshold`` is the per-question pass threshold (same as the all_pass target)."""
    """Sums with a fixed key set (safe to all-reduce even with no refined episodes)."""
    s = {k: 0.0 for k in JANUS_COUNTS + JANUS_SUMS}
    for t in batch.trajectories:
        oi = t.one_iter
        passed = verdict_target(oi.question_scores_draft, fail_threshold)
        s["n"] += 1
        s["looks_good"] += oi.verdict
        s["refined"] += not oi.verdict
        s["draft_pass"] += passed
        s["looks_good_on_pass"] += oi.verdict and passed
        s["looks_good_on_fail"] += oi.verdict and not passed
        s["verdict_correct"] += oi.verdict == passed
        if oi.mismatch:
            s[f"mismatch_{oi.mismatch}"] += 1
        s["rule_disagreement"] += oi.verdict_disagreement
        s["no_correction_clause"] += not oi.has_correction_clause
        s["am_draft"] += oi.am_draft
        s["am_final"] += t.am
        s["gm_final"] += t.gm
        s["reward_verdict"] += oi.reward_verdict
        s["reflection_tokens"] += len(oi.reflection)
        for name in ("draft", "verdict", "final"):
            s[f"reward_{name}_sq"] += getattr(oi, f"reward_{name}") ** 2
        if not oi.verdict:
            delta = t.am - oi.am_draft
            s["delta_am_refined"] += delta
            s["improved"] += delta > 0
            s["degraded"] += delta < 0
    return {prefix + k: v for k, v in s.items()}


def finalize_janus_stats(s: Dict[str, float], prefix: str) -> Dict[str, float]:
    g = lambda k: s[prefix + k]
    n, refined, passed = max(g("n"), 1.0), max(g("refined"), 1.0), g("draft_pass")
    out = {k: g(k) / n for k in ("am_draft", "am_final", "gm_final", "reward_verdict", "reflection_tokens")}
    out.update(looks_good_rate=g("looks_good") / n, refine_rate=g("refined") / n,
               draft_pass_rate=passed / n, delta_am_refined=g("delta_am_refined") / refined,
               improved_rate=g("improved") / refined, degraded_rate=g("degraded") / refined,
               looks_good_on_pass=g("looks_good_on_pass") / max(passed, 1.0),
               looks_good_on_fail=g("looks_good_on_fail") / max(g("n") - passed, 1.0),
               verdict_accuracy=g("verdict_correct") / n,
               verdict_precision=g("looks_good_on_pass") / max(g("looks_good"), 1.0),
               verdict_recall=g("looks_good_on_pass") / max(passed, 1.0),
               rule_disagreement_rate=g("rule_disagreement") / n,
               no_correction_clause_rate=g("no_correction_clause") / n)
    for m in ("looks_good_im_start", "correction_eos", "cap"):
        out[f"mismatch_{m}_rate"] = g(f"mismatch_{m}") / n
    for name in ("draft", "verdict", "final"):
        mean = {"draft": g("am_draft"), "verdict": g("reward_verdict"), "final": g("am_final")}[name] / n
        out[f"reward_{name}_std"] = math.sqrt(max(g(f"reward_{name}_sq") / n - mean ** 2, 0.0))
    return {prefix + k: v for k, v in out.items()}


# ---------------------------------------------------------------------------
# teacher_tree: teacher reflections, judging, shadow credit, statistics
# ---------------------------------------------------------------------------

class TeacherReflector:
    """``teacher_fn`` for ``EpisodeRollout.run_teacher_tree``: render each new image
    (batched renderer) and ask the judge model for a reflection where requested."""

    def __init__(self, reward, decoder, render_batch_size=1, max_new_tokens=96, batch_size=16):
        self.reward, self.decoder = reward, decoder
        self.render_batch_size, self.max_new_tokens, self.batch_size = render_batch_size, max_new_tokens, batch_size
        self.render_seconds = self.teacher_seconds = 0.0

    @torch.no_grad()
    def __call__(self, prompts, codes, ask):
        start = time.time()
        images = render_codes(self.decoder, codes, self.render_batch_size)
        rendered = time.time()
        asked = [i for i, flag in enumerate(ask) if flag]
        replies = [None] * len(codes)
        if asked:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # the judge process shares this GPU
            texts = self.reward.generate_texts([images[i] for i in asked],
                                               [teacher_instruction(prompts[i]) for i in asked],
                                               self.max_new_tokens, self.batch_size)
            for i, text in zip(asked, texts):
                replies[i] = text
        self.render_seconds += rendered - start
        self.teacher_seconds += time.time() - rendered
        return list(zip(images, replies))


@torch.no_grad()
def score_teacher_tree(batch: RolloutBatch, reward, threshold: float = 0.5):
    """Judge every image node once, on the exact pixels the teacher saw. A leaf's
    reward is its final image's AM; a shadow reflection's reward is 1 when its
    verdict equals the all_pass target of the image it follows."""
    image_nodes = [n for n in batch.nodes if n.kind == "image"]
    if not image_nodes or any(n.image is None for n in image_nodes):
        raise ValueError("Every teacher_tree image node must keep its rendered image.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    am, gm, per_question = reward.score_images(
        [n.image for n in image_nodes], [batch.prompts[n.prompt_idx]["vqa_list"] for n in image_nodes])
    for node, a, g, q in zip(image_nodes, am, gm, per_question):
        node.am, node.gm, node.question_scores = a, g, list(q)
        passed = verdict_target(node.question_scores, threshold)
        for shadow in node.shadows:
            shadow.reward = float(shadow.verdict == passed)
    for t in batch.trajectories:
        path = [n for n in t.nodes if n.kind == "image"]
        t.vqa_list = batch.prompts[t.prompt_idx]["vqa_list"]
        t.am, t.gm, t.question_scores = path[-1].am, path[-1].gm, list(path[-1].question_scores)
        t.reward = t.am
        t.final_image = path[-1].image
        t.scored_images = [n.image for n in path]
        t.image_scores = [(n.am, n.gm, list(n.question_scores)) for n in path]


def assign_shadow_advantages(batch: RolloutBatch, gcfg: GRPOConfig) -> int:
    """Shadow verdict rewards normalized within each prompt group (unbiased std, as Janus)."""
    shadows = [(n.prompt_idx, s) for n in batch.nodes for s in n.shadows]
    advantages, zero_var = normalize_groups([s.reward for _, s in shadows], [p for p, _ in shadows],
                                            gcfg, unbiased=True)
    for (_, shadow), advantage in zip(shadows, advantages):
        shadow.adv = advantage
    return zero_var


TEACHER_STOPS = ("teacher_looks_good", "max_refinements", "teacher_malformed", "max_seq_len")
TEACHER_TREE_KEYS = (
    "n_final", "am_final", "gm_final", "refinements", "drafts", "am_draft", "draft_pass",
    "refined_images", "delta_am_refined", "improved", "degraded",
    "teacher_asked", "teacher_parsed", "teacher_looks_good", "teacher_correct", "teacher_pass",
    "teacher_looks_good_on_pass", "teacher_looks_good_on_fail", "teacher_tokens",
    "shadows", "shadow_correct", "shadow_looks_good", "shadow_pass", "shadow_looks_good_on_pass",
    "shadow_looks_good_on_fail", "shadow_mismatch", "shadow_capped", "shadow_tokens",
) + tuple(f"stop_{reason}" for reason in TEACHER_STOPS)


def teacher_tree_stats(batch: RolloutBatch, prefix: str, threshold: float = 0.5) -> Dict[str, float]:
    """Sums with a fixed key set (safe to all-reduce). Leaves are weighted by mass."""
    s = {k: 0.0 for k in TEACHER_TREE_KEYS}
    for t in batch.trajectories:
        s["n_final"] += t.mass
        s["am_final"] += t.mass * t.am
        s["gm_final"] += t.mass * t.gm
        s["refinements"] += t.mass * (len(t.images) - 1)
        if t.stop_reason in TEACHER_STOPS:
            s[f"stop_{t.stop_reason}"] += t.mass
    for node in batch.nodes:
        if node.kind == "reflection":
            s["teacher_tokens"] += len(node.actions)
            continue
        passed = verdict_target(node.question_scores, threshold)
        if node.parent is None:
            s["drafts"] += 1
            s["am_draft"] += node.am
            s["draft_pass"] += passed
        else:
            delta = node.am - node.parent.parent.am  # image <- teacher reflection <- image
            s["refined_images"] += 1
            s["delta_am_refined"] += delta
            s["improved"] += delta > 0
            s["degraded"] += delta < 0
        if node.teacher_reply is not None:
            s["teacher_asked"] += 1
            if node.teacher_verdict is not None:
                s["teacher_parsed"] += 1
                s["teacher_pass"] += passed
                s["teacher_looks_good"] += node.teacher_verdict
                s["teacher_correct"] += node.teacher_verdict == passed
                s["teacher_looks_good_on_pass"] += node.teacher_verdict and passed
                s["teacher_looks_good_on_fail"] += node.teacher_verdict and not passed
        for shadow in node.shadows:
            s["shadows"] += 1
            s["shadow_correct"] += shadow.reward
            s["shadow_looks_good"] += shadow.verdict
            s["shadow_pass"] += passed
            s["shadow_looks_good_on_pass"] += shadow.verdict and passed
            s["shadow_looks_good_on_fail"] += shadow.verdict and not passed
            s["shadow_mismatch"] += not shadow.stop_token_agrees and not shadow.capped
            s["shadow_capped"] += shadow.capped
            s["shadow_tokens"] += len(shadow.tokens)
    return {prefix + k: float(v) for k, v in s.items()}


def finalize_teacher_tree_stats(s: Dict[str, float], prefix: str) -> Dict[str, float]:
    g = lambda k: s[prefix + k]
    div = lambda a, b: g(a) / max(g(b), 1.0) if isinstance(b, str) else g(a) / max(b, 1.0)
    out = {k: div(k, "n_final") for k in ("am_final", "gm_final", "refinements")}
    out.update({f"stop_{reason}_rate": div(f"stop_{reason}", "n_final") for reason in TEACHER_STOPS})
    out.update(am_draft=div("am_draft", "drafts"), draft_pass_rate=div("draft_pass", "drafts"),
               delta_am_refined=div("delta_am_refined", "refined_images"),
               improved_rate=div("improved", "refined_images"), degraded_rate=div("degraded", "refined_images"),
               teacher_malformed_reply_rate=(g("teacher_asked") - g("teacher_parsed")) / max(g("teacher_asked"), 1.0),
               teacher_looks_good_rate=div("teacher_looks_good", "teacher_parsed"),
               teacher_verdict_accuracy=div("teacher_correct", "teacher_parsed"),
               teacher_verdict_precision=div("teacher_looks_good_on_pass", "teacher_looks_good"),
               teacher_verdict_recall=div("teacher_looks_good_on_pass", "teacher_pass"),
               teacher_looks_good_on_fail=g("teacher_looks_good_on_fail") / max(g("teacher_parsed") - g("teacher_pass"), 1.0),
               teacher_tokens=div("teacher_tokens", "teacher_parsed"),
               shadow_verdict_accuracy=div("shadow_correct", "shadows"),
               shadow_looks_good_rate=div("shadow_looks_good", "shadows"),
               shadow_verdict_precision=div("shadow_looks_good_on_pass", "shadow_looks_good"),
               shadow_verdict_recall=div("shadow_looks_good_on_pass", "shadow_pass"),
               shadow_looks_good_on_pass=div("shadow_looks_good_on_pass", "shadow_pass"),
               shadow_looks_good_on_fail=g("shadow_looks_good_on_fail") / max(g("shadows") - g("shadow_pass"), 1.0),
               shadow_mismatch_rate=div("shadow_mismatch", "shadows"),
               shadow_capped_rate=div("shadow_capped", "shadows"),
               shadow_tokens=div("shadow_tokens", "shadows"),
               images_per_draft=(g("drafts") + g("refined_images")) / max(g("drafts"), 1.0))
    return {prefix + k: v for k, v in out.items()}


@torch.no_grad()
def score_all_images(batch: RolloutBatch, reward, decoder, render_batch_size: int = 1):
    """Render every image of every episode once and judge them in one call."""
    owners = [t for t in batch.trajectories for _ in t.images]
    images = render_codes(decoder, [code for t in batch.trajectories for code in t.images], render_batch_size)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    am, gm, per_question = reward.score_images(images, [batch.prompts[t.prompt_idx]["vqa_list"] for t in owners])
    position = 0
    for t in batch.trajectories:
        n = len(t.images)
        t.scored_images = images[position:position + n]
        t.image_scores = [(a, g, list(q)) for a, g, q in zip(am[position:position + n], gm[position:position + n],
                                                            per_question[position:position + n])]
        position += n
        t.vqa_list = batch.prompts[t.prompt_idx]["vqa_list"]
        t.am, t.gm, t.question_scores = t.image_scores[-1][0], t.image_scores[-1][1], list(t.image_scores[-1][2])
        t.reward = t.am
        t.final_image = t.scored_images[-1]


def looks_good_stats(batch: RolloutBatch, prefix: str, threshold: float, max_images: int) -> Dict[str, float]:
    """Sums for the self-reflection looks_good protocol (teacher_tree validation).
    ``am_upto_k``: AM of the last image at or before image k (an episode that
    stopped earlier keeps its final image)."""
    keys = ["n", "am_final", "gm_final", "am_draft", "refinements", "verdicts", "verdict_correct",
            "looks_good", "image_pass", "looks_good_on_pass", "looks_good_on_fail"]
    keys += [f"stop_{r}" for r in ("looks_good", "max_refinements", "max_seq_len")]
    keys += [f"am_upto_{k}" for k in range(1, max_images + 1)]
    s = {k: 0.0 for k in keys}
    for t in batch.trajectories:
        scores = t.image_scores
        s["n"] += 1
        s["am_final"] += t.am
        s["gm_final"] += t.gm
        s["am_draft"] += scores[0][0]
        s["refinements"] += len(t.images) - 1
        if f"stop_{t.stop_reason}" in s:
            s[f"stop_{t.stop_reason}"] += 1
        for k in range(1, max_images + 1):
            s[f"am_upto_{k}"] += scores[min(k, len(scores)) - 1][0]
        for verdict, (_, _, q) in zip(t.verdicts, scores):
            passed = verdict_target(q, threshold)
            s["verdicts"] += 1
            s["verdict_correct"] += verdict == passed
            s["looks_good"] += verdict
            s["image_pass"] += passed
            s["looks_good_on_pass"] += verdict and passed
            s["looks_good_on_fail"] += verdict and not passed
    return {prefix + k: float(v) for k, v in s.items()}


def finalize_looks_good_stats(s: Dict[str, float], prefix: str, max_images: int) -> Dict[str, float]:
    g = lambda k: s[prefix + k]
    n, verdicts = max(g("n"), 1.0), max(g("verdicts"), 1.0)
    out = {k: g(k) / n for k in ("am_final", "gm_final", "am_draft", "refinements")}
    out.update({f"am_upto_{k}": g(f"am_upto_{k}") / n for k in range(1, max_images + 1)})
    out.update({f"{r}_rate": g(f"stop_{r}") / n for r in ("looks_good", "max_refinements", "max_seq_len")})
    out.update(verdict_accuracy=g("verdict_correct") / verdicts,
               verdict_looks_good_rate=g("looks_good") / verdicts,
               verdict_precision=g("looks_good_on_pass") / max(g("looks_good"), 1.0),
               verdict_recall=g("looks_good_on_pass") / max(g("image_pass"), 1.0),
               looks_good_on_fail=g("looks_good_on_fail") / max(g("verdicts") - g("image_pass"), 1.0),
               delta_am_final_vs_draft=(g("am_final") - g("am_draft")) / n)
    return {prefix + k: v for k, v in out.items()}


def rollout_stats(batch: RolloutBatch, prefix: str) -> Dict[str, float]:
    # Tree leaves are weighted by their share of a draft (mass), so rates and
    # means stay per-episode quantities; without branching every mass is 1.
    trajectories = batch.trajectories
    stats = {
        "n_final": sum(t.mass for t in trajectories),
        "am_final": sum(t.mass * t.am for t in trajectories),
        "gm_final": sum(t.mass * t.gm for t in trajectories),
        "reward_final": sum(t.mass * t.reward for t in trajectories),
        "mean_refinements": sum(t.mass * (len(t.images) - 1) for t in trajectories),
        "mean_sampled_tokens": sum(t.mass * sum(k != 0 for k in t.kinds) for t in trajectories),
        "truncated_rate": sum(t.mass * (t.stop_reason != "eos") for t in trajectories),
    }
    for reason in ("eos", "max_refinements", "max_seq_len", "reflection_limit"):
        stats[f"{reason}_rate"] = sum(t.mass * (t.stop_reason == reason) for t in trajectories)
    return {prefix + k: v for k, v in stats.items()}


TREE_STAT_GROUPS = ("draft", "reflection", "refined", "stop", "continue")


def tree_stats(batch: RolloutBatch, eos_id: int, im_start_id: int, prefix: str) -> Dict[str, float]:
    """Sums for tree diagnostics (fixed key set, so they can be all-reduced).
    stop/continue: reflections ending in EOS/<im_start> that have siblings."""
    stats = {"tree_prompts": len({t.prompt_idx for t in batch.trajectories}),
             "tree_leaves": len(batch.trajectories), "tree_nodes": len(batch.nodes)}
    for group in TREE_STAT_GROUPS:
        stats[f"tree_{group}_adv_sum"] = 0.0
        stats[f"tree_{group}_count"] = 0
    def add(group, value):
        stats[f"tree_{group}_adv_sum"] += value
        stats[f"tree_{group}_count"] += 1
    for node in batch.nodes:
        if node.parent is None:
            add("draft", abs(node.adv))
        elif node.kind == "image":
            add("refined", abs(node.adv))
        else:
            add("reflection", abs(node.adv))
            if len(node.parent.children) > 1 and node.actions:
                if node.actions[-1] == eos_id:
                    add("stop", node.adv)
                elif node.actions[-1] == im_start_id:
                    add("continue", node.adv)
    return {prefix + k: v for k, v in stats.items()}


def finalize_tree_stats(s: Dict[str, float], prefix: str) -> Dict[str, float]:
    prompts = max(s[f"{prefix}tree_prompts"], 1.0)
    out = {f"{prefix}tree_leaves_per_prompt": s[f"{prefix}tree_leaves"] / prompts,
           f"{prefix}tree_nodes_per_prompt": s[f"{prefix}tree_nodes"] / prompts}
    for group in TREE_STAT_GROUPS:
        name = "mean_adv" if group in ("stop", "continue") else "mean_abs_adv"
        out[f"{prefix}tree_{name}_{group}"] = (s[f"{prefix}tree_{group}_adv_sum"]
                                               / max(s[f"{prefix}tree_{group}_count"], 1.0))
    return out


def finalize_stats(s: Dict[str, float], prefix: str) -> Dict[str, float]:
    count = max(s[f"{prefix}n_final"], 1.0)
    return {prefix + name: s[prefix + name] / count for name in (
        "am_final", "gm_final", "reward_final", "mean_refinements", "mean_sampled_tokens",
        "truncated_rate", "eos_rate", "max_refinements_rate", "max_seq_len_rate", "reflection_limit_rate")}


def rl_method_settings(args):
    mode = getattr(args, "length_normalization", "episode")
    objective = getattr(args, "objective", "final_episode")
    verdicts, teacher = objective in VERDICT_OBJECTIVES, objective == "teacher_tree"
    return {"length_normalization": mode,
            "normalization_constant": args.max_seq_len if mode == "constant" else None,
            "kl_gradient_correction": getattr(args, "kl_gradient_correction", False),
            "reset_context_per_turn": getattr(args, "reset_context_per_turn", False),
            "text_policy_gradient": getattr(args, "text_policy_gradient", "all"),
            "tree_branch": parse_tree_branch(getattr(args, "tree_branch", [])),
            "objective": objective,
            "verdict_rule": getattr(args, "verdict_rule", "correction") if verdicts else None,
            "adv_std_floor": float(getattr(args, "adv_std_floor", 0.0)),
            "verdict_target": getattr(args, "verdict_target", "soft_am") if verdicts else None,
            "verdict_pass_threshold": float(getattr(args, "verdict_pass_threshold", 0.5)) if verdicts else None,
            "shadow_reflections": int(args.shadow_reflections) if teacher else None,
            "shadow_coef": float(args.shadow_coef) if teacher else None,
            "teacher_ce_coef": float(args.teacher_ce_coef) if teacher else None,
            "teacher_max_tokens": int(args.teacher_max_tokens) if teacher else None,
            "teacher_prompt_version": TEACHER_PROMPT_VERSION if teacher else None,
            "teacher_model": (os.path.basename(os.path.normpath(args.reward_model_name_or_path))
                              if teacher else None)}


def validate_resume_checkpoint(path, args=None):
    with open(os.path.join(path, "state.json")) as f:
        state = json.load(f)
    if state.get("objective") != RL_OBJECTIVE:
        raise ValueError("Cannot resume a checkpoint from the old local-reward or semantic-reward objective. "
                         "Use a new output directory/run name for pixel Qwen soft-TIFA AM GRPO.")
    if args is not None:
        legacy = {"length_normalization": "episode", "normalization_constant": None,
                  "kl_gradient_correction": False}
        saved = dict(state.get("rl_method", legacy))
        saved.setdefault("reset_context_per_turn", False)
        saved.setdefault("text_policy_gradient", "all")
        saved["tree_branch"] = parse_tree_branch(saved.get("tree_branch", []))
        saved.setdefault("objective", "final_episode")
        saved.setdefault("verdict_rule", None)
        saved.setdefault("adv_std_floor", 0.0)
        janus_saved = saved["objective"] == "janus_one_iteration"
        # Janus checkpoints written before these keys used the soft-AM verdict reward.
        saved.setdefault("verdict_target", "soft_am" if janus_saved else None)
        saved.setdefault("verdict_pass_threshold", 0.5 if janus_saved else None)
        for key in ("shadow_reflections", "shadow_coef", "teacher_ce_coef", "teacher_max_tokens",
                    "teacher_prompt_version", "teacher_model"):
            saved.setdefault(key, None)  # written before teacher_tree existed
        requested = rl_method_settings(args)
        if saved != requested:
            raise ValueError(f"RL method settings differ from checkpoint: saved={saved}, "
                             f"requested={requested}. Use matching flags to resume, or a fresh "
                             "output directory/run name for the new method.")
    return state


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_on_rollouts(model, rollout: EpisodeRollout, batch: RolloutBatch, args, gcfg: GRPOConfig,
                  optimizer, scheduler, device) -> Dict[str, float]:
    rollout.cfg.validate_for_training()
    img_start, img_end = rollout.img_start, rollout.img_end
    text_pg = getattr(args, "text_policy_gradient", "all")
    janus = getattr(args, "objective", "final_episode") == "janus_one_iteration"
    teacher_tree = getattr(args, "objective", "final_episode") == "teacher_tree"
    if janus:
        # Three segment losses per episode; weights already hold the per-segment means.
        rows = [rollout.build_janus_rows(batch.trajectories)]
        denom = max(len(batch.trajectories), 1)
    elif teacher_tree:
        # Image PG/KL per draft subtree, mean shadow loss and mean teacher NLL;
        # the row weights already hold those normalizations for this denominator.
        rows = [rollout.build_teacher_tree_rows(batch, args.teacher_ce_coef, args.shadow_coef)]
        denom = max(sum(1 for n in batch.nodes if n.parent is None), 1)
    elif batch.nodes:
        # Tree: each node's actions are trained once; each draft's subtree
        # weighs like one episode, so the denominator counts drafts.
        rows = [rollout.build_tree_training_rows(batch, args.reflect_token_weight,
                                                 gcfg.length_normalization, text_pg)]
        denom = max(sum(1 for n in batch.nodes if n.parent is None), 1)
    else:
        trajectories = sorted(batch.trajectories, key=lambda t: len(t.seq))
        micro = [trajectories[i:i + args.train_micro_batch]
                 for i in range(0, len(trajectories), args.train_micro_batch)]
        rows = [rollout.build_training_rows(m, args.reflect_token_weight, gcfg.length_normalization,
                                            text_pg)
                for m in micro]
        denom = max(len(trajectories), 1)  # average episodes after token normalization
    if rollout.cfg.reset_context_per_turn or batch.nodes or janus:
        # Segment count varies independently of episode count. Bound GPU batch
        # size without changing the episode denominator or segment weights.
        split_rows = []
        for r in rows:
            for start in range(0, len(r["input_ids"]), args.train_micro_batch):
                end = start + args.train_micro_batch
                width = int(r["attention_mask"][start:end].sum(1).max())
                split_rows.append({k: v[start:end, :width] for k, v in r.items()})
        rows = split_rows
    total_weight = sum(float(r["weight"].sum()) for r in rows)
    if gcfg.length_normalization == "constant" and dist.is_initialized():
        # The caller averages metrics across ranks. Divide ratio diagnostic
        # numerators by the mean rank weight so that this gives a global
        # weighted mean even when ranks sample different episode lengths.
        rank_weight = torch.tensor(total_weight, dtype=torch.float64, device=device)
        dist.all_reduce(rank_weight, op=dist.ReduceOp.SUM)
        total_weight = float(rank_weight) / dist.get_world_size()
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
            pg_mask = r["pg_mask"].to(device)[:, 1:][sel] if "pg_mask" in r else None
            segments = r["pos_segment"].to(device)[:, 1:][sel] if "pos_segment" in r else None
            ce_weight = r["ce_weight"].to(device)[:, 1:][sel] if "ce_weight" in r else None
            loss, metrics = grpo_loss(logp, old_logps[i], ref_logps[i], adv, weight, kinds, denom, gcfg,
                                      pg_mask=pg_mask, segments=segments, ce_weight=ce_weight,
                                      segment_names=TEACHER_TREE_SEGMENTS if teacher_tree else SEGMENT_NAMES)
            loss.backward()
            w = float(weight.sum())
            for k, v in metrics.items():
                # sum_* metrics are plain sums; the others are weighted means.
                agg[k] = agg.get(k, 0.0) + (v if k.startswith("sum_") else v * w)
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
    if "sum_ce_loss" in agg:  # teacher_tree distillation (last epoch)
        agg["ce_loss"] = agg.pop("sum_ce_loss")  # divided by denom below, like pg_loss
        agg["teacher_nll"] = agg.pop("sum_teacher_nll") / max(agg.pop("sum_teacher_tokens"), 1.0)
    for k in list(agg):
        if k not in ("loss", "grad_norm", "teacher_nll"):
            # PG/KL retain the objective's normalization. Ratio diagnostics
            # remain weighted means rather than shrinking with episode length.
            metric_denom = (max(total_weight, 1e-30)
                            if gcfg.length_normalization == "constant"
                            and k in ("clip_frac", "approx_kl_old") else denom)
            agg[k] /= metric_denom
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
    janus = getattr(args, "objective", "final_episode") == "janus_one_iteration"
    teacher_tree = getattr(args, "objective", "final_episode") == "teacher_tree"
    threshold = getattr(args, "verdict_pass_threshold", 0.5)
    max_images = rollout.cfg.max_refinements + 1
    render_batch = getattr(args, "render_batch_size", 1)
    for start in range(0, len(val_rows), bs):
        # Same stochastic policy as training; one episode per prompt.
        if teacher_tree:
            # Test-time behaviour: the policy's own reflections, no teacher.
            batch = rollout.run_looks_good(model, val_rows[start:start + bs], num_rollouts=1,
                                           verdict_rule=args.verdict_rule)
            score_all_images(batch, reward, decoder, render_batch)
            stats = looks_good_stats(batch, "val/", threshold, max_images)
        elif janus:
            batch = rollout.run_one_iteration(model, val_rows[start:start + bs], num_rollouts=1,
                                              verdict_rule=args.verdict_rule)
            threshold = getattr(args, "verdict_pass_threshold", 0.5)
            score_one_iteration(batch, reward, decoder, render_batch,
                                getattr(args, "verdict_target", "soft_am"), threshold)
            stats = janus_stats(batch, "val/", threshold)
        else:
            batch = rollout.run(model, val_rows[start:start + bs], num_rollouts=1)
            score_rollouts(batch, reward, decoder, render_batch)
            stats = rollout_stats(batch, "val/")
        for k, v in stats.items():
            sums[k] = sums.get(k, 0.0) + v
        if is_main() and len(samples) < args.log_images:
            samples.extend((batch.prompts[t.prompt_idx]["prompt"], t)
                           for t in batch.trajectories[:args.log_images - len(samples)])
    sums = reduce_sum(sums, device)
    if teacher_tree:
        out = finalize_looks_good_stats(sums, "val/", max_images)
    else:
        out = finalize_janus_stats(sums, "val/") if janus else finalize_stats(sums, "val/")
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
                            reset_context_per_turn=bool(trajectory.segments),
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
        if trajectory.scored_images:
            return list(trajectory.scored_images)  # every image was scored: reuse exact pixels
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
    resume_state = validate_resume_checkpoint(resume_dir, args) if resume_dir else None
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
        reset_context_per_turn=args.reset_context_per_turn,
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
                             device, args.reward_timeout,
                             question_batch_size=args.reward_question_batch_size)
    teacher = (TeacherReflector(reward, decoder, args.render_batch_size, args.teacher_max_tokens,
                                args.teacher_batch_size) if args.objective == "teacher_tree" else None)
    gcfg = GRPOConfig(clip_eps=args.clip_eps, kl_coef=args.kl_coef, adv_norm=args.adv_norm,
                      reflect_token_weight=args.reflect_token_weight,
                      length_normalization=args.length_normalization,
                      kl_gradient_correction=args.kl_gradient_correction,
                      adv_std_floor=args.adv_std_floor)

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
    completed_step = start_step
    last_saved_step = start_step if resume_dir is not None else None
    for step in range(start_step + 1, args.max_steps + 1):
        if coordinated_stop(args, device):
            break
        prompts = next(prompt_iter)
        torch.manual_seed(args.seed * 1_000_003 + step * 7919 + rank)
        t0 = time.time()
        janus = args.objective == "janus_one_iteration"
        teacher_tree = args.objective == "teacher_tree"
        if janus:
            batch = rollout.run_one_iteration(model, prompts, verdict_rule=args.verdict_rule)
        elif teacher_tree:
            teacher.render_seconds = teacher.teacher_seconds = 0.0
            batch = rollout.run_teacher_tree(model, prompts, args.tree_branch, teacher,
                                             args.shadow_reflections, args.verdict_rule)
        elif args.tree_branch:
            batch = rollout.run_tree(model, prompts, args.tree_branch)
        else:
            batch = rollout.run(model, prompts)
        t1 = time.time()
        model.eval()
        if janus:
            score_one_iteration(batch, reward, decoder, args.render_batch_size,
                                args.verdict_target, args.verdict_pass_threshold)
            janus_counts = assign_janus_advantages(batch, gcfg)
            zero_var, n_groups = janus_counts["zero_var_final"], janus_counts["groups"]
        elif teacher_tree:
            score_teacher_tree(batch, reward, args.verdict_pass_threshold)
            zero_var, n_groups = assign_tree_advantages(batch, gcfg)
            shadow_zero_var = assign_shadow_advantages(batch, gcfg)
        else:
            score_rollouts(batch, reward, decoder, args.render_batch_size)
            zero_var, n_groups = (assign_tree_advantages(batch, gcfg) if batch.nodes
                                  else assign_advantages(batch, gcfg))
        t2 = time.time()
        train_metrics = train_on_rollouts(model, rollout, batch, args, gcfg, optimizer, scheduler, device)
        completed_step = step
        t3 = time.time()

        if step % args.logging_steps == 0:
            if teacher_tree:
                sums = teacher_tree_stats(batch, "train/", args.verdict_pass_threshold)
                sums["train/zero_var_groups_shadow"] = shadow_zero_var
            elif janus:
                sums = janus_stats(batch, "train/", args.verdict_pass_threshold)
                for name in ("draft", "verdict"):
                    sums[f"train/zero_var_groups_{name}"] = janus_counts[f"zero_var_{name}"]
            else:
                sums = rollout_stats(batch, "train/")
            sums["train/zero_var_groups"] = zero_var
            sums["train/groups"] = n_groups
            if args.tree_branch or teacher_tree:
                sums.update(tree_stats(batch, rollout.eos_id, rollout.im_start_tok, "train/"))
            sums = reduce_sum(sums, device)
            if teacher_tree:
                metrics = finalize_teacher_tree_stats(sums, "train/")
                metrics["train/frac_zero_adv_groups_shadow"] = (
                    sums["train/zero_var_groups_shadow"] / max(sums["train/groups"], 1.0))
            else:
                metrics = finalize_janus_stats(sums, "train/") if janus else finalize_stats(sums, "train/")
            if janus:
                for name in ("draft", "verdict"):
                    metrics[f"train/frac_zero_adv_groups_{name}"] = (
                        sums[f"train/zero_var_groups_{name}"] / max(sums["train/groups"], 1.0))
            if args.tree_branch or teacher_tree:
                metrics.update(finalize_tree_stats(sums, "train/"))
            metrics["train/frac_zero_adv_groups"] = sums["train/zero_var_groups"] / max(sums["train/groups"], 1.0)
            metrics.update({f"train/{k}": v for k, v in reduce_mean(train_metrics, device).items()})
            if teacher_tree:  # parts of time/rollout
                metrics.update({"time/teacher_render": teacher.render_seconds,
                                "time/teacher_generate": teacher.teacher_seconds})
            metrics.update({"time/rollout": t1 - t0, "time/reward": t2 - t1, "time/train": t3 - t2,
                            "train/lr": scheduler.get_last_lr()[0], "train/epoch_frac":
                            step * args.prompts_per_gpu / max(train_ds.per_epoch(), 1)})
            log(metrics, step)

        if args.save_steps and step % args.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, step, args, device)
            last_saved_step = step
        if coordinated_stop(args, device):
            break
        if args.eval_steps and val_rows and step % args.eval_steps == 0:
            log(run_validation(model, rollout, reward, val_rows, args, device, step, decoder), step, validation=True)

    if last_saved_step != completed_step:
        save_checkpoint(model, optimizer, scheduler, completed_step, args, device)
    reward.close()
    if is_main():
        atomic_json(os.path.join(args.output_dir,
                    f"training_status_{os.environ.get('SLURM_JOB_ID', 'local')}.json"),
                    dict(start_step=start_step, step=completed_step, target_step=args.max_steps,
                         complete=completed_step >= args.max_steps,
                         reason='target_reached' if completed_step >= args.max_steps else 'deadline_or_stop_file',
                         checkpoint=os.path.join(args.output_dir, f'checkpoint-{completed_step}')))
    rank0_print(f"Done at step {completed_step}/{args.max_steps}.")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
