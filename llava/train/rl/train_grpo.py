"""GRPO fine-tuning of Tar on iterative image generation
(draft -> self-reflect -> refine), with a latent-space GenEval2-style reward.

Per optimizer step, each rank:
  1. samples a rollout tree for ``--prompts_per_gpu`` prompts
     (``--branch G0,G1,...``: G0 drafts per prompt, G_k children per node),
  2. scores every image by answering the prompt's VQA questions on it, either
     in latent space with a frozen Tar LM (``--reward_kind latent``, no pixel
     decode) or on the de-tokenized PNG with a served VLM judge
     (``--reward_kind pixel``, see llava/train/rl/pixel_reward_server.py),
     score(node)    = a*AM + (1-a)*GM of the node's image
     reward(draft)  = score(draft)                          (--draft_reward own)
                    = mean score over the draft's leaves    (--draft_reward children)
     reward(round k)= score(child) - score(parent)          (refined)
                    = -stop_penalty * max(0, stop_threshold - score(parent))
                                                            ("looks good")
  3. normalises rewards within groups (drafts of a prompt / children of a node),
  4. takes a clipped policy-gradient step on the LoRA adapters with a KL penalty
     to the frozen base (adapters disabled).

``--critique_source oracle`` trains the refiner alone: the reflection is not
sampled but teacher-forced from the draft's failed VQA questions
(llava/train/rl/oracle_critique.py), so all children of a draft share one
correct critique and their image tokens are compared under a fixed context.

Only the LoRA parameters train, so plain torch DDP-style gradient all-reduce
is used instead of DeepSpeed. Resumable: ``checkpoint-N/`` holds the adapter,
optimizer, scheduler and data position.

Launch: see output_dir/rl_ft/bash.sh.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from transformers import (AutoTokenizer, Qwen2ForCausalLM, get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llava.train.rl.dataset import GenEval2PromptDataset
from llava.train.rl.grpo import GRPOConfig, grpo_loss, masked_token_logprobs, normalize_groups
from llava.train.rl.oracle_critique import SKILL_ORDER, oracle_critique
from llava.train.rl.pixel_reward import PixelVQAReward
from llava.train.rl.reward import ANSWER_SUFFIXES, RewardConfig, TarLatentVQAReward
from llava.train.rl.rollout import Critique, LOOKS_GOOD_RE, Node, RolloutBatch, RolloutConfig, TreeRollout

# Per-skill stats buckets; questions with any other skill tag land in "other".
STAT_SKILLS = tuple(SKILL_ORDER) + ("other",)
CRITIQUE_SOURCES = ("oracle", "policy")   # frozen-critique sources split in the stats


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
    p.add_argument("--branch", default="4,2", help="Fan-out per round: drafts,children,...")
    p.add_argument("--scale", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--gen_seq_len", type=int, default=729)
    p.add_argument("--img_temperature", type=float, default=1.0)
    p.add_argument("--img_top_k", type=int, default=1200)
    p.add_argument("--img_top_p", type=float, default=0.95)
    p.add_argument("--reflect_tokens", type=int, default=128)
    p.add_argument("--reflect_temperature", type=float, default=1.0)
    p.add_argument("--reflect_top_k", type=int, default=0)
    p.add_argument("--reflect_top_p", type=float, default=0.95)
    p.add_argument("--gen_batch_size", type=int, default=16)
    # Reward
    p.add_argument("--alpha", type=float, default=1.0, help="reward = a*AM + (1-a)*GM")
    p.add_argument("--answer_suffix", default="llava", choices=sorted(ANSWER_SUFFIXES))
    p.add_argument("--reward_batch_size", type=int, default=16)
    p.add_argument("--reward_model_name_or_path", default=None,
                   help="Checkpoint scoring the VQA reward. Default: reuse the policy's "
                        "frozen base weights (LoRA disabled), which costs no extra GPU memory.")
    p.add_argument("--reward_kind", default="latent", choices=["latent", "pixel"],
                   help="latent: score the emitted image tokens with a Tar LM. "
                        "pixel: de-tokenize to a PNG and score it with the VLM judge "
                        "served by llava/train/rl/pixel_reward_server.py.")
    p.add_argument("--reward_server_url", default=None,
                   help="Base URL of the pixel reward server (--reward_kind pixel).")
    p.add_argument("--reward_images_per_request", type=int, default=8,
                   help="Decoded images per POST to the reward server.")
    p.add_argument("--reward_decode_batch", type=int, default=8,
                   help="Images de-tokenized per de-tokenizer forward pass.")
    p.add_argument("--draft_reward", default="own", choices=["own", "children"],
                   help="own: a draft is rewarded with its own score. children: with the "
                        "mean final score of its leaves, so it is credited for the outcome "
                        "of the whole trajectory (refinement tokens keep the parent baseline).")
    p.add_argument("--stop_penalty", type=float, default=0.0,
                   help="A 'looks good' child is rewarded -stop_penalty * max(0, "
                        "stop_threshold - parent_score) instead of 0, so stopping on an "
                        "imperfect image costs something. 0 disables.")
    p.add_argument("--stop_threshold", type=float, default=1.0,
                   help="Parent score at or above which 'looks good' is free (see --stop_penalty).")
    p.add_argument("--min_refines", type=int, default=0,
                   help="The first N children of every parent must refine: a 'looks good' "
                        "reflection is re-sampled (up to --refine_resample_tries times). "
                        "Training only; validation always uses the policy as is.")
    p.add_argument("--refine_resample_tries", type=int, default=3)
    p.add_argument("--critique_source", default="policy", choices=["policy", "oracle", "hybrid"],
                   help="policy: sample the reflection from the policy and train it. oracle: "
                        "teacher-force the oracle critique of the parent's failed VQA "
                        "questions (llava/train/rl/oracle_critique.py), untrained, so every "
                        "child of a parent shares one correct, complete critique and only the "
                        "refiner's image tokens get gradient. hybrid: per parent, with prob "
                        "--hybrid_oracle_frac the oracle critique, otherwise ONE reflection "
                        "sampled from the (frozen) policy and shared by all children, also "
                        "untrained; a policy reflection that says 'looks good' is re-sampled "
                        "--refine_resample_tries times, then the oracle is used. In both, "
                        "parents that fail nothing get no children. Validation always uses "
                        "the policy's own critic.")
    p.add_argument("--hybrid_oracle_frac", type=float, default=0.5,
                   help="--critique_source hybrid: fraction of parents given the oracle critique.")
    p.add_argument("--oracle_fail_threshold", type=float, default=0.5,
                   help="A question fails when the judge's answer probability is below this "
                        "(oracle critique, and the fix/break rates in the stats).")
    p.add_argument("--break_weight", type=float, default=1.0,
                   help="Refine reward = alpha * mean over questions of w_q * (p_child - p_parent) "
                        "+ (1 - alpha) * (GM_child - GM_parent), with w_q = this for the "
                        "questions the parent passed and 1 otherwise (the GM term has no "
                        "per-question split, so it is unweighted). 1.0 is exactly "
                        "score(child) - score(parent); >1 makes breaking a passed atom cost "
                        "more than fixing a failed one earns.")
    p.add_argument("--eval_oracle", action=argparse.BooleanOptionalAction, default=False,
                   help="Also validate with oracle critiques (val_oracle/*): the refiner's "
                        "fix/break rates under a correct critique, independent of the critic.")
    # GRPO
    p.add_argument("--group", default="parent", choices=["parent", "prompt"],
                   help="Normalise refinement rewards among siblings (parent) or all "
                        "same-round nodes of the prompt.")
    p.add_argument("--adv_norm", default="std", choices=["std", "mean"])
    p.add_argument("--adv_std_floor", type=float, default=0.0,
                   help="With --adv_norm std, divide by max(group std, this). Groups whose "
                        "reward spread is below the floor (e.g. re-decode noise, sd ~0.04) "
                        "keep proportionally small advantages instead of unit scale.")
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--reflect_token_weight", type=float, default=1.0)
    p.add_argument("--num_ppo_epochs", type=int, default=1)
    p.add_argument("--train_micro_batch", type=int, default=2)
    # Optim
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", choices=["cosine", "constant"], default="cosine",
                   help="LR after warmup: cosine decay to 0 at max_steps, or held constant.")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=500)
    # Eval / IO
    p.add_argument("--output_dir", required=True)
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--eval_on_start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_max_prompts", type=int, default=None)
    p.add_argument("--eval_branch", default=None,
                   help="Fan-out per round for validation, e.g. '1,1,1,1' for a 3-round chain. "
                        "Default: a chain as deep as --branch. Rounds beyond --branch are "
                        "reported as val/am_k etc., so am can be tracked past the trained depth.")
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    p.add_argument("--run_name", default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_seed", type=int, default=None,
                   help="Seed of the training-prompt shuffle only (default: --seed). Resumed jobs "
                        "must keep the same value to continue the same prompt stream.")
    # Optional decoded-image logging (needs the visual de-tokenizer weights).
    p.add_argument("--log_images", type=int, default=0)
    p.add_argument("--ar_path", default=None)
    p.add_argument("--encoder_path", default=None)
    p.add_argument("--decoder_path", default=None)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    return p.parse_args()


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
    model.to(device)
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
            json.dump({"step": step}, f)
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
# Reward + advantages over a rollout tree
# ---------------------------------------------------------------------------

def descendant_leaves(n: Node) -> List[Node]:
    if not n.children:
        return [n]
    return [leaf for c in n.children for leaf in descendant_leaves(c)]


@torch.no_grad()
def score_nodes(nodes: List[Node], prompts: List[Dict], reward):
    """Score the images of ``nodes`` that have no score yet (each image once)."""
    todo = [n for n in nodes if n.per_question is None]
    if not todo:
        return
    am, gm, pq = reward.score_images(
        [n.codes for n in todo], [prompts[n.prompt_idx]["vqa_list"] for n in todo])
    for n, a, g, q in zip(todo, am, gm, pq):
        n.am, n.gm, n.per_question = a, g, list(q)


def refine_reward(n: Node, reward, fail_threshold: float, break_weight: float) -> float:
    """score(child) - score(parent), where score = alpha * AM + (1 - alpha) * GM.

    With ``break_weight != 1`` the AM delta is taken per question with the
    parent's passed questions up-weighted so collateral damage costs more; the
    GM delta (no per-question split) stays unweighted, and ``alpha`` balances
    the two exactly as in ``reward.combine``. At ``break_weight == 1`` this is
    identically ``combine(child) - combine(parent)``.
    """
    if break_weight == 1.0 or not n.per_question or not n.parent.per_question:
        return reward.combine(n.am, n.gm) - reward.combine(n.parent.am, n.parent.gm)
    am_delta = 0.0
    for pc, pp in zip(n.per_question, n.parent.per_question):
        am_delta += (break_weight if pp >= fail_threshold else 1.0) * (pc - pp)
    am_delta /= len(n.per_question)
    gm_delta = n.gm - n.parent.gm
    alpha = reward.cfg.alpha
    return alpha * am_delta + (1.0 - alpha) * gm_delta


def make_critique_fn(reward, fail_threshold: float, source: str = "oracle",
                     oracle_frac: float = 1.0, resample_tries: int = 3):
    """``critique_fn`` for TreeRollout.run: scores the parents and returns one
    frozen ``Critique`` per parent (None when the parent fails nothing). The
    source / fallback flag travel with the Critique and land on the children
    (the parent node is not touched, so a round-k node keeps the source it was
    itself refined under when it later becomes a parent).

    source="oracle": the oracle critique of the failed questions.
    source="hybrid": with prob ``oracle_frac`` the oracle critique, otherwise a
    single reflection sampled from the policy (``sample_fn``) and shared by all
    children; "looks good" reflections are re-sampled ``resample_tries`` times,
    then the oracle critique is used (``Critique.fallback``). Note the retries
    only help when the rollout samples reflections (``reflect_sample=True``);
    with greedy decoding every retry returns the same text (the rollout warns).
    """
    assert source in ("oracle", "hybrid")

    def oracle_text(n: Node, prompts: List[Dict]) -> str:
        p = prompts[n.prompt_idx]
        return oracle_critique(p["vqa_list"], p["skills"], n.per_question, fail_threshold)

    def fn(parents: List[Node], prompts: List[Dict], sample_fn=None) -> List[Optional[Critique]]:
        score_nodes(parents, prompts, reward)
        texts: List[Optional[Critique]] = [None] * len(parents)
        todo = [i for i, n in enumerate(parents)
                if not all(q >= fail_threshold for q in n.per_question)]
        policy_idx = []
        if source == "hybrid":
            # One coin per parent; the trainer seeds torch per (step, rank).
            coins = torch.rand(len(todo)).tolist()
            policy_idx = [i for i, c in zip(todo, coins) if c >= oracle_frac]
        pending = list(policy_idx)
        for _ in range(resample_tries + 1):
            if not pending:
                break
            got = sample_fn([parents[i] for i in pending])
            for i, t in zip(pending, got):
                if t and not LOOKS_GOOD_RE.search(t):
                    texts[i] = Critique(t, source="policy")
            pending = [i for i in pending if texts[i] is None]
        for i in todo:
            if texts[i] is None:
                texts[i] = Critique(oracle_text(parents[i], prompts), source="oracle",
                                    fallback=i in policy_idx)
        return texts
    return fn


@torch.no_grad()
def score_tree(batch: RolloutBatch, reward, draft_reward: str = "own",
               stop_penalty: float = 0.0, stop_threshold: float = 1.0,
               fail_threshold: float = 0.5, break_weight: float = 1.0):
    """Fill node.am / node.gm / node.per_question / node.reward for every node
    (each image scored once; nodes already scored mid-rollout by the oracle
    critique_fn are not re-scored).

    ``reward`` is a TarLatentVQAReward or a PixelVQAReward; both expose the same
    ``score_images`` / ``combine`` API.

    Rewards (see the module docstring): a refined child gets its score minus its
    parent's (the parent score is the within-group baseline; see refine_reward
    for ``break_weight``); a "looks good" child gets
    ``-stop_penalty * max(0, stop_threshold - parent_score)``; a draft gets its
    own score (``draft_reward="own"``) or the mean final score of its leaves
    (``"children"``). Intermediate rounds of deeper trees keep the
    parent-relative delta.
    """
    score_nodes([n for level in batch.nodes_by_round for n in level
                 if n.round == 0 or not n.looks_good], batch.prompts, reward)
    for level in batch.nodes_by_round:
        for n in level:
            if n.round == 0:
                n.reward = reward.combine(n.am, n.gm)
            elif n.looks_good:
                n.am, n.gm, n.per_question = n.parent.am, n.parent.gm, n.parent.per_question
                parent_score = reward.combine(n.parent.am, n.parent.gm)
                n.reward = -stop_penalty * max(0.0, stop_threshold - parent_score)
            else:
                n.reward = refine_reward(n, reward, fail_threshold, break_weight)
    if draft_reward == "children":
        # Leaves' am/gm are final at this point (inherited for "looks good").
        for root in batch.roots:
            leaves = descendant_leaves(root)
            root.reward = sum(reward.combine(l.am, l.gm) for l in leaves) / len(leaves)
    elif draft_reward != "own":
        raise ValueError(f"Unknown draft_reward {draft_reward}")


def assign_advantages(batch: RolloutBatch, args, gcfg: GRPOConfig):
    zero_var = 0
    groups_total = 0
    for level in batch.nodes_by_round:
        if not level:
            continue
        if level[0].round == 0 or args.group == "prompt":
            keys = [n.prompt_idx for n in level]
        else:
            keys = [id(n.parent) for n in level]
        adv, zv = normalize_groups([n.reward for n in level], keys, gcfg)
        for n, a in zip(level, adv):
            n.adv = a
        zero_var += zv
        groups_total += len(set(keys))
    return zero_var, groups_total


def tree_stats(batch: RolloutBatch, prefix: str, num_rounds: int,
               fail_threshold: float = 0.5) -> Dict[str, float]:
    """Sums (not means) so they can be all-reduced; '*_n' carries counts.

    Emits the same key set on every rank (all rounds, even empty ones) so the
    all-reduce cannot desynchronise.

    fix / brk (refined children only): over the questions the parent failed /
    passed at ``fail_threshold``, how many the child passes / fails. A refine
    that executes its critique has a high fix rate and a low break rate. Both
    are also split by question skill (``fix_count_1``) and, for frozen
    critiques, by source (``fix_oracle_1`` / ``fix_policy_1``).
    pass_<skill> (every round, and _final): fraction of that skill's questions
    the image passes.
    """
    s: Dict[str, float] = {}

    def skill_of(n: Node, qi: int) -> str:
        sk = batch.prompts[n.prompt_idx]["skills"][qi]
        return sk if sk in STAT_SKILLS else "other"

    def rate(key: str, num: int, den: int):
        s[f"{prefix}{key}_num"] = s.get(f"{prefix}{key}_num", 0.0) + num
        s[f"{prefix}{key}_den"] = s.get(f"{prefix}{key}_den", 0.0) + den

    def pass_rates(tag: str, nodes: List[Node]):
        for sk in STAT_SKILLS:
            rate(f"pass_{sk}_{tag}", 0, 0)        # fixed key set on every rank
        for n in nodes:
            if n.per_question is None:
                continue
            for qi, p in enumerate(n.per_question):
                rate(f"pass_{skill_of(n, qi)}_{tag}", p >= fail_threshold, 1)

    for r in range(num_rounds + 1):
        level = batch.nodes_by_round[r] if r < len(batch.nodes_by_round) else []
        s[f"{prefix}am_{r}"] = sum(n.am for n in level)
        s[f"{prefix}gm_{r}"] = sum(n.gm for n in level)
        s[f"{prefix}reward_{r}"] = sum(n.reward for n in level)
        s[f"{prefix}n_{r}"] = len(level)
        pass_rates(str(r), level)
        if r > 0:
            s[f"{prefix}looks_good_{r}"] = sum(n.looks_good for n in level)
            s[f"{prefix}forced_refine_{r}"] = sum(n.forced_refine for n in level)
            s[f"{prefix}reflect_len_{r}"] = sum(n.reflection_len for n in level)
            deltas = [n.am - n.parent.am for n in level]
            s[f"{prefix}delta_am_{r}"] = sum(deltas)
            s[f"{prefix}improved_{r}"] = sum(d > 0.05 for d in deltas)
            s[f"{prefix}degraded_{r}"] = sum(d < -0.05 for d in deltas)
            for src in CRITIQUE_SOURCES:
                s[f"{prefix}critique_{src}_{r}"] = sum(n.critique_source == src for n in level)
            s[f"{prefix}critique_fallback_{r}"] = sum(n.critique_fallback for n in level)
            keys = ([f"{kind}_{r}" for kind in ("fix", "brk")]
                    + [f"{kind}_{sk}_{r}" for kind in ("fix", "brk") for sk in STAT_SKILLS]
                    + [f"{kind}_{src}_{r}" for kind in ("fix", "brk") for src in CRITIQUE_SOURCES])
            for k in keys:
                rate(k, 0, 0)
            for n in level:
                if n.looks_good or n.per_question is None or n.parent.per_question is None:
                    continue
                src = n.critique_source if n.critique_source in CRITIQUE_SOURCES else None
                for qi, (pc, pp) in enumerate(zip(n.per_question, n.parent.per_question)):
                    if pp < fail_threshold:
                        kind, hit = "fix", pc >= fail_threshold
                    else:
                        kind, hit = "brk", pc < fail_threshold
                    rate(f"{kind}_{r}", hit, 1)
                    rate(f"{kind}_{skill_of(n, qi)}_{r}", hit, 1)
                    if src:
                        rate(f"{kind}_{src}_{r}", hit, 1)
    # Final-image score per leaf (what the benchmark measures).
    s[f"{prefix}am_final"] = sum(n.am for n in batch.leaves)
    s[f"{prefix}n_final"] = len(batch.leaves)
    pass_rates("final", batch.leaves)
    return s


def finalize_stats(s: Dict[str, float], prefix: str) -> Dict[str, float]:
    out = {}
    rounds = sorted({int(k.rsplit("_", 1)[1]) for k in s
                     if k.startswith(f"{prefix}n_") and k.rsplit("_", 1)[1].isdigit()})
    for r in rounds:
        n = max(s[f"{prefix}n_{r}"], 1.0)
        out[f"{prefix}am_{r}"] = s[f"{prefix}am_{r}"] / n
        out[f"{prefix}gm_{r}"] = s[f"{prefix}gm_{r}"] / n
        out[f"{prefix}reward_{r}"] = s[f"{prefix}reward_{r}"] / n
        if r > 0:
            out[f"{prefix}looks_good_{r}"] = s[f"{prefix}looks_good_{r}"] / n
            out[f"{prefix}forced_refine_{r}"] = s[f"{prefix}forced_refine_{r}"] / n
            out[f"{prefix}reflect_len_{r}"] = s[f"{prefix}reflect_len_{r}"] / n
            out[f"{prefix}delta_am_{r}"] = s[f"{prefix}delta_am_{r}"] / n
            out[f"{prefix}improved_{r}"] = s[f"{prefix}improved_{r}"] / n
            out[f"{prefix}degraded_{r}"] = s[f"{prefix}degraded_{r}"] / n
            for src in CRITIQUE_SOURCES:
                out[f"{prefix}critique_{src}_{r}"] = s[f"{prefix}critique_{src}_{r}"] / n
            out[f"{prefix}critique_fallback_{r}"] = s[f"{prefix}critique_fallback_{r}"] / n
    out[f"{prefix}am_final"] = s[f"{prefix}am_final"] / max(s[f"{prefix}n_final"], 1.0)
    # Ratio stats (fix_*, brk_*, pass_*): "<key>_num" / "<key>_den". A bucket
    # with no questions is logged as 0, so the wandb key set stays fixed.
    for k in s:
        if k.startswith(prefix) and k.endswith("_num"):
            key = k[len(prefix):-4]
            out[f"{prefix}{key}"] = s[k] / max(s[f"{prefix}{key}_den"], 1.0)
    return out


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_on_tree(model, rollout: TreeRollout, batch: RolloutBatch, args, gcfg: GRPOConfig,
                  optimizer, scheduler, device) -> Dict[str, float]:
    img_start, img_end = rollout.img_start, rollout.img_end
    leaves = sorted(batch.leaves, key=lambda n: len(n.seq))
    micro = [leaves[i:i + args.train_micro_batch] for i in range(0, len(leaves), args.train_micro_batch)]
    rows = [rollout.build_training_rows(m, args.reflect_token_weight) for m in micro]
    denom = max(float(sum(r["weight"].sum() for r in rows)), 1.0)
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
def run_validation(model, rollout: TreeRollout, reward, val_rows, args, device, step, decoder=None,
                   critique_fn=None, prefix: str = "val/"):
    """Greedy-critic validation of the loop as deployed; with ``critique_fn``
    (oracle critiques) it measures the refiner alone, under ``prefix``."""
    if not val_rows:
        return {}
    torch.manual_seed(args.seed * 7919 + 17)
    branch = args.eval_branch_list or [1] * len(rollout.cfg.branch)
    num_rounds = len(branch) - 1
    saved = (rollout.cfg.reflect_sample, rollout.cfg.min_refines)
    rollout.cfg.reflect_sample = False   # greedy reflection, like the eval script
    rollout.cfg.min_refines = 0          # the policy decides when to stop, as in eval
    sums: Dict[str, float] = {}
    samples = []
    bs = max(args.gen_batch_size, 1)
    for start in range(0, len(val_rows), bs):
        batch = rollout.run(model, val_rows[start:start + bs], branch=branch, critique_fn=critique_fn)
        score_tree(batch, reward, args.draft_reward, args.stop_penalty, args.stop_threshold,
                   args.oracle_fail_threshold, args.break_weight)
        for k, v in tree_stats(batch, prefix, num_rounds, args.oracle_fail_threshold).items():
            sums[k] = sums.get(k, 0.0) + v
        if decoder is not None and is_main() and len(samples) < args.log_images:
            for leaf in batch.leaves[:args.log_images - len(samples)]:
                samples.append((leaf, batch.prompts[leaf.prompt_idx]["prompt"]))
    rollout.cfg.reflect_sample, rollout.cfg.min_refines = saved
    sums = reduce_sum(sums, device)
    out = finalize_stats(sums, prefix)
    if decoder is not None and is_main() and samples:
        out[f"{prefix}images"] = decoder.wandb_images([l for l, _ in samples],
                                                      prompts=[p for _, p in samples])
    return out


class ImageDecoder:
    """Decodes image codes to pixels, for the pixel reward and/or wandb logging."""

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
        self.decode_batch = max(1, args.reward_decode_batch)

    @torch.inference_mode()
    def decode_pils(self, codes_list):
        """list of equal-length code lists -> list of PIL images."""
        from PIL import Image
        out = []
        for start in range(0, len(codes_list), self.decode_batch):
            chunk = codes_list[start:start + self.decode_batch]
            codes = torch.tensor([list(c) for c in chunk], dtype=torch.long, device=self.device)
            imgs = self.tok.decode_from_encoder_indices(codes, {"cfg_scale": self.cfg_scale})
            out.extend(Image.fromarray(im.numpy()) for im in imgs)
        return out

    @torch.inference_mode()
    def wandb_images(self, leaves: List[Node], prompts: Optional[List[str]] = None):
        import wandb
        from PIL import Image
        out = []
        for idx, leaf in enumerate(leaves):
            chain = leaf.ancestors()
            pils = self.decode_pils([n.codes for n in chain])
            w, h = pils[0].size
            canvas = Image.new("RGB", (w * len(pils), h))
            for i, im in enumerate(pils):
                canvas.paste(im, (i * w, 0))
            caption = " | ".join(
                ([f"prompt: {prompts[idx]}"] if prompts else []) +
                [f"AM0={chain[0].am:.2f}"] +
                [f"r{n.round}: {n.reflection} -> AM={n.am:.2f}" for n in chain[1:]])
            out.append(wandb.Image(canvas, caption=caption))
        return out


# ---------------------------------------------------------------------------
# Reward construction
# ---------------------------------------------------------------------------

def build_latent_reward(args, model, base, tokenizer, image_start_id,
                        num_image_tokens, device) -> TarLatentVQAReward:
    """Reward that answers the VQA questions on the image *tokens* with a Tar LM."""
    if args.reward_model_name_or_path:
        # Separate frozen reward model: one extra bf16 copy of the weights per GPU.
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name_or_path)
        if reward_tokenizer.pad_token is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_model = Qwen2ForCausalLM.from_pretrained(
            args.reward_model_name_or_path, torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation).to(device).eval()
        reward_model.config.use_cache = False
        for rp in reward_model.parameters():
            rp.requires_grad_(False)
        reward_image_start_id = reward_tokenizer.convert_tokens_to_ids("<I0>")
        assert reward_image_start_id is not None and \
            reward_image_start_id != reward_tokenizer.unk_token_id, \
            "reward model tokenizer has no <I0> image vocab"
        n_reward_image_tokens = sum(1 for t in reward_tokenizer.get_vocab()
                                    if t.startswith("<I") and t[2:-1].isdigit())
        assert n_reward_image_tokens == num_image_tokens, (
            f"image vocab mismatch: policy {num_image_tokens} vs "
            f"reward {n_reward_image_tokens}")
        rank0_print(f"reward model: {args.reward_model_name_or_path} "
                    f"(image start={reward_image_start_id})")
        reward_lm_head = reward_model.lm_head

        def forward_hidden(input_ids, attention_mask):
            return reward_model.model(input_ids=input_ids, attention_mask=attention_mask,
                                      use_cache=False, return_dict=True).last_hidden_state
    else:
        reward_tokenizer = tokenizer
        reward_image_start_id = image_start_id
        reward_lm_head = base.lm_head

        def forward_hidden(input_ids, attention_mask):
            with model.disable_adapter():
                return base.model(input_ids=input_ids, attention_mask=attention_mask,
                                  use_cache=False, return_dict=True).last_hidden_state

    return TarLatentVQAReward(
        reward_tokenizer, forward_hidden, reward_lm_head, device,
        RewardConfig(batch_size=args.reward_batch_size, answer_suffix=args.answer_suffix,
                     scale=args.scale, alpha=args.alpha),
        reward_image_start_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    branch = [int(x) for x in args.branch.split(",") if x.strip()]
    assert len(branch) >= 1 and all(b >= 1 for b in branch), "--branch must be positive ints"
    if branch[0] < 2:
        rank0_print("WARNING: branch[0] < 2 gives zero advantage for every draft.")
    args.eval_branch_list = None
    if args.eval_branch:
        eb = [int(x) for x in args.eval_branch.split(",") if x.strip()]
        assert len(eb) >= 1 and all(b >= 1 for b in eb), "--eval_branch must be positive ints"
        args.eval_branch_list = eb
    rank0_print(f"branch: train={branch} eval={args.eval_branch_list or [1] * len(branch)}")

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
    model = load_policy(args, device, adapter_dir=resume_dir)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)
    total_updates = args.max_steps * args.num_ppo_epochs
    warmup_updates = int(args.warmup_ratio * total_updates)
    if args.lr_scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_updates)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_updates, total_updates)
    start_step = 0
    if resume_dir is not None:
        optimizer.load_state_dict(torch.load(os.path.join(resume_dir, "optimizer.pt"), map_location=device))
        scheduler.load_state_dict(torch.load(os.path.join(resume_dir, "scheduler.pt")))
        with open(os.path.join(resume_dir, "state.json")) as f:
            start_step = int(json.load(f)["step"])
        rank0_print(f"Resumed from {resume_dir} at step {start_step}")

    rcfg = RolloutConfig(
        branch=branch, scale=args.scale, gen_seq_len=args.gen_seq_len,
        img_temperature=args.img_temperature, img_top_k=args.img_top_k, img_top_p=args.img_top_p,
        reflect_tokens=args.reflect_tokens, reflect_sample=True,
        reflect_temperature=args.reflect_temperature, reflect_top_k=args.reflect_top_k,
        reflect_top_p=args.reflect_top_p, gen_batch_size=args.gen_batch_size,
        min_refines=args.min_refines, refine_resample_tries=args.refine_resample_tries)
    rollout = TreeRollout(tokenizer, rcfg, image_start_id, num_image_tokens, device)
    if len(branch) > 1 and args.min_refines > branch[1]:
        rank0_print(f"WARNING: --min_refines {args.min_refines} > branch[1] {branch[1]}; "
                    f"every child is forced to refine.")
    rank0_print(f"reward: draft_reward={args.draft_reward} stop_penalty={args.stop_penalty} "
                f"stop_threshold={args.stop_threshold} min_refines={args.min_refines} "
                f"critique_source={args.critique_source} "
                f"hybrid_oracle_frac={args.hybrid_oracle_frac} break_weight={args.break_weight} "
                f"fail_threshold={args.oracle_fail_threshold}")

    base = model.get_base_model()

    # The visual de-tokenizer: on every rank for the pixel reward (it scores
    # decoded PNGs), on rank 0 for decoded-image logging. One instance serves both.
    decoder = None
    if args.reward_kind == "pixel" or (args.log_images > 0 and is_main()):
        assert args.ar_path and args.encoder_path and args.decoder_path, \
            "--reward_kind pixel / --log_images need the de-tokenizer paths"
        decoder = ImageDecoder(args, device)

    if args.reward_kind == "pixel":
        assert args.reward_server_url, "--reward_kind pixel needs --reward_server_url"
        assert not args.reward_model_name_or_path, (
            "--reward_kind pixel scores with the served judge; "
            "--reward_model_name_or_path only applies to the latent reward")
        reward = PixelVQAReward(
            decoder.decode_pils, args.reward_server_url,
            RewardConfig(batch_size=args.reward_batch_size, answer_suffix=args.answer_suffix,
                         scale=args.scale, alpha=args.alpha),
            images_per_request=args.reward_images_per_request)
        info = reward.health()      # fail fast (per rank) if the judge is not up
        rank0_print(f"pixel reward judge: {info.get('model')} at {args.reward_server_url} "
                    f"(answer_suffix={info.get('answer_suffix')})")
    else:
        reward = build_latent_reward(args, model, base, tokenizer, image_start_id,
                                     num_image_tokens, device)
    gcfg = GRPOConfig(clip_eps=args.clip_eps, kl_coef=args.kl_coef, adv_norm=args.adv_norm,
                      adv_std_floor=args.adv_std_floor,
                      reflect_token_weight=args.reflect_token_weight)

    dataset_seed = args.seed if args.dataset_seed is None else args.dataset_seed
    train_ds = GenEval2PromptDataset(args.data_path, seed=dataset_seed, rank=rank,
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

    def log(metrics: Dict, step: int, wandb_step: Optional[int] = None):
        if is_main():
            printable = {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in metrics.items() if not k.endswith("images")}
            print(f"step={step} {json.dumps(printable)}", flush=True)
            if use_wandb:
                import wandb
                wandb.log(metrics, step=step if wandb_step is None else wandb_step)

    # Frozen critiques for training (--critique_source oracle | hybrid); the
    # extra validation pass (--eval_oracle) always uses the pure oracle.
    oracle_fn = None
    if args.critique_source != "policy" or args.eval_oracle:
        oracle_fn = make_critique_fn(reward, args.oracle_fail_threshold, "oracle")
    train_critique_fn = None
    if args.critique_source == "oracle":
        train_critique_fn = oracle_fn
    elif args.critique_source == "hybrid":
        train_critique_fn = make_critique_fn(reward, args.oracle_fail_threshold, "hybrid",
                                             args.hybrid_oracle_frac, args.refine_resample_tries)

    def validate(step):
        # Always the loop as deployed (policy critic, greedy); optionally the
        # refiner alone under oracle critiques.
        out = run_validation(model, rollout, reward, val_rows, args, device, step, decoder)
        if args.eval_oracle:
            out.update(run_validation(model, rollout, reward, val_rows, args, device, step, None,
                                      critique_fn=oracle_fn, prefix="val_oracle/"))
        return out

    # A marker per finished validation, so a job killed mid-validation (e.g. at a
    # time limit) has the next job redo that validation after resuming.
    def eval_marker(step):
        return os.path.join(args.output_dir, f"eval_done_{step}")

    def validate_and_log(step, wandb_step=None):
        log(validate(step), step, wandb_step)
        if is_main():
            open(eval_marker(step), "w").close()

    if args.eval_on_start and start_step == 0 and val_rows:
        validate_and_log(0)
    elif (start_step > 0 and val_rows and args.eval_steps and start_step % args.eval_steps == 0
          and not os.path.exists(eval_marker(start_step))):
        rank0_print(f"Validation at step {start_step} did not finish before the last job ended; redoing it.")
        # wandb already moved past start_step in the previous job, so this lands on the next step.
        validate_and_log(start_step, wandb_step=start_step + 1)

    prompt_iter = train_ds.iterate(args.prompts_per_gpu, skip_batches=start_step)
    for step in range(start_step + 1, args.max_steps + 1):
        prompts = next(prompt_iter)
        torch.manual_seed(args.seed * 1_000_003 + step * 7919 + rank)
        t0 = time.time()
        # With oracle critiques the drafts are scored inside the rollout, so
        # time/rollout then includes that part of the reward cost.
        batch = rollout.run(model, prompts, critique_fn=train_critique_fn)
        t1 = time.time()
        model.eval()
        score_tree(batch, reward, args.draft_reward, args.stop_penalty, args.stop_threshold,
                   args.oracle_fail_threshold, args.break_weight)
        zero_var, n_groups = assign_advantages(batch, args, gcfg)
        t2 = time.time()
        train_metrics = train_on_tree(model, rollout, batch, args, gcfg, optimizer, scheduler, device)
        t3 = time.time()

        if step % args.logging_steps == 0:
            sums = tree_stats(batch, "train/", rollout.cfg.num_rounds, args.oracle_fail_threshold)
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
            validate_and_log(step)

    if args.max_steps % max(args.save_steps, 1) != 0:
        save_checkpoint(model, optimizer, scheduler, args.max_steps, args, device)
    rank0_print("Done.")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
