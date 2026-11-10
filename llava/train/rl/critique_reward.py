"""Critique / stop reward for end-to-end RL (stage 3, 2026-11-10): a small extra
signal on the model's OWN critique tokens, added to the usual image advantage.

Image tokens and the plain GRPO advantage are unchanged. For a refinement node n
(round >= 1, critique sampled by the policy) whose parent image p was scored by
the judge, let F = p's failed questions (prob < --oracle_fail_threshold) and
P = p's passed ones. The critique reward c(n) is

    p passes everything, n says "looks good"   -> +1
    p passes everything, n asks for edits      -> -1
    p fails something,  n says "looks good"    -> -1
    p fails something,  n asks for edits       -> +1                        (--critique_reward stop, run A)
                                                 recall(F) - beta * fa(P)   (--critique_reward full, run B)

recall(F) = mean over F of P(the critique names that check), fa(P) = the same mean
over P (0 when P is empty); "names" is decided by the judge in a text-only call
(``PixelVQAReward.critique_mentions``). c is normalised within each parent's
siblings exactly like the image reward (same group keys and --adv_norm), and the
critique tokens train with

    adv_txt = adv + lambda * adv_c                     (lambda = --critique_reward_coef)

Nodes alone in their group (chain nodes before the branch round) get adv_c = 0,
so in chain mode the signal comes from the B siblings at each draft's branch
round, where the siblings' decisions (stop vs edit) and critiques differ.
"""

from typing import Dict, List

from llava.train.rl.grpo import GRPOConfig, normalize_groups
from llava.train.rl.rollout import RolloutBatch

CRITIQUE_REWARDS = ("none", "stop", "full")


def _eligible(n) -> bool:
    return (n.round > 0 and not n.teacher_forced and n.critique_leader
            and n.parent is not None and n.parent.per_question is not None)


def assign_critique_rewards(batch: RolloutBatch, args, reward) -> Dict[str, float]:
    """Fill ``node.crit_reward`` for every eligible node that has at least one
    sibling (only those can get a non-zero critique advantage). Returns summed
    statistics for logging (``reduce_sum`` then divide by the matching count)."""
    mode = args.critique_reward
    thr = args.oracle_fail_threshold
    s = {"crit/n": 0.0, "crit/reward_sum": 0.0, "crit/stop_correct": 0.0,
         "crit/n_edit_fail": 0.0, "crit/recall_sum": 0.0, "crit/fa_sum": 0.0}
    groups: Dict[int, List] = {}
    for level in batch.nodes_by_round[1:]:
        for n in level:
            if _eligible(n):
                groups.setdefault(id(n.parent), []).append(n)
    nodes = [n for g in groups.values() if len(g) > 1 for n in g]
    need_mentions = []
    for n in nodes:
        failed = [q < thr for q in n.parent.per_question]
        all_pass = not any(failed)
        if all_pass:
            n.crit_reward = 1.0 if n.looks_good else -1.0
        elif n.looks_good:
            n.crit_reward = -1.0
        else:
            n.crit_reward = 1.0
            if mode == "full":
                need_mentions.append(n)
        s["crit/stop_correct"] += float(all_pass == n.looks_good)
    if need_mentions:
        items = []
        for n in need_mentions:
            p = batch.prompts[n.prompt_idx]
            items.append({"prompt": p["prompt"], "critique": n.reflection,
                          "vqa": [list(qa) for qa in p["vqa_list"]]})
        mentions = reward.critique_mentions(items)
        beta = args.critique_fa_beta
        for n, m in zip(need_mentions, mentions):
            F = [pm for pm, q in zip(m, n.parent.per_question) if q < thr]
            P = [pm for pm, q in zip(m, n.parent.per_question) if q >= thr]
            recall = sum(F) / len(F)
            fa = sum(P) / len(P) if P else 0.0
            n.crit_reward = recall - beta * fa
            n.crit_recall, n.crit_fa = recall, fa
            s["crit/n_edit_fail"] += 1.0
            s["crit/recall_sum"] += recall
            s["crit/fa_sum"] += fa
    for n in nodes:
        s["crit/n"] += 1.0
        s["crit/reward_sum"] += n.crit_reward
    return s


def add_critique_advantages(batch: RolloutBatch, args, gcfg: GRPOConfig) -> Dict[str, float]:
    """adv_txt += lambda * (c normalised within the parent's siblings). Call after
    assign_advantages (which sets adv_txt = adv for --images_per_critique 1)."""
    lam = args.critique_reward_coef
    s = {"crit/groups": 0.0, "crit/zero_var": 0.0, "crit/abs_adv_sum": 0.0, "crit/adv_n": 0.0}
    for level in batch.nodes_by_round[1:]:
        nodes = [n for n in level if _eligible(n) and n.crit_reward is not None]
        if not nodes:
            continue
        keys = [id(n.parent) for n in nodes]
        adv_c, zv = normalize_groups([n.crit_reward for n in nodes], keys, gcfg)
        for n, a in zip(nodes, adv_c):
            n.adv_c = a
            n.adv_txt = n.adv_txt + lam * a
            s["crit/abs_adv_sum"] += abs(a)
            s["crit/adv_n"] += 1.0
        s["crit/groups"] += len(set(keys))
        s["crit/zero_var"] += zv
    return s


def finalize_critique_stats(s: Dict[str, float]) -> Dict[str, float]:
    out = {}
    n = max(s.get("crit/n", 0.0), 1.0)
    out["train/crit_reward"] = s.get("crit/reward_sum", 0.0) / n
    out["train/crit_stop_acc"] = s.get("crit/stop_correct", 0.0) / n
    out["train/crit_nodes"] = s.get("crit/n", 0.0)
    ne = s.get("crit/n_edit_fail", 0.0)
    if ne:
        out["train/crit_recall"] = s["crit/recall_sum"] / ne
        out["train/crit_false_alarm"] = s["crit/fa_sum"] / ne
    if s.get("crit/groups"):
        out["train/crit_frac_zero_adv_groups"] = s["crit/zero_var"] / s["crit/groups"]
    if s.get("crit/adv_n"):
        out["train/crit_abs_adv"] = s["crit/abs_adv_sum"] / s["crit/adv_n"]
    return out
