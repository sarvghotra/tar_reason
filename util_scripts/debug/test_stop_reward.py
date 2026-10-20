"""CPU check of --stop_reward sibling and --critique_break_weight in
train_grpo.score_tree (fake sampler and judge, real tokenizer; same harness as
test_critique_groups.py).

    python util_scripts/debug/test_stop_reward.py [tokenizer_dir]
"""

import random
import sys
import types

import torch

sys.path.insert(0, ".")
from llava.train.rl import train_grpo as tg  # noqa: E402
from llava.train.rl.grpo import GRPOConfig  # noqa: E402
from util_scripts.debug.test_critique_groups import FakeReward, TOK, make_rollout, M  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main():
    random.seed(3)
    torch.manual_seed(3)
    tok = AutoTokenizer.from_pretrained(TOK)
    tok.padding_side = "left"
    prompts = [{"prompt": f"p{i}",
                "vqa_list": [("How many cats are in the image?", "two"),
                             ("Is the cat red?", "Yes"), ("Is the cat running?", "Yes")],
                "skills": ["count", "attribute", "verb"]} for i in range(12)]
    model = types.SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)
    ro = make_rollout(tok, M)
    rw = FakeReward()
    b = ro.run(model, prompts)
    level = b.nodes_by_round[1]
    stops = [n for n in level if n.looks_good]
    assert stops, "no looks-good leaves sampled; re-seed"

    # Baseline: penalty mode, same break weight on both levels.
    tg.score_tree(b, rw, "own", 0.0, 1.0, 0.5, 2.0)
    assert all(n.reward == 0.0 and n.reward_txt == 0.0 for n in stops)
    assert all(n.reward_txt == n.reward for n in level)

    # Sibling mode with critique_break_weight 1 and image break weight 2.
    tg.score_tree(b, rw, "own", 0.0, 1.0, 0.5, 2.0,
                  stop_reward="sibling", critique_break_weight=1.0)
    n_checked = n_zero = 0
    for s in stops:
        sib = [m for m in s.parent.children if not m.looks_good]
        if not sib:
            assert s.reward == 0.0 and s.reward_txt == 0.0
            n_zero += 1
            continue
        exp = -sum(m.reward for m in sib) / len(sib)
        exp_txt = -sum(m.reward_txt for m in sib) / len(sib)
        assert abs(s.reward - exp) < 1e-9, (s.reward, exp)
        assert abs(s.reward_txt - exp_txt) < 1e-9, (s.reward_txt, exp_txt)
        n_checked += 1
    for n in level:
        if n.looks_good:
            continue
        # reward_txt is the plain delta (break weight 1); reward carries the 2x breaks
        plain = rw.combine(n.am, n.gm) - rw.combine(n.parent.am, n.parent.gm)
        assert abs(n.reward_txt - plain) < 1e-9, (n.reward_txt, plain)
        assert abs(n.reward - tg.refine_reward(n, rw, 0.5, 2.0)) < 1e-9
    print(f"stops with refining siblings: {n_checked}, all-stop parents: {n_zero}")

    # The critique-level advantage must read reward_txt; stops are advantaged
    # exactly when their siblings' critique-level mean is negative.
    args = types.SimpleNamespace(group="parent", images_per_critique=M)
    gcfg = GRPOConfig(adv_norm="std", adv_std_floor=0.0)
    tg.assign_advantages(b, args, gcfg)
    for s in stops:
        sib = [m for m in s.parent.children if not m.looks_good]
        if not sib:
            assert s.adv_txt == 0.0
            continue
        crit = {}
        for m in sib:
            crit.setdefault(m.critique_idx, []).append(m.reward_txt)
        means = [sum(v) / len(v) for v in crit.values()]
        n_stop = sum(1 for c in s.parent.children if c.looks_good)
        group = means + [s.reward_txt] * n_stop
        mu = sum(group) / len(group)
        if abs(s.reward_txt - mu) > 1e-9:
            assert (s.adv_txt > 0) == (s.reward_txt > mu), (s.adv_txt, s.reward_txt, mu)
    stats = tg.finalize_stats(tg.tree_stats(b, "t/", 1), "t/")
    assert "t/reward_txt_1" in stats and "t/reward_txt_0" in stats
    print({k: round(v, 4) for k, v in stats.items() if k.startswith("t/reward")})
    print("OK")


if __name__ == "__main__":
    main()
