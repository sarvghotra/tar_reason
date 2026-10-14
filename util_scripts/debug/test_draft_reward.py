"""CPU test for --draft_reward children_r1: a draft is rewarded with the mean score
of its round-1 children (a "looks good" child carries the draft's score, a draft
with no child keeps its own). Equals "children" on one-round trees; in chain mode
every draft is judged after one refinement although drafts reach different depths.
Fakes the sampler and the judge; only the tokenizer is real.

    python util_scripts/debug/test_draft_reward.py [tokenizer_dir]
"""

import random
import sys
import types

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from llava.train.rl import train_grpo as tg  # noqa: E402
from llava.train.rl.rollout import RolloutConfig, TreeRollout  # noqa: E402

TOK = sys.argv[1] if len(sys.argv) > 1 else "output_dir/slf_ref_edit_t21/checkpoint-17000"
D, B = 4, 3


class FakeReward:
    cfg = types.SimpleNamespace(alpha=1.0)

    def score_images(self, codes, vqas):
        pq = [[random.uniform(0.6, 1.0) if random.random() < 0.3 else random.uniform(0.0, 0.45)]
              + [random.uniform(0.6, 1.0) for _ in v[1:]] for v in vqas]
        return [sum(q) / len(q) for q in pq], [0.0] * len(pq), pq

    def combine(self, am, gm):
        return am


def make_rollout(tok, branch, chain_siblings):
    img0 = tok.convert_tokens_to_ids("<I0>")
    ro = TreeRollout(tok, RolloutConfig(branch=branch, gen_batch_size=8, chain_siblings=chain_siblings),
                     img0, 65536, torch.device("cpu"))
    ro._sample_images = lambda model, rows: [[img0 + random.randrange(65536) for _ in range(729)] for _ in rows]
    good = tok("no issues, it matches the prompt.\nCorrection: looks good.", add_special_tokens=False).input_ids
    fix = tok("the cat is not red\nCorrection: make the cat red", add_special_tokens=False).input_ids
    ro._sample_reflections = lambda model, rows: [good if random.random() < 0.2 else fix for _ in rows]
    return ro


def rewards(b):
    return [r.reward for r in b.roots]


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    tok.padding_side = "left"
    prompts = [{"prompt": f"p{i}", "vqa_list": [("Is the cat red?", "Yes"), ("Is the cat running?", "Yes")],
                "skills": ["attribute", "verb"]} for i in range(12)]
    model = types.SimpleNamespace(training=True, eval=lambda: None, train=lambda: None)
    rw = FakeReward()

    # One-round tree (t8's BRANCH 2,4): children_r1 == children.
    for seed in range(3):
        out = {}
        for mode in ("children", "children_r1"):
            random.seed(seed)
            torch.manual_seed(seed)
            ro = make_rollout(tok, [2, 4], 0)
            b = ro.run(model, prompts, critique_fn=tg.make_critique_fn(rw, 0.5, "hybrid", 0.5, 3))
            tg.score_tree(b, rw, mode, 0.5, 0.8, 0.5, 2.0)
            out[mode] = rewards(b)
        assert all(abs(x - y) < 1e-9 for x, y in zip(out["children"], out["children_r1"])), seed

    # Chain mode to depth D: children_r1 = mean of round-1 children, own score if none.
    random.seed(5)
    torch.manual_seed(5)
    ro = make_rollout(tok, [2] + [1] * D, B)
    b = ro.run(model, prompts, critique_fn=tg.make_critique_fn(rw, 0.5, "hybrid", 0.5, 3))
    tg.score_tree(b, rw, "children_r1", 0.5, 0.8, 0.5, 2.0)
    n_kids = {}
    deeper = differs = 0
    for r in b.roots:
        if r.children:
            want = sum(c.am for c in r.children) / len(r.children)
            assert all(c.round == 1 for c in r.children)
            for c in r.children:
                if c.looks_good:
                    assert c.am == r.am
        else:
            want = r.am
        assert abs(r.reward - want) < 1e-9, (r.reward, want)
        n_kids[len(r.children)] = n_kids.get(len(r.children), 0) + 1
        leaves = tg.descendant_leaves(r)
        if any(lf.round > 1 for lf in leaves):
            deeper += 1
            final = sum(lf.am for lf in leaves) / len(leaves)
            differs += abs(final - r.reward) > 1e-9
    # Refined nodes keep their parent-relative reward (unchanged by draft_reward).
    for level in b.nodes_by_round[1:]:
        for n in level:
            if not n.looks_good:
                assert n.reward == tg.refine_reward(n, rw, 0.5, 2.0)
    print(f"chain mode: drafts by #round-1 children {dict(sorted(n_kids.items()))}, "
          f"drafts with deeper leaves {deeper} (reward differs from 'children' for {differs})")
    assert deeper > 0 and differs > 0 and n_kids.get(B, 0) > 0 and n_kids.get(1, 0) > 0
    print("OK")


if __name__ == "__main__":
    main()
