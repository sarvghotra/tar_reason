"""CPU test for chain-mode training rollouts (--chain_siblings): each draft grows a
single chain up to a random round k*, gets B siblings at k*, and stops there.
Fakes the sampler and the judge; only the tokenizer is real.

    python util_scripts/debug/test_chain_mode.py [tokenizer_dir]
"""

import collections
import random
import sys
import types

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from llava.train.rl import train_grpo as tg  # noqa: E402
from llava.train.rl.grpo import GRPOConfig  # noqa: E402
from llava.train.rl.rollout import KIND_IMG, Critique, RolloutConfig, TreeRollout  # noqa: E402

TOK = sys.argv[1] if len(sys.argv) > 1 else "output_dir/slf_ref_edit_t21/checkpoint-17000"
G0, D, B = 2, 4, 4          # drafts per prompt, max refinement rounds, siblings at the branch round


class FakeReward:
    cfg = types.SimpleNamespace(alpha=1.0)

    def score_images(self, codes, vqas):
        # Never all-pass (so chains keep going under the oracle critique), random otherwise.
        pq = [[random.uniform(0.0, 0.45)] + [random.random() for _ in v[1:]] for v in vqas]
        return [sum(q) / len(q) for q in pq], [0.0] * len(pq), pq

    def combine(self, am, gm):
        return am


def make_rollout(tok):
    img0 = tok.convert_tokens_to_ids("<I0>")
    ro = TreeRollout(tok, RolloutConfig(branch=[G0] + [1] * D, gen_batch_size=8, chain_siblings=B),
                     img0, 65536, torch.device("cpu"))
    ro._sample_images = lambda model, rows: [
        [img0 + random.randrange(65536) for _ in range(729)] for _ in rows]
    ro._sample_reflections = lambda model, rows: [
        tok("the cat is not red\nCorrection: make the cat red", add_special_tokens=False).input_ids for _ in rows]
    return ro


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    tok.padding_side = "left"
    prompts = [{"prompt": f"p{i}", "vqa_list": [("Is the cat red?", "Yes"), ("Is the cat running?", "Yes")],
                "skills": ["attribute", "verb"]} for i in range(6)]
    model = types.SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)
    rw = FakeReward()
    crit_fn = tg.make_critique_fn(rw, 0.5, "hybrid", 0.5, 3)
    random.seed(5)
    torch.manual_seed(5)
    ro = make_rollout(tok)

    b = ro.run(model, prompts, critique_fn=crit_fn)
    targets = collections.Counter()
    for root in b.roots:
        k_star = root.branch_round
        assert 1 <= k_star <= D, k_star
        targets[k_star] += 1
        n = root
        for k in range(1, k_star):                 # single chain before k*
            assert len(n.children) == 1, (k_star, k, len(n.children))
            n = n.children[0]
            assert n.round == k and n.branch_round == k_star
        sibs = n.children                           # B siblings at k*
        assert len(sibs) == B and all(s.round == k_star for s in sibs), (k_star, len(sibs))
        assert all(not s.children for s in sibs), "siblings must be leaves"
        assert len({s.reflection for s in sibs}) == 1, "siblings share one critique"
        assert all(s.seq[:s.seg_start] == sibs[0].seq[:sibs[0].seg_start] for s in sibs), "same history"
        assert all(s in b.leaves for s in sibs)
    assert len(b.leaves) == len(b.roots) * B, (len(b.leaves), len(b.roots))
    n_images = sum(len(lv) for lv in b.nodes_by_round)
    print(f"targets per round {dict(sorted(targets.items()))}; images {n_images} for {len(prompts)} prompts "
          f"({n_images / len(prompts):.1f}/prompt; full tree {G0}x{B}^... would be far more)")

    # Rewards / advantages: siblings are a group, lone chain nodes get 0.
    tg.score_tree(b, rw, "children", 0.5, 0.8, 0.5, 2.0)
    args = types.SimpleNamespace(group="parent", images_per_critique=1)
    tg.assign_advantages(b, args, GRPOConfig(adv_norm="std", adv_std_floor=0.04))
    for root in b.roots:
        n = root
        while len(n.children) == 1:
            n = n.children[0]
            assert n.adv == 0.0, ("lone chain node must have advantage 0", n.round, n.adv)
        assert abs(sum(s.adv for s in n.children)) < 1e-6, "sibling advantages are centred"
    st = tg.chain_stats(b, D)
    assert sum(st[f"train/chain_target_{k}"] for k in range(1, D + 1)) == len(b.roots)
    assert sum(st[f"train/chain_groups_{k}"] for k in range(1, D + 1)) == len(b.roots)
    print("chain stats:", {k.split("/")[1]: v for k, v in st.items() if v})

    # Training rows: every sampled image token is trained exactly once across the leaves.
    rows = ro.build_training_rows(b.leaves, 1.0, True, True, 1.0)
    trained_img = int(((rows["pos_kind"] == KIND_IMG) & rows["train_mask"]).sum())
    assert trained_img == n_images * 729, (trained_img, n_images * 729)
    print(f"trained image tokens {trained_img} = {n_images} images x 729 (each once)")

    # MDP training: chain mode with context_mode="mdp", one row per node, own segment only.
    random.seed(9)
    torch.manual_seed(9)
    ro_m = make_rollout(tok)
    ro_m.cfg.context_mode = "mdp"
    bm = ro_m.run(model, prompts, critique_fn=crit_fn)
    tg.score_tree(bm, rw, "children", 0.5, 0.8, 0.5, 2.0)
    tg.assign_advantages(bm, args, GRPOConfig(adv_norm="std", adv_std_floor=0.04))
    nodes = [n for n in bm.all_nodes() if any(k != 0 for k in n.kinds[n.seg_start:])]
    n_img_m = sum(len(lv) for lv in bm.nodes_by_round)
    assert len(nodes) == n_img_m, (len(nodes), n_img_m)
    rows_m = ro_m.build_training_rows(nodes, 1.0, True, True, 1.0, own_segment_only=True)
    trained_m = int(((rows_m["pos_kind"] == KIND_IMG) & rows_m["train_mask"]).sum())
    assert trained_m == n_img_m * 729, (trained_m, n_img_m * 729)
    max_len = int(rows_m["attention_mask"].sum(1).max())
    assert max_len <= len(bm.roots[0].seq) + 800, max_len   # prefix + 2 images + critique, never a full chain
    per_row = (rows_m["train_mask"] & (rows_m["pos_kind"] == KIND_IMG)).sum(1).tolist()
    assert all(c == 729 for c in per_row), set(per_row)
    print(f"mdp training rows: {len(nodes)} rows, each trains exactly one image (729 tokens), max row {max_len} tokens")

    # Evaluation passes an explicit branch: chain mode must not apply.
    e = ro.run(model, prompts[:2], branch=[1] * (D + 1))
    assert all(r.branch_round == 0 for r in e.roots)
    assert [len(lv) for lv in e.nodes_by_round] == [2] * (D + 1), [len(lv) for lv in e.nodes_by_round]
    print("eval rollouts unaffected (plain depth-4 chains)")
    print("OK")


if __name__ == "__main__":
    main()
