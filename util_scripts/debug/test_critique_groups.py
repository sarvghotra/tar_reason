"""CPU smoke test for K critiques x M images rollouts (--images_per_critique)
and the two-level advantages in train_grpo. Fakes the sampler and the judge;
only the tokenizer is real.

    python util_scripts/debug/test_critique_groups.py [tokenizer_dir]
"""

import random
import sys
import types

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from llava.train.rl import train_grpo as tg  # noqa: E402
from llava.train.rl.grpo import GRPOConfig  # noqa: E402
from llava.train.rl.rollout import KIND_IMG, KIND_TXT, RolloutConfig, TreeRollout  # noqa: E402

TOK = sys.argv[1] if len(sys.argv) > 1 else "output_dir/slf_ref_edit_t21/checkpoint-17000"
K, M = 3, 4


class FakeReward:
    cfg = types.SimpleNamespace(alpha=1.0)

    def score_images(self, codes, vqas):
        pq = [[random.random() for _ in v] for v in vqas]
        return [sum(q) / len(q) for q in pq], [0.0] * len(pq), pq

    def combine(self, am, gm):
        return am


def make_rollout(tok, M):
    img0 = tok.convert_tokens_to_ids("<I0>")
    ro = TreeRollout(tok, RolloutConfig(branch=[1, K], gen_batch_size=8, images_per_critique=M),
                     img0, 65536, torch.device("cpu"))
    ro._sample_images = lambda model, rows: [
        [img0 + random.randrange(65536) for _ in range(729)] for _ in rows]
    looks_good = tok("no issues, it matches the prompt.\nCorrection: looks good.",
                     add_special_tokens=False).input_ids
    bads = [tok(t, add_special_tokens=False).input_ids for t in (
        "the cat is not red\nCorrection: make the cat red",
        "there are two cats, not three\nCorrection: add one more cat",
        "the cat is not running\nCorrection: show the cat running")]
    ro._sample_reflections = lambda model, rows: [
        looks_good if random.random() < 0.25 else random.choice(bads) for _ in rows]
    return ro


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    tok.padding_side = "left"
    prompts = [{"prompt": f"p{i}",
                "vqa_list": [("How many cats are in the image?", "two"),
                             ("Is the cat red?", "Yes"), ("Is the cat running?", "Yes")],
                "skills": ["count", "attribute", "verb"]} for i in range(5)]
    model = types.SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)
    args = types.SimpleNamespace(group="parent", images_per_critique=M)
    gcfg = GRPOConfig(adv_norm="std", adv_std_floor=0.04)
    rw = FakeReward()
    random.seed(3)
    torch.manual_seed(3)

    ro = make_rollout(tok, M)
    b = ro.run(model, prompts)
    kids = b.nodes_by_round[1]
    # Structure: per parent, K critiques; a refining critique has M nodes, a
    # "looks good" one is a single leaf.
    for root in b.roots:
        by_c = {}
        for c in root.children:
            by_c.setdefault(c.critique_idx, []).append(c)
        assert set(by_c) == set(range(K)), by_c.keys()
        for group in by_c.values():
            lead, rest = group[0], group[1:]
            assert lead.critique_leader and all(not f.critique_leader for f in rest)
            assert len(group) == (1 if lead.looks_good else M), (len(group), lead.looks_good)
            assert all(f.reflection == lead.reflection and f.seg_start == lead.seg_start for f in rest)
            assert KIND_TXT in lead.kinds and all(KIND_TXT not in f.kinds for f in rest)
            assert all(f.seq[:lead.seg_start + lead.reflection_len]
                       == lead.seq[:lead.seg_start + lead.reflection_len] for f in rest)
            if not lead.looks_good:
                imgs = {tuple(n.codes) for n in group}
                assert len(imgs) == M, "followers must sample their own image"
        assert root.children == [c for c in kids if c.parent is root]
    n_lg = sum(c.looks_good for c in kids)
    print(f"children: {len(kids)} over {len(b.roots)} parents, looks_good leaves: {n_lg}, "
          f"refining: {len(kids) - n_lg} (= M x {(len(kids) - n_lg) // M} critiques)")
    assert set(b.leaves) == set(kids)

    # Rewards + two-level advantages.
    tg.score_tree(b, rw, "own", 0.0, 1.0, 0.5, 2.0)
    counts = tg.assign_advantages(b, args, gcfg)
    print("adv counts:", counts)
    assert counts["groups"] == len(b.roots) + sum(len({c.critique_idx for c in r.children}) for r in b.roots)
    assert counts["groups_crit"] == len(b.roots)
    assert all(r.adv == 0.0 and r.adv_txt == 0.0 for r in b.roots), "drafts (group of 1) untrained"
    for root in b.roots:
        by_c = {}
        for c in root.children:
            by_c.setdefault(c.critique_idx, []).append(c)
        crit_adv = {}
        for ci, group in by_c.items():
            # Image advantages are centred within the critique; a single leaf gets 0.
            if len(group) > 1:
                assert abs(sum(n.adv for n in group)) < 1e-6
            else:
                assert group[0].adv == 0.0
            assert len({n.adv_txt for n in group}) == 1, "one critique advantage per critique"
            crit_adv[ci] = group[0].adv_txt
            mean_r = sum(n.reward for n in group) / len(group)
            crit_adv[ci] = (crit_adv[ci], mean_r)
        advs = [a for a, _ in crit_adv.values()]
        rewards = [r for _, r in crit_adv.values()]
        if len(set(round(r, 9) for r in rewards)) > 1:
            assert abs(sum(advs)) < 1e-6, advs
            order = sorted(range(K), key=lambda i: rewards[i])
            assert [advs[i] for i in order] == sorted(advs), "critique adv must be monotone in mean reward"

    # Training rows: text positions only on leaders, with adv_txt; images with adv.
    # Drafts excluded (--no-train_drafts); with them, the first-born leader also
    # carries its parent's 729 draft tokens (first-born rule).
    rows_d = ro.build_training_rows(b.leaves, 20.0, train_drafts=True)
    n_img_d = int((rows_d["train_mask"] & (rows_d["pos_kind"] == KIND_IMG)).sum())
    rows = ro.build_training_rows(b.leaves, 20.0, train_drafts=False)
    n_img = int((rows["train_mask"] & (rows["pos_kind"] == KIND_IMG)).sum())
    assert n_img_d - n_img == 729 * len(b.roots), (n_img_d, n_img)
    for i, leaf in enumerate(b.leaves):
        tm, pk, adv, w = rows["train_mask"][i], rows["pos_kind"][i], rows["adv"][i], rows["weight"][i]
        txt = tm & (pk == KIND_TXT)
        img = tm & (pk == KIND_IMG)
        assert txt.any() == leaf.critique_leader, (leaf.critique_leader, txt.any())
        if txt.any():
            assert torch.allclose(adv[txt], torch.full_like(adv[txt], leaf.adv_txt))
            assert (w[txt] == 20.0).all()
        assert img.any() == (not leaf.looks_good)
        if img.any():
            assert torch.allclose(adv[img], torch.full_like(adv[img], leaf.adv))
            assert int(img.sum()) == 729
    n_txt = int((rows["train_mask"] & (rows["pos_kind"] == KIND_TXT)).sum())
    n_img = int((rows["train_mask"] & (rows["pos_kind"] == KIND_IMG)).sum())
    print(f"trained tokens: txt {n_txt} (x20 weight = {20 * n_txt}), img {n_img}")
    # Critic-only: no image positions; followers have nothing to train.
    leaders = [n for n in b.leaves if n.critique_leader]
    rows_c = ro.build_training_rows(leaders, 1.0, train_drafts=False, train_images=False)
    assert not (rows_c["train_mask"] & (rows_c["pos_kind"] == KIND_IMG)).any()
    assert int(rows_c["train_mask"].sum()) == n_txt
    rows_f = ro.build_training_rows([n for n in b.leaves if not n.critique_leader], 1.0,
                                    train_drafts=False, train_images=False)
    assert not rows_f["train_mask"].any()
    print(f"critic-only rows: {len(leaders)} leaders of {len(b.leaves)} leaves, {n_txt} tokens")
    # KL-only anchor: images (and drafts) stay in the mask with advantage 0.
    rows_a = ro.build_training_rows(b.leaves, 20.0, train_drafts=True, train_images=True,
                                    image_adv_scale=0.0)
    img_a = rows_a["train_mask"] & (rows_a["pos_kind"] == KIND_IMG)
    txt_a = rows_a["train_mask"] & (rows_a["pos_kind"] == KIND_TXT)
    assert int(img_a.sum()) == n_img_d and (rows_a["adv"][img_a] == 0).all()
    assert torch.equal(rows_a["adv"][txt_a], rows["adv"][rows["train_mask"] & (rows["pos_kind"] == KIND_TXT)])
    print(f"kl-only anchor rows: {int(img_a.sum())} image tokens at adv 0, text adv unchanged")

    # Stats: critique variance split present, key set independent of the batch.
    o = tg.finalize_stats(tg.tree_stats(b, "train/", 1, 0.5), "train/")
    print({k: round(v, 4) for k, v in o.items() if "crit_" in k or k.endswith("looks_good_1")})
    assert o["train/crit_within_var_1"] > 0 and "train/crit_icc_1" in o
    b2 = ro.run(model, prompts[:1])
    tg.score_tree(b2, rw, "own", 0.0, 1.0, 0.5, 2.0)
    o2 = tg.finalize_stats(tg.tree_stats(b2, "train/", 1, 0.5), "train/")
    assert set(o) == set(o2)

    # M = 1: the old single-level path, adv_txt == adv on every node.
    ro1 = make_rollout(tok, 1)
    b1 = ro1.run(model, prompts[:3])
    assert all(len(r.children) == K for r in b1.roots)
    tg.score_tree(b1, rw, "own", 0.0, 1.0, 0.5, 2.0)
    c1 = tg.assign_advantages(b1, types.SimpleNamespace(group="parent", images_per_critique=1), gcfg)
    assert c1["groups_crit"] == 0 and all(n.adv == n.adv_txt for n in b1.nodes_by_round[1])
    print("M=1 path:", c1)
    print("OK")


if __name__ == "__main__":
    main()
