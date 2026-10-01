"""CPU smoke test for train_grpo's frozen-critique rollouts (oracle / hybrid)
and the per-skill / per-source stats. Fakes the sampler and the judge; only
the tokenizer is real.

    python util_scripts/debug/test_frozen_critique.py [tokenizer_dir]
"""

import random
import sys
import types

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from llava.train.rl import train_grpo as tg  # noqa: E402
from llava.train.rl.rollout import KIND_TXT, RolloutConfig, TreeRollout  # noqa: E402

TOK = sys.argv[1] if len(sys.argv) > 1 else \
    "output_dir/slf_ref_edit_t20/weights_only/checkpoint-16000"


class FakeReward:
    calls = 0

    def score_images(self, codes, vqas):
        FakeReward.calls += len(codes)
        pq = [[random.random() for _ in v] for v in vqas]
        return [sum(q) / len(q) for q in pq], [0.0] * len(pq), pq

    def combine(self, am, gm):
        return am


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    tok.padding_side = "left"
    img0 = tok.convert_tokens_to_ids("<I0>")
    ro = TreeRollout(tok, RolloutConfig(branch=[1, 4], gen_batch_size=8), img0, 65536,
                     torch.device("cpu"))
    ro._sample_images = lambda model, rows: [
        [img0 + random.randrange(65536) for _ in range(729)] for _ in rows]
    looks_good = tok("no issues, it matches the prompt.\nCorrection: looks good.",
                     add_special_tokens=False).input_ids
    bad = tok("the cat is not red\nCorrection: make the cat red",
              add_special_tokens=False).input_ids
    n_reflect = [0]

    def fake_reflections(model, rows):
        n_reflect[0] += len(rows)
        return [looks_good if random.random() < 0.5 else bad for _ in rows]
    ro._sample_reflections = fake_reflections

    prompts = [{"prompt": f"p{i}",
                "vqa_list": [("How many cats are in the image?", "two"),
                             ("Is the cat red?", "Yes"), ("Is the cat running?", "Yes")],
                "skills": ["count", "attribute", "verb"]} for i in range(6)]
    model = types.SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)
    rw = FakeReward()

    # -- hybrid ---------------------------------------------------------------
    random.seed(7)
    torch.manual_seed(7)
    fn = tg.make_critique_fn(rw, 0.5, "hybrid", 0.5, 3)
    b = ro.run(model, prompts, critique_fn=fn)
    # Sources live on the children (the nodes refined under the critique);
    # the parents are never mutated by critique_fn.
    assert all(n.critique_source == "" and not n.critique_fallback for n in b.roots)
    assert {k.critique_source for k in b.nodes_by_round[1]} == {"oracle", "policy"}, \
        "expected both sources at frac 0.5 with this seed"
    print("per parent:", [({k.critique_source for k in n.children}, {k.critique_fallback for k in n.children})
                          for n in b.roots])
    print("children/parent:", [len(n.children) for n in b.roots], "reflections sampled:", n_reflect[0])
    kids = b.nodes_by_round[1]
    print("child sources:", {s: sum(k.critique_source == s for k in kids) for s in ("oracle", "policy")},
          "fallback:", sum(k.critique_fallback for k in kids))
    assert all(not k.looks_good and k.reflection_len == 0 and KIND_TXT not in k.kinds for k in kids)
    pol = [k for k in kids if k.critique_source == "policy"]
    if pol:
        k = pol[0]
        print("policy critique spliced:", repr(tok.decode(k.seq[k.parent.seg_start + 729:k.seg_start])))
    before = FakeReward.calls
    tg.score_tree(b, rw, "children", 0.5, 0.8, 0.5, 1.0)
    assert FakeReward.calls - before == len(kids), "parents must not be re-scored"
    s = tg.tree_stats(b, "train/", 1, 0.5)
    o = tg.finalize_stats(s, "train/")
    print({k: round(v, 3) for k, v in o.items()
           if any(t in k for t in ("fix", "brk", "critique", "pass_count", "pass_verb"))})

    # -- key set must not depend on the batch (all-reduce safety) -----------
    rw2 = FakeReward()
    rw2.score_images = lambda codes, vqas: ([1.0] * len(codes), [0.0] * len(codes),
                                            [[0.99] * len(v) for v in vqas])
    b2 = ro.run(model, prompts[:2], critique_fn=tg.make_critique_fn(rw2, 0.5, "hybrid", 0.5, 3))
    o2 = tg.finalize_stats(tg.tree_stats(b2, "train/", 1, 0.5), "train/")
    print("all-pass batch rounds:", [len(l) for l in b2.nodes_by_round],
          "same keys:", set(o) == set(o2), "n keys:", len(o))
    assert set(o) == set(o2)

    # -- oracle only ----------------------------------------------------------
    b4 = ro.run(model, prompts[:3], critique_fn=tg.make_critique_fn(rw, 0.5, "oracle"))
    assert {k.critique_source for k in b4.nodes_by_round[1]} <= {"oracle"}
    print("oracle-only sources:", {k.critique_source for k in b4.nodes_by_round[1]})

    # -- sampled critic path unchanged ---------------------------------------
    b3 = ro.run(model, prompts[:2])
    tg.score_tree(b3, rw, "children", 0.5, 0.8, 0.5, 1.0)
    o3 = tg.finalize_stats(tg.tree_stats(b3, "val/", 1, 0.5), "val/")
    print("sampled path sources:", {k.critique_source for k in b3.nodes_by_round[1]},
          "val/fix_1:", round(o3["val/fix_1"], 3), "val/critique_policy_1:", o3["val/critique_policy_1"])
    rows = ro.build_training_rows(b.leaves, 1.0)
    assert not (rows["train_mask"] & (rows["pos_kind"] == KIND_TXT)).any()
    print("OK")


if __name__ == "__main__":
    main()
