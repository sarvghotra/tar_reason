"""CPU test for the in-loop rollback verifier (--verdict_coef) and rollback validation (--eval_rollback).

Fakes the sampler, the critic and the judge; the tokenizer is real, and the last part runs one real
train_on_tree step on a tiny random Qwen2 + LoRA so the verdict CE path is exercised end to end.

    python util_scripts/debug/test_rollback.py [tokenizer_dir]
"""

import random
import sys
import types

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from llava.train.rl import train_grpo as tg  # noqa: E402
from llava.train.rl.grpo import KIND_IMG, KIND_VERDICT, GRPOConfig  # noqa: E402
from llava.train.rl.rollout import RolloutConfig, TreeRollout  # noqa: E402

TOK = sys.argv[1] if len(sys.argv) > 1 else "/home/dars11/scratch/models/tar/sft/darshan_pm1_E_merged450"
G0, D, B = 2, 4, 4
MARGIN = 0.02


class FakeReward:
    """Deterministic per image (keyed on its first code): never all-pass, so chains keep going."""
    cfg = types.SimpleNamespace(alpha=1.0)

    def score_images(self, codes, vqas):
        pq = []
        for c, v in zip(codes, vqas):
            rng = random.Random(c[0])
            pq.append([rng.uniform(0.0, 0.45)] + [rng.random() for _ in v[1:]])
        return [sum(q) / len(q) for q in pq], [0.0] * len(pq), pq

    def combine(self, am, gm):
        return am


def make_rollout(tok, chain=B, mdp=True):
    img0 = tok.convert_tokens_to_ids("<I0>")
    ro = TreeRollout(tok, RolloutConfig(branch=[G0] + [1] * D, gen_batch_size=8, chain_siblings=chain,
                                        context_mode="mdp" if mdp else "full"),
                     img0, 65536, torch.device("cpu"))
    ro._sample_images = lambda model, rows: [[img0 + random.randrange(65536) for _ in range(729)] for _ in rows]
    ro._sample_reflections = lambda model, rows: [
        tok("the cat is not red\nCorrection: make the cat red", add_special_tokens=False).input_ids for _ in rows]
    return ro


def check_states(e):
    """Validation rollback: a rejected node is a leaf, the next refinement hangs off its parent, and each
    root's state after round k is its state after k-1 or a kept refinement of it."""
    n_rev = 0
    for root in e.roots:
        assert root.states is not None and len(root.states) == D + 1, len(root.states or [])
        for k in range(1, D + 1):
            prev, cur = root.states[k - 1], root.states[k]
            assert cur is prev or (cur.parent is prev and not cur.reverted), k
    for n in e.all_nodes():
        if n.reverted:
            n_rev += 1
            assert not n.children and n in e.leaves, "a rejected image is never refined"
        for c in n.children:
            assert c.round > n.round
    return n_rev


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    tok.padding_side = "left"
    prompts = [{"prompt": f"p{i}", "vqa_list": [("Is the cat red?", "Yes"), ("Is the cat running?", "Yes")],
                "skills": ["attribute", "verb"]} for i in range(8)]
    model = types.SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)
    rw = FakeReward()
    crit_fn = tg.make_critique_fn(rw, 0.5, "oracle")
    random.seed(3)
    torch.manual_seed(3)

    # 1. Training rollout: chain mode exactly as before (no rollback in training).
    ro = make_rollout(tok)
    b = ro.run(model, prompts, critique_fn=crit_fn)
    tg.score_tree(b, rw, "children_r1", 0.0, 1.0, 0.5, 2.0)
    args = types.SimpleNamespace(group="parent", images_per_critique=1)
    tg.assign_advantages(b, args, GRPOConfig(adv_norm="std", adv_std_floor=0.04))
    try:
        ro.run(model, prompts, critique_fn=crit_fn, rollback_fn=lambda ns: [True] * len(ns))
        raise AssertionError("rollback in chain mode must be refused")
    except AssertionError as ex:
        assert "chain mode" in str(ex), ex

    # 2. Verdict labels and training rows.
    tg.set_verdict_labels(b, MARGIN)
    nodes = [n for n in b.all_nodes() if any(k != 0 for k in n.kinds[n.seg_start:])]
    labelled = [n for n in nodes if n.verdict_label is not None]
    assert labelled and all(n.round >= 1 for n in labelled)
    for n in labelled:
        d = n.am - n.parent.am
        assert (n.verdict_label == 1) == (d > MARGIN) and abs(d) > MARGIN
    rows = ro.build_training_rows(nodes, 1.0, True, True, 1.0, own_segment_only=True, verdict=True)
    vm = rows["verdict"]
    assert int(vm.sum()) == len(labelled), (int(vm.sum()), len(labelled))
    assert bool((rows["pos_kind"][vm] == KIND_VERDICT).all()) and bool(rows["train_mask"][vm].all())
    assert float(rows["weight"][vm].abs().sum()) == 0.0 and float(rows["adv"][vm].abs().sum()) == 0.0
    for i, n in enumerate(nodes):
        L = len(n.seq)
        if n.verdict_label is None:
            assert not vm[i].any()
            continue
        ext = ro.verdict_splice + [ro.yes_id if n.verdict_label else ro.no_id]
        assert rows["input_ids"][i, L:L + len(ext)].tolist() == ext
        assert int(rows["attention_mask"][i].sum()) == L + len(ext)
    trained_img = int(((rows["pos_kind"] == KIND_IMG) & rows["train_mask"]).sum())
    assert trained_img == len(nodes) * 729, "image tokens unchanged by the verdict rows"
    print(f"verdict rows: {len(labelled)} labelled of {len(nodes)} rows "
          f"({sum(n.verdict_label for n in labelled)} Yes), one answer token each, weight 0 in GRPO")

    # 3. Eval with the policy's own rollback (fake verdicts), depth curve from states.
    ro_e = make_rollout(tok, chain=0)
    ro_e.verdict_probs = lambda m, ns: [random.random() for _ in ns]
    eargs = types.SimpleNamespace(verdict_threshold=0.5)
    e = ro_e.run(model, prompts, branch=[1] * (D + 1), critique_fn=crit_fn,
                 rollback_fn=tg.make_self_rollback_fn(model, ro_e, eargs.verdict_threshold))
    tg.score_tree(e, rw, "children_r1", 0.0, 1.0, 0.5, 2.0)
    tg.set_verdict_labels(e, MARGIN)
    n_rev = check_states(e)
    assert n_rev > 0, "random verdicts should reject some refinements"
    s = tg.tree_stats(e, "val_rb/", D)
    out = tg.finalize_stats(s, "val_rb/")
    exp = sum(r.states[-1].am for r in e.roots) / len(e.roots)
    assert abs(out["val_rb/am_final"] - exp) < 1e-9 and abs(out[f"val_rb/am_upto_{D}"] - exp) < 1e-9
    acc_keys = [k for k in out if k.startswith("val_rb/verdict_acc_")]
    assert len(acc_keys) == D, acc_keys
    print("eval rollback: am_upto_d from the kept images;",
          {k.split("/")[1]: round(v, 3) for k, v in out.items() if "revert" in k or "verdict_acc" in k})
    # Same key set as a run without rollback (all-reduce safety).
    e0 = ro_e.run(model, prompts, branch=[1] * (D + 1), critique_fn=crit_fn)
    tg.score_tree(e0, rw, "children_r1", 0.0, 1.0, 0.5, 2.0)
    assert set(tg.tree_stats(e0, "val_rb/", D)) == set(s), "key sets differ with / without rollback"

    # 4. One real train_on_tree step on a tiny random model: the verdict CE moves P(label).
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096)
    torch.manual_seed(0)
    tiny = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "lm_head"]))
    opt = torch.optim.AdamW([p for p in tiny.parameters() if p.requires_grad], lr=5e-3)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    targs = types.SimpleNamespace(train_drafts=True, train_images=True, image_adv_scale=1.0, reflect_token_weight=1.0,
                                  train_micro_batch=4, num_ppo_epochs=1, draft_kl_scale=1.0, max_grad_norm=1.0,
                                  verdict_coef=1.0)
    small = make_rollout(tok)
    sb = small.run(model, prompts[:2], critique_fn=crit_fn)
    tg.score_tree(sb, rw, "children_r1", 0.0, 1.0, 0.5, 2.0)
    tg.assign_advantages(sb, args, GRPOConfig(adv_norm="std", adv_std_floor=0.04))
    tg.set_verdict_labels(sb, MARGIN)
    gcfg = GRPOConfig(kl_coef=0.05, adv_norm="std", adv_std_floor=0.04)
    hist = [tg.train_on_tree(tiny, small, sb, targs, gcfg, opt, sch, torch.device("cpu")) for _ in range(3)]
    for h in hist:
        assert all(v == v for v in h.values()), ("NaN in train metrics", h)
    print("train_on_tree verdict_nll over 3 steps:", [round(h["verdict_nll"], 4) for h in hist],
          "n", hist[0]["verdict_n"], "loss", [round(h["loss"], 4) for h in hist])
    assert hist[-1]["verdict_nll"] < hist[0]["verdict_nll"], "verdict CE should fall on repeated data"
    # verdict_probs (batched, left-padded, last position only) == a plain forward of each row alone.
    tiny.eval()
    ns = [n for n in sb.all_nodes() if n.round >= 1][:3] + [sb.roots[0]]   # different lengths -> padding
    got = small.verdict_probs(tiny, ns)
    with torch.no_grad():
        for n, g in zip(ns, got):
            lg = tiny(input_ids=torch.tensor([n.seq + small.verdict_splice])).logits[0, -1].float()
            ref = torch.softmax(lg[[small.yes_id, small.no_id]], -1)[0].item()
            assert abs(ref - g) < 2e-3, (ref, g)
    print("verdict_probs matches per-row forwards:", [round(g, 4) for g in got])
    print("OK")


if __name__ == "__main__":
    main()
