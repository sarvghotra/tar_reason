"""CPU test for stop-SFT (llava/train/rl/stop_sft/): labels, balancing, and that a training
example's context is exactly the row the MDP rollout critiques from. Fakes the sampler;
only the tokenizer is real.

    python util_scripts/debug/test_stop_sft.py [tokenizer_dir]
"""

import random
import sys
import types

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from llava.train.rl.grpo import KIND_TXT  # noqa: E402
from llava.train.rl.oracle_critique import NO_ISSUE_CRITIQUE  # noqa: E402
from llava.train.rl.rollout import RolloutConfig, TreeRollout  # noqa: E402
from llava.train.rl.stop_sft.build_data import balance, label  # noqa: E402
from llava.train.rl.stop_sft.train_stop_sft import Encoder  # noqa: E402

TOK = sys.argv[1] if len(sys.argv) > 1 else "output_dir/slf_ref_edit_t21/checkpoint-17000"
FIX = "the cat is not red\nCorrection: make the cat red"
VQA = [["How many cats are in the image?", "one"], ["Is the cat red?", "Yes"]]


def dump_line(critique, pq, rnd=1):
    return {"prompt": "a red cat", "round": rnd, "questions": [q for q, _ in VQA],
            "answers": [a for _, a in VQA], "skills": ["count", "attribute"], "critique": critique,
            "looks_good": "looks good" in critique, "parent_per_question": pq,
            "parent_codes": [1, 2, 3], "codes": [4, 5, 6]}


def test_labels():
    # pass + own looks good -> own; pass + fix -> template; fail + fix -> own; fail + looks good -> oracle
    e = label(dump_line(NO_ISSUE_CRITIQUE, [0.9, 0.8]))
    assert e["passed"] and e["source"] == "own" and e["target"] == NO_ISSUE_CRITIQUE and e["round"] == 0
    e = label(dump_line(FIX, [0.9, 0.8]))
    assert e["passed"] and e["source"] == "template" and e["target"] == NO_ISSUE_CRITIQUE
    e = label(dump_line(FIX, [0.9, 0.1], rnd=3))
    assert not e["passed"] and e["source"] == "own" and e["target"] == FIX and e["round"] == 2
    e = label(dump_line(NO_ISSUE_CRITIQUE, [0.9, 0.1]))
    assert not e["passed"] and e["source"] == "oracle" and "looks good" not in e["target"]
    assert "red" in e["target"] and "\nCorrection: " in e["target"], e["target"]
    assert e["codes"] == [1, 2, 3]                     # the critiqued image is the parent's
    assert label({**dump_line(FIX, [0.9, 0.1]), "parent_codes": None}) is None
    ex = [{"passed": i < 3} for i in range(10)]
    out = balance(ex, seed=1)
    assert sum(e["passed"] for e in out) == 3 and len(out) == 6
    print("labels + balance ok")


def test_layout(tok):
    """The encoder's context = the row the rollout's critic sees for that image (mdp)."""
    img0 = tok.convert_tokens_to_ids("<I0>")
    ro = TreeRollout(tok, RolloutConfig(branch=[1, 1, 1, 1], gen_batch_size=8, context_mode="mdp"),
                     img0, 65536, torch.device("cpu"))
    ro._sample_images = lambda model, rows: [[img0 + random.randrange(65536) for _ in range(729)]
                                             for _ in rows]
    seen = []
    fix = tok(FIX, add_special_tokens=False).input_ids
    # copies: the rollout extends these lists after the critique
    ro._sample_reflections = lambda model, rows: (seen.extend(list(r) for r in rows), [fix for _ in rows])[1]
    random.seed(3)
    prompts = [{"prompt": "a red cat", "vqa_list": VQA, "skills": ["count", "attribute"]}]
    model = types.SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)
    batch = ro.run(model, prompts=prompts, branch=[1, 1, 1, 1])
    chain = batch.leaves[0].ancestors()
    assert len(chain) == 4 and len(seen) == 3        # 3 critiques: on images of rounds 0, 1, 2
    enc = Encoder(tok)
    for k in range(3):
        ex = {"prompt": "a red cat", "codes": chain[k].codes, "target": FIX}
        ctx, tgt = enc.encode(ex)
        assert ctx == seen[k], f"context of round-{k} image differs from the rollout's row"
        assert tgt[-len(enc.stop_fix):] == enc.stop_fix
        assert tok.decode(tgt[:-len(enc.stop_fix)]).strip() == FIX
    ctx, tgt = enc.encode({"prompt": "a red cat", "codes": chain[0].codes, "target": NO_ISSUE_CRITIQUE})
    assert tgt[-1] == tok.convert_tokens_to_ids("<|im_end|>")
    b = enc.batch([{"prompt": "a red cat", "codes": chain[0].codes, "target": NO_ISSUE_CRITIQUE},
                   {"prompt": "a red cat", "codes": chain[1].codes, "target": FIX}], tok.pad_token_id)
    for i, t in enumerate([NO_ISSUE_CRITIQUE, FIX]):
        ids = b["input_ids"][i][b["train_mask"][i]].tolist()
        assert tok.decode(ids, skip_special_tokens=False).strip().startswith(t.split("\n")[0])
        assert (b["pos_kind"][i][b["train_mask"][i]] == KIND_TXT).all()
        assert not b["train_mask"][i][b["attention_mask"][i] == 0].any()
    n_img = int(((b["input_ids"][0] >= img0) & (b["input_ids"][0] < img0 + 65536)).sum())
    assert n_img == 729, n_img
    print("layout ok (encoder context == rollout critic row, rounds 0-2)")


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(TOK)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    test_labels()
    test_layout(tok)
    print("all ok")
