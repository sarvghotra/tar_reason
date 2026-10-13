"""CPU test for --context_mode (rollout.py): "full" keeps the whole chain, "mdp"
rebuilds every refinement round from the prompt prefix + the latest image only.
Fakes the sampler; only the tokenizer is real.

    python util_scripts/debug/test_context_mode.py [tokenizer_dir]
"""

import random
import sys
import types

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from llava.train.rl.rollout import KIND_IMG, KIND_NONE, Critique, RolloutConfig, TreeRollout  # noqa: E402

TOK = sys.argv[1] if len(sys.argv) > 1 else "output_dir/slf_ref_edit_t21/checkpoint-17000"
DEPTH = 4          # refinement rounds: branch = [1] + [1] * DEPTH


def make_rollout(tok, mode, seen_reflection_rows, looks_good_after=None):
    img0 = tok.convert_tokens_to_ids("<I0>")
    ro = TreeRollout(tok, RolloutConfig(branch=[1] + [1] * DEPTH, gen_batch_size=8, context_mode=mode),
                     img0, 65536, torch.device("cpu"))
    ro._sample_images = lambda model, rows: [
        [img0 + random.randrange(65536) for _ in range(729)] for _ in rows]
    fix = tok("the cat is not red\nCorrection: make the cat red", add_special_tokens=False).input_ids
    good = tok("no issues, it matches the prompt.\nCorrection: looks good.", add_special_tokens=False).input_ids

    def reflections(model, rows):
        seen_reflection_rows.extend(rows)
        return [good if looks_good_after is not None and len(seen_reflection_rows) > looks_good_after
                else fix for _ in rows]
    ro._sample_reflections = reflections
    return ro, img0


def run(tok, mode, critique_fn=None, looks_good_after=None, seed=7):
    random.seed(seed)
    torch.manual_seed(seed)
    rows = []
    ro, img0 = make_rollout(tok, mode, rows, looks_good_after)
    prompts = [{"prompt": "two cats and a red dog", "vqa_list": [("Is the cat red?", "Yes")],
                "skills": ["attribute"]}]
    model = types.SimpleNamespace(training=False, eval=lambda: None, train=lambda: None)
    return ro, img0, ro.run(model, prompts, critique_fn=critique_fn), rows


def chain(batch):
    n, out = batch.roots[0], [batch.roots[0]]
    while n.children:
        n = n.children[0]
        out.append(n)
    return out


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    tok.padding_side = "left"

    # --- sampled critiques -------------------------------------------------------
    ro_f, img0, full, _ = run(tok, "full")
    ro_m, _, mdp, _ = run(tok, "mdp")
    cf, cm = chain(full), chain(mdp)
    assert len(cf) == len(cm) == DEPTH + 1, (len(cf), len(cm))
    prefix_len = cf[0].seg_start
    prefix = cf[0].seq[:prefix_len]

    # Round 1 is the same in both modes (same seed, same context).
    assert cf[1].seq == cm[1].seq, "round 1 must be identical in full and mdp"

    for k in range(1, DEPTH + 1):
        nf, nm = cf[k], cm[k]
        # full: the parent's whole sequence is a prefix of the child's.
        assert nf.seq[:len(cf[k - 1].seq)] == cf[k - 1].seq, f"full round {k} lost history"
        n_images = sum(1 for t in nf.kinds if t == KIND_IMG) + sum(
            1 for i, t in enumerate(nf.seq) if nf.kinds[i] == KIND_NONE and img0 <= t < img0 + 65536)
        assert n_images == 729 * (k + 1), (k, n_images)
        # mdp: prefix + the parent's image only, then this round's splice/critique/image.
        parent_img = [img0 + c for c in cm[k - 1].codes]
        assert nm.seq[:prefix_len] == prefix, f"mdp round {k}: prefix changed"
        assert nm.seq[prefix_len:prefix_len + 729] == parent_img, f"mdp round {k}: not the latest image"
        imgs_in_ctx = sum(1 for t in nm.seq if img0 <= t < img0 + 65536)
        assert imgs_in_ctx == 2 * 729, f"mdp round {k}: {imgs_in_ctx} image tokens, want 2 x 729"
        if k >= 2:   # round 1's context is the draft itself (trained, as in full mode)
            assert all(nm.kinds[i] == KIND_NONE for i in range(nm.seg_start)), "mdp context must be untrained"
        assert len(nm.seq) == len(cm[1].seq), f"mdp round {k} length {len(nm.seq)} != round 1 {len(cm[1].seq)}"
    print(f"full: sequence lengths per round {[len(n.seq) for n in cf]}")
    print(f"mdp : sequence lengths per round {[len(n.seq) for n in cm]}")

    # --- frozen critiques via critique_fn: the critic also sees the compact state ---
    def critique_fn(parents, prompts, sample_fn):
        texts = sample_fn(parents)
        return [Critique(t, source="policy") for t in texts]
    _, _, mdp_c, rows = run(tok, "mdp", critique_fn=critique_fn)
    want = prefix_len + 729 + len(ro_m.reflect_splice)
    assert rows and all(len(r) == want for r in rows), ([len(r) for r in rows], want)
    _, _, full_c, rows_f = run(tok, "full", critique_fn=critique_fn)
    assert [len(r) for r in rows_f] == sorted(len(r) for r in rows_f) and len(set(map(len, rows_f))) == DEPTH, \
        "full: critic context should grow every round"
    print(f"critic context lengths, mdp: {[len(r) for r in rows]}, full: {[len(r) for r in rows_f]}")

    # --- a 'looks good' ends the chain in both modes ------------------------------
    for mode in ("full", "mdp"):
        _, _, b, _ = run(tok, mode, looks_good_after=2)
        c = chain(b)
        assert c[-1].looks_good and c[-1].codes == c[-2].codes, mode
        assert len(c) < DEPTH + 1, (mode, len(c))
    print("looks good stops the chain in both modes")

    print("OK")


if __name__ == "__main__":
    main()
