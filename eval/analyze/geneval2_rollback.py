"""Rollback / selection on an iterative GenEval2 run, scored with the run's stored judge outputs (no new judging).

Each prompt's chain has images 1..N; ``<judge_run>/seed_S/<k>/results_geneval2.jsonl`` holds, per prompt, the judge's
soft-TIFA probability of every question for image k. A rule picks one image per prompt from images 1..B using the
policy's self-scores (eval/geneval2_self_judge.py); the picked image's judge probabilities give the prompt's AM
(mean), GM (geometric mean) and all-correct (every probability >= 0.5, as viz_geneval2_iter_results.py). Rules are
those of tiif_rollback.py: plain (image B), rollback, rollback_after1, best, and oracle_* (the same rules on the
judge's own per-prompt AM: an optimistic ceiling). Prints mean +- se over seeds and the paired change vs plain image N.

  python eval/analyze/geneval2_rollback.py --self_dir <dir> --judge_run_dir <run>[_judge-Qwen3.5-27B] [--score am_ans]
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path

from tiif_rollback import pick


def gmean(xs):
    return math.exp(sum(math.log(max(x, 1e-12)) for x in xs) / len(xs))


def metrics(chosen_probs):
    am = 100 * sum(sum(p) / len(p) for p in chosen_probs) / len(chosen_probs)
    gm = 100 * sum(gmean(p) for p in chosen_probs) / len(chosen_probs)
    ac = 100 * sum(all(x >= 0.5 for x in p) for p in chosen_probs) / len(chosen_probs)
    return {"AM": am, "GM": gm, "all_correct": ac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_dir", required=True)
    ap.add_argument("--judge_run_dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2213, 713, 13])
    ap.add_argument("--n_iters", type=int, default=4)
    ap.add_argument("--scores", nargs="+", default=["am_ans", "am_rephr"])
    ap.add_argument("--out", default=None, help="optional json with every seed's numbers")
    args = ap.parse_args()

    res = {}
    for seed in args.seeds:
        jd = Path(args.judge_run_dir) / f"seed_{seed}"
        probs = {k: json.load(open(jd / str(k) / "results_geneval2.jsonl")) for k in range(1, args.n_iters + 1)}
        n = len(probs[1])
        rows = [json.loads(l) for l in open(Path(args.self_dir) / f"seed_{seed}.jsonl")]
        self_s = {sc: {(int(r["key"]), r["iter"]): r[sc] for r in rows} for sc in args.scores}
        judge_am = {(i, k): sum(probs[k][i]) / len(probs[k][i]) for i in range(n) for k in probs}
        variants = {f"plain_iter{k}": [k] * n for k in range(1, args.n_iters + 1)}
        for budget in range(2, args.n_iters + 1):
            for sc in args.scores:
                for rule in ("rollback", "rollback_after1", "best"):
                    variants[f"{sc}:{rule}_B{budget}"] = [
                        pick([self_s[sc][(i, k)] for k in range(1, args.n_iters + 1)], budget, rule) for i in range(n)]
            for rule in ("rollback", "best"):
                variants[f"oracle_{rule}_B{budget}"] = [
                    pick([judge_am[(i, k)] for k in range(1, args.n_iters + 1)], budget, rule) for i in range(n)]
        for v, ks in variants.items():
            res.setdefault(v, {})[seed] = {**metrics([probs[k][i] for i, k in enumerate(ks)]),
                                           "mean_image": sum(ks) / n}
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    se = lambda x: st.stdev(x) / len(x) ** 0.5 if len(x) > 1 else float("nan")
    base = f"plain_iter{args.n_iters}"
    print(f"{'variant':32s} " + "  ".join(f"{m:>22s}" for m in ("AM", "GM", "all_correct")) + "   image")
    for v in res:
        if not (v.startswith("plain") or v.endswith(f"_B{args.n_iters}")):
            continue
        cells = []
        for m in ("AM", "GM", "all_correct"):
            vals = [res[v][s][m] for s in args.seeds]
            d = [res[v][s][m] - res[base][s][m] for s in args.seeds]
            cells.append(f"{st.mean(vals):6.2f} ({st.mean(d):+5.2f}±{se(d):.2f})")
        print(f"{v:32s} " + "  ".join(f"{c:>22s}" for c in cells) +
              f"   {st.mean(res[v][s]['mean_image'] for s in args.seeds):.2f}")


if __name__ == "__main__":
    main()
