"""Rollback selection on an iterative TIIF-Bench run, scored with the run's existing judge outputs.

Each prompt's chain has images 1..N (``<seed>/images/<dim>/<tag>_iter<k>/<desc>/<idx>.png``), each already judged
(``<seed>/<judge_dir>/<tag>_iter<k>/<dim>/<short|long>/<idx>.json``). A rule picks one image per prompt from the
first B images (budget B = 2..N); the picked image's judge file is copied into a new "model" folder
``<out>/seed_S/eval_results/<variant>_B<B>/...`` next to links to the plain iterations, and TIIF-Bench's own
``summary_results.py`` + ``summary_dimension_results.py`` score every folder the same way.

Rules (s_k = the policy's self-score of image k from eval/tiif_self_judge.py; ties keep the earlier image):
  plain     the last image (= the run's iter_B; chains stopped by "looks good" repeat their last image)
  rollback  start at image 1; accept image k+1 only if s_{k+1} > s_k, else keep image k and stop
            (what a rollback rule would have produced: a rejected refinement ends the chain)
  best      argmax_k s_k over images 1..B
  oracle_*  the same rules with s_k = the judge's per-prompt accuracy (fraction of its questions answered as
            the ground truth): an upper bound that uses the evaluation judge itself

  python eval/analyze/tiif_rollback.py --run_dir <run> --self_dir <tiif_self_judge out> --out <dir> \\
      --score approve_lp --tiif_dir <TIIF-Bench checkout> --judge_py <python with pandas/openpyxl>
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

DESC = {"short_description": "short", "long_description": "long"}


def judge_file(seed_dir, judge_dir, tag, k, key):
    dim, desc, idx = key.split("/")
    return seed_dir / judge_dir / f"{tag}_iter{k}" / dim / DESC[desc] / f"{idx}.json"


def accuracy(path):
    d = json.load(open(path))
    pairs = list(zip(d.get("gt_answers", []), d.get("model_pred", [])))
    return sum(str(g).strip().lower() == str(p).strip().lower() for g, p in pairs) / max(len(pairs), 1)


def pick(scores, budget, rule):
    s = scores[:budget]
    if rule == "plain":
        return budget
    if rule == "best":
        return max(range(budget), key=lambda i: (s[i], -i)) + 1
    cur = 0
    for k in range(1, budget):
        if s[k] > s[cur]:
            cur = k
        else:
            break
    return cur + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--self_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 17, 91])
    ap.add_argument("--tag", default="E-450_ORC")
    ap.add_argument("--judge_dir", default="eval_results")
    ap.add_argument("--n_iters", type=int, default=4)
    ap.add_argument("--score", default="approve_lp", choices=["approve_lp", "approve_lp_mean", "vqa_yes"])
    ap.add_argument("--tiif_dir", required=True)
    ap.add_argument("--judge_py", required=True)
    args = ap.parse_args()

    stats = {}
    for seed in args.seeds:
        seed_dir = Path(args.run_dir) / f"seed_{seed}"
        self_scores = {}
        for r in map(json.loads, open(Path(args.self_dir) / f"seed_{seed}.jsonl")):
            self_scores[(r["key"], r["iter"])] = r[args.score]
        keys = sorted({k for k, _ in self_scores})
        ev = Path(args.out) / f"seed_{seed}" / "eval_results"
        if ev.exists():
            shutil.rmtree(ev)
        ev.mkdir(parents=True)
        for k in range(1, args.n_iters + 1):
            os.symlink(seed_dir / args.judge_dir / f"{args.tag}_iter{k}", ev / f"plain_iter{k}")
        picks = {}
        for key in keys:
            s = [self_scores[(key, k)] for k in range(1, args.n_iters + 1)]
            acc = [accuracy(judge_file(seed_dir, args.judge_dir, args.tag, k, key)) for k in range(1, args.n_iters + 1)]
            for budget in range(2, args.n_iters + 1):
                for rule, sc in [("rollback", s), ("best", s), ("oracle_rollback", acc), ("oracle_best", acc)]:
                    base = rule.replace("oracle_", "")
                    picks.setdefault(f"{rule}_B{budget}", {})[key] = pick(sc, budget, base)
        for variant, chosen in picks.items():
            for key, k in chosen.items():
                dim, desc, idx = key.split("/")
                dst = ev / variant / dim / DESC[desc] / f"{idx}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(judge_file(seed_dir, args.judge_dir, args.tag, k, key), dst)
            n = len(chosen)
            stats.setdefault(variant, {})[seed] = {
                "mean_pick": sum(chosen.values()) / n,
                "kept_draft": sum(v == 1 for v in chosen.values()) / n}
        # TIIF-Bench's own summaries (run from its checkout, as bench_jeet_mila.sh does)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        subprocess.run([args.judge_py, "eval/summary_results.py", "--input_dir", str(ev)], cwd=args.tiif_dir,
                       env=env, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([args.judge_py, "eval/summary_dimension_results.py", "--input_excel",
                        str(ev / "result_summary.xlsx"), "--output_txt", str(ev / "result_summary_dimension.txt")],
                       cwd=args.tiif_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert (ev / "result_summary_dimension.txt").stat().st_size > 0, ev
        print(f"seed {seed}: {len(keys)} prompts, {len(picks)} variants -> {ev}", flush=True)
    json.dump(stats, open(Path(args.out) / "pick_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
