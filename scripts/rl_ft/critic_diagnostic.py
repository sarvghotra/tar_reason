"""Critic diagnostic over validation dumps (train_grpo.py --val_dump_dir).

For every draft in the own-critic validation (prefix ``val/``, greedy, one critique
per draft) it asks two questions:

1. Stop decision: did the critic say "looks good" exactly when the draft passes every
   question (judge probability >= --fail_threshold)? Reported as a 2x2 table plus
   error recall (failing drafts sent to refinement) and stop precision (stops on
   drafts that really pass).
2. Content: when the critic refines a failing draft, does the critique mention the
   failed questions? "Mention" is a keyword heuristic (a content word of the question,
   plural-tolerant, appears in the critique), so it over-counts on prompts
   whose questions share object names; read it as approximate. The fix rate is split
   by mentioned / not mentioned, which checks whether mentions matter.

Also: corrections that say "no edit" / "no change" without "looks good" (they are
refined today, since the rollout only stops on "looks good"), a per-skill breakdown,
a split by atom count, and example critiques from each wrong cell.

Usage: python scripts/rl_ft/critic_diagnostic.py <dump_dir> [--step N] [--examples 5]
"""
import argparse
import collections
import glob
import json
import os
import re

STOPWORDS = set("""
how many are there any the image in is of on to with what color a an does do it this that
be being been was were has have its their they them shown visible picture photo some other
""".split())
NO_EDIT_RE = re.compile(r"\bno (edits?|changes?|corrections?)\b|\bnothing to (fix|change|correct)\b",
                        re.IGNORECASE)


def content_words(text: str):
    return {w.lower() for w in re.findall(r"[A-Za-z]+", text) if len(w) >= 3 and w.lower() not in STOPWORDS}


def same_word(a: str, b: str) -> bool:
    """Plural-tolerant match: dog/dogs, horse/horses, glass/glasses, box/boxes."""
    short, long_ = sorted((a, b), key=len)
    return long_.startswith(short) and len(long_) - len(short) <= 2


def mentions(question: str, critique_words) -> bool:
    return any(same_word(q, c) for q in content_words(question) for c in critique_words)


def load(dump_dir, prefix, step):
    rows = []
    for path in sorted(glob.glob(os.path.join(dump_dir, f"{prefix.strip('/')}_step*_rank*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if r["round"] == 1 and (step is None or r["step"] == step):
                rows.append(r)
    return rows


def pct(num, den):
    return f"{100.0 * num / den:5.1f}% ({num}/{den})" if den else "   n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--step", type=int, default=None, help="Only this validation step.")
    ap.add_argument("--fail_threshold", type=float, default=0.5)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    rows = load(args.dump_dir, "val/", args.step)
    if not rows:
        raise SystemExit(f"no val/ round-1 rows in {args.dump_dir}")
    steps = sorted({r["step"] for r in rows})
    print(f"{len(rows)} drafts with an own critique, step(s) {steps}\n")
    thr = args.fail_threshold

    cells = collections.Counter()
    examples = collections.defaultdict(list)
    by_skill = collections.defaultdict(collections.Counter)
    by_atoms = collections.defaultdict(collections.Counter)
    mention = collections.Counter()
    mention_skill = collections.defaultdict(collections.Counter)
    no_edit = collections.Counter()

    for r in rows:
        ppq = r["parent_per_question"] or []
        failed = [i for i, p in enumerate(ppq) if p < thr]
        passes = not failed
        stop = bool(r["looks_good"])
        cell = ("correct_stop" if passes else "missed_error") if stop else \
               ("unneeded_refine" if passes else "correct_refine")
        cells[cell] += 1
        if len(examples[cell]) < args.examples:
            examples[cell].append(r)
        by_atoms[len(ppq)][cell] += 1
        for sk in {r["skills"][i] for i in failed}:
            by_skill[sk]["fail_drafts"] += 1
            by_skill[sk]["missed"] += stop
        if not stop and NO_EDIT_RE.search(r["critique"] or ""):
            no_edit["passing" if passes else "failing"] += 1

        if stop or passes:
            continue
        crit = content_words(r["critique"] or "")
        cpq = r["per_question"] or []
        for i in failed:
            sk = r["skills"][i]
            hit = mentions(r["questions"][i], crit)
            fixed = i < len(cpq) and cpq[i] >= thr
            key = "mentioned" if hit else "not_mentioned"
            mention[key] += 1
            mention[f"{key}_fixed"] += fixed
            mention_skill[sk][key] += 1

    n = len(rows)
    n_pass = cells["correct_stop"] + cells["unneeded_refine"]
    n_fail = cells["missed_error"] + cells["correct_refine"]
    n_stop = cells["correct_stop"] + cells["missed_error"]
    print("== Stop decision (rows: critic; columns: judge on the draft)")
    print(f"{'':28s}{'draft passes all':>22s}{'draft fails >=1':>22s}")
    print(f"{'critic: looks good (stop)':28s}{cells['correct_stop']:>22d}{cells['missed_error']:>22d}")
    print(f"{'critic: refine':28s}{cells['unneeded_refine']:>22d}{cells['correct_refine']:>22d}")
    print(f"drafts passing every question: {pct(n_pass, n)}")
    print(f"error recall  (failing drafts sent to refine): {pct(cells['correct_refine'], n_fail)}")
    print(f"stop precision (stops on drafts that pass):    {pct(cells['correct_stop'], n_stop)}")
    print(f"stop recall    (passing drafts the critic stops on): {pct(cells['correct_stop'], n_pass)}")
    print(f"'no edit/no change' without 'looks good' (refined anyway): on passing drafts "
          f"{no_edit['passing']}, on failing drafts {no_edit['failing']}\n")

    m, nm = mention["mentioned"], mention["not_mentioned"]
    print("== Content of refinements on failing drafts (keyword heuristic, approximate)")
    print(f"failed questions mentioned by the critique: {pct(m, m + nm)}")
    print(f"fix rate when mentioned:     {pct(mention['mentioned_fixed'], m)}")
    print(f"fix rate when not mentioned: {pct(mention['not_mentioned_fixed'], nm)}\n")

    print("== By skill of the failed questions")
    print(f"{'skill':12s}{'failing drafts':>16s}{'stopped (missed)':>20s}{'mentioned':>22s}")
    for sk in sorted(by_skill, key=lambda k: -by_skill[k]["fail_drafts"]):
        b, ms = by_skill[sk], mention_skill[sk]
        print(f"{sk:12s}{b['fail_drafts']:>16d}{pct(b['missed'], b['fail_drafts']):>20s}"
              f"{pct(ms['mentioned'], ms['mentioned'] + ms['not_mentioned']):>22s}")
    print()

    print("== By number of questions (atoms) per prompt")
    print(f"{'atoms':>6s}{'drafts':>8s}{'pass all':>18s}{'error recall':>20s}{'stop precision':>20s}")
    for a in sorted(by_atoms):
        c = by_atoms[a]
        tot = sum(c.values())
        print(f"{a:>6d}{tot:>8d}{pct(c['correct_stop'] + c['unneeded_refine'], tot):>18s}"
              f"{pct(c['correct_refine'], c['correct_refine'] + c['missed_error']):>20s}"
              f"{pct(c['correct_stop'], c['correct_stop'] + c['missed_error']):>20s}")
    print()

    for cell in ("missed_error", "unneeded_refine", "correct_refine"):
        print(f"== Examples: {cell}")
        for r in examples[cell]:
            ppq = r["parent_per_question"] or []
            failed = [f"{r['questions'][i]} (want {r['answers'][i]}, p={ppq[i]:.2f})"
                      for i, p in enumerate(ppq) if p < thr]
            print(f"- prompt: {r['prompt']}")
            print(f"  failed: {failed or 'none'}")
            print(f"  critique: {(r['critique'] or '').strip()[:400]}")
        print()


if __name__ == "__main__":
    main()
