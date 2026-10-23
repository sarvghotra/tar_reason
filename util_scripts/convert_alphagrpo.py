"""Convert AlphaGRPO's DVReward prompt set (alpha_grpo/dataset/alphagrpo20k/{train,test}.jsonl, Git LFS in
https://github.com/huangrh99/AlphaGRPO) into our GRPO JSONL: prompt, vqa_list (valid semantic questions, answer
"Yes"), skills (the semantic tag, lower-cased), atom_count, plus quality_list (valid quality questions, "Yes").
Negative-tagged semantic questions keep answer "Yes" as well: AlphaGRPO phrases every question so that "yes"
is the desired answer (e.g. "Is the cup empty?" for "no coffee in the cup").

    python util_scripts/convert_alphagrpo.py <in.jsonl> <out.jsonl> [--limit N]
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    n = kept = 0
    with open(a.src) as f, open(a.dst, "w") as out:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            sem = [q for q in r["semantic_questions"] if q.get("is_valid", True)]
            qual = [q for q in r["quality_questions"] if q.get("is_valid", True)]
            if not sem:
                continue
            row = {
                "prompt": r["prompt"].strip(),
                "vqa_list": [[q["question"], "Yes"] for q in sem],
                "skills": [str(q.get("tag", "unknown")).lower() for q in sem],
                "atom_count": len(sem),
                "quality_list": [[q["question"], "Yes"] for q in qual],
                "task": r.get("task"), "difficulty": r.get("difficulty"),
            }
            out.write(json.dumps(row) + "\n")
            kept += 1
            if a.limit and kept >= a.limit:
                break
    print(f"{a.src}: {n} rows read, {kept} written to {a.dst}")


if __name__ == "__main__":
    main()
