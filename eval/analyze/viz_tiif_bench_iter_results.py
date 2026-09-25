"""Analyze and visualize iterative TIIF-Bench results.

Consumes the layout written by ``eval/iterative_tiif_bench_adhoc.py`` and
judged by ``eval/tiif_bench_vlm_judge.py`` (see
scripts/eval/iter_tiif_bench_adhoc_e2e.sh)::

    {results_dir}/images/{dimension}/{model}_iter{1,2}/{short,long}_description/{idx}.png
    {results_dir}/eval_results/{model}_iter{1,2}/{dimension}/{short,long}/{idx}.json
    {results_dir}/self_reflect.json    # "{dimension}/{desc}/{idx}" -> reflections

Each judge json holds ``questions``, ``gt_answers`` and ``model_pred``. A
prompt's score is the fraction of its yes/no questions answered as expected; it
counts as correct when every question is. Accuracies per dimension/register are
question-level (micro), which is exactly what TIIF-Bench's summary_results.py
reports per cell.
"""

import argparse
import json
from pathlib import Path

from iter_viz_common import (
    format_score,
    has_true_iteration,
    print_coverage,
    print_group_table,
    print_movement,
    print_overall,
    print_reflection_vs_delta,
    render_and_report,
    scored_records,
)


RESULTS_DIR = Path(
    "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/"
    "tiif-bench-testmini_eval/iter_adhoc_7B_512px_repeat1/seed_13"
)
DESCRIPTIONS = ("short_description", "long_description")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize paired outputs from iterative TIIF-Bench generation."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--model-name",
        help="Model-name level without the '_iter{1,2}' suffix. "
        "Inferred from images/ when there is only one.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("-n", "--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=8811)
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.05,
        help="Minimum per-prompt accuracy change for a sample to count as "
        "improved/degraded.",
    )
    return parser.parse_args()


def infer_model_name(images_dir):
    names = {
        path.name[: -len("_iter1")]
        for path in images_dir.glob("*/*_iter1")
        if path.is_dir()
    }
    if len(names) != 1:
        raise ValueError(
            f"Found model names {sorted(names)} under {images_dir}; pass --model-name"
        )
    return names.pop()


def load_judgement(path):
    """Per-question (question, gt, pred, correct) tuples, or None if unjudged."""
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    questions = data.get("questions") or []
    gts = data.get("gt_answers") or []
    preds = data.get("model_pred") or []
    if not questions or len(preds) != len(gts):
        return None
    return [
        (q.strip(), gt.strip().lower(), pred.strip().lower(), gt.strip().lower() == pred.strip().lower())
        for q, gt, pred in zip(questions, gts, preds)
    ]


def question_accuracy(records, iteration):
    """Micro accuracy over all questions of ``records`` (TIIF-Bench's metric)."""
    answers = [qa[3] for r in records for qa in r["judgement"][iteration]]
    return sum(answers) / len(answers) if answers else None


def format_info(prompt, dimension, desc, judgement):
    lines = [f"Prompt: {prompt}", "", f"Dimension: {dimension}   Register: {desc}", ""]
    first, second = judgement
    if first is None and second is None:
        lines.append("(no judge results)")
        return "\n".join(lines)
    reference = first or second
    for index, (question, gt, _, _) in enumerate(reference):
        pred1 = "n/a" if first is None else first[index][2]
        pred2 = "n/a" if second is None else second[index][2]
        lines.append(f"[{pred1} -> {pred2}] {question} (gt: {gt})")
    return "\n".join(lines)


def iteration_title(score, iteration):
    if score is None:
        return f"Iteration {iteration + 1}"
    return f"Iteration {iteration + 1} — question accuracy {format_score(score)}"


def collect_samples(results_dir, model_name):
    """One record per (dimension, register, prompt) present in both iterations."""
    images_dir = results_dir / "images"
    eval_dir = results_dir / "eval_results"
    try:
        reflections = json.loads((results_dir / "self_reflect.json").read_text())
    except FileNotFoundError:
        reflections = {}

    records = []
    mismatched = 0
    for dimension_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        dimension = dimension_dir.name
        for desc in DESCRIPTIONS:
            first_dir = dimension_dir / f"{model_name}_iter1" / desc
            second_dir = dimension_dir / f"{model_name}_iter2" / desc
            register = desc.split("_")[0]
            # Sample 0 keeps the bare "<idx>.png" name; repeats are "<idx>_<k>.png".
            for first_image in sorted(
                (p for p in first_dir.glob("*.png") if p.stem.isdigit()),
                key=lambda p: int(p.stem),
            ):
                idx = first_image.stem
                second_image = second_dir / first_image.name
                if not second_image.is_file():
                    continue
                key = f"{dimension}/{desc}/{idx}"
                judgement = [
                    load_judgement(
                        eval_dir / f"{model_name}_iter{it}" / dimension / register / f"{idx}.json"
                    )
                    for it in (1, 2)
                ]
                if all(j is not None for j in judgement) and [
                    qa[:2] for qa in judgement[0]
                ] != [qa[:2] for qa in judgement[1]]:
                    # Both passes must be judged against the same question list.
                    mismatched += 1
                    judgement = [None, None]
                score = [
                    None if j is None else sum(qa[3] for qa in j) / len(j)
                    for j in judgement
                ]
                entry = reflections.get(key) or {}
                reflection = ((entry.get("reflections") or [None])[0] or "").strip()
                prompt = entry.get("prompt", "(missing)")
                records.append(
                    {
                        "key": key,
                        "dimension": dimension,
                        "register": register,
                        "images": (first_image, second_image),
                        "judgement": judgement,
                        "score": score,
                        "correct": [None if j is None else all(qa[3] for qa in j) for j in judgement],
                        "reflection": reflection,
                        "flagged": has_true_iteration(reflection),
                        "info": format_info(prompt, dimension, desc, judgement),
                        "titles": [iteration_title(s, i) for i, s in enumerate(score)],
                    }
                )
    if mismatched:
        print(
            f"Warning: dropped scores for {mismatched} prompts whose iter-1 and iter-2 "
            "judge files disagree on the question list"
        )
    return records


def group_by(records, field):
    groups = {}
    for record in records:
        groups.setdefault(record[field], []).append(record)
    return dict(sorted(groups.items()))


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    model_name = args.model_name or infer_model_name(args.results_dir / "images")
    records = collect_samples(args.results_dir, model_name)
    if not records:
        raise FileNotFoundError(f"No paired samples found in {args.results_dir}")
    scored = scored_records(records)

    print_coverage(f"TIIF-Bench coverage ({model_name})", records, scored)
    if scored:
        first, second = question_accuracy(scored, 0), question_accuracy(scored, 1)
        print_overall(
            "TIIF-Bench yes/no accuracy",
            scored,
            rows_before=[
                [
                    "Question accuracy",
                    format_score(first),
                    format_score(second),
                    format_score(second - first),
                ]
            ],
        )
        print_movement(scored, args.delta_threshold)
        print_reflection_vs_delta(scored)
        print_group_table(
            "Question accuracy by register",
            "register",
            group_by(scored, "register"),
            metric=question_accuracy,
        )
        print_group_table(
            "Question accuracy by dimension (both registers)",
            "dimension",
            group_by(scored, "dimension"),
            metric=question_accuracy,
        )
    else:
        print("No judge results found under eval_results/; skipping score stats.")

    output_dir = args.output_dir or args.results_dir / "viz"
    render_and_report(
        records, scored, output_dir, args.num_samples, args.seed, args.delta_threshold
    )


if __name__ == "__main__":
    main()
