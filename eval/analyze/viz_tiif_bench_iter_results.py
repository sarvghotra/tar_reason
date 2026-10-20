"""Analyze and visualize iterative TIIF-Bench results.

Consumes the layout written by ``eval/iterative_tiif_bench_adhoc.py`` (or its
``_MDP`` variant) and judged by ``eval/tiif_bench_vlm_judge.py`` (see
scripts/eval/iter_tiif_bench_adhoc_e2e.sh), for any number of iterations k::

    {results_dir}/images/{dimension}/{model}_iter{k}/{short,long}_description/{idx}.png
    {results_dir}/eval_results/{model}_iter{k}/{dimension}/{short,long}/{idx}.json
    {results_dir}/self_reflect.json    # "{dimension}/{desc}/{idx}" -> reflections

Each judge json holds ``questions``, ``gt_answers`` and ``model_pred``. A
prompt's score is the fraction of its yes/no questions answered as expected; it
counts as correct when every question is. Accuracies per dimension/register are
question-level (micro), which is exactly what TIIF-Bench's summary_results.py
reports per cell. Deltas compare the last iteration with the draft; movement
and reflection tables cover every step k -> k+1 (plus 1 -> n when n > 2).
"""

import argparse
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from iter_viz_common import add_text
from viz_geneval2_iter_results import (
    classify_pair,
    delta_header,
    format_score,
    has_true_iteration,
    iteration_headers,
    mean,
    print_stats,
    print_table,
    ratio,
    transition_label,
    transitions,
)


RESULTS_DIR = Path(
    "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/"
    "tiif-bench-testmini_eval/iter_adhoc_7B_512px_repeat1/seed_13"
)
DESCRIPTIONS = ("short_description", "long_description")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize outputs from iterative TIIF-Bench generation."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--model-name",
        help="Model-name level without the '_iter<k>' suffix. "
        "Inferred from images/ when there is only one.",
    )
    parser.add_argument(
        "--n-iters",
        "--n_iters",
        type=int,
        help="Iterations to analyze (default: every '_iter<k>' found, k = 1, 2, ...).",
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
    parser.add_argument(
        "--no_images",
        "--no-images",
        action="store_true",
        help="Only print the tables and stats; do not render or save any figures.",
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


def infer_n_iters(images_dir, model_name):
    """Largest n such that '_iter1'..'_iter<n>' all exist for some dimension."""
    pattern = re.compile(re.escape(model_name) + r"_iter(\d+)$")
    n_iters = 0
    for dimension_dir in (p for p in images_dir.iterdir() if p.is_dir()):
        found = {
            int(match.group(1))
            for path in dimension_dir.iterdir()
            if (match := pattern.match(path.name)) and path.is_dir()
        }
        n = 0
        while n + 1 in found:
            n += 1
        n_iters = max(n_iters, n)
    return n_iters


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


def mean_score(records, iteration):
    return mean([r["score"][iteration] for r in records])


def format_info(prompt, dimension, desc, judgement):
    lines = [f"Prompt: {prompt}", "", f"Dimension: {dimension}   Register: {desc}", ""]
    reference = next((j for j in judgement if j is not None), None)
    if reference is None:
        lines.append("(no judge results)")
        return "\n".join(lines)
    for index, (question, gt, _, _) in enumerate(reference):
        preds = " -> ".join("n/a" if j is None else j[index][2] for j in judgement)
        lines.append(f"[{preds}] {question} (gt: {gt})")
    return "\n".join(lines)


def iteration_title(score, iteration):
    if score is None:
        return f"Iteration {iteration + 1}"
    return f"Iteration {iteration + 1} — question accuracy {format_score(score)}"


def sample_reflections(entry, n_steps):
    """Sample 0's critiques, one per step; older runs only stored the first."""
    per_iter = (entry.get("reflections_per_iter") or [None])[0]
    if per_iter is None:
        per_iter = [(entry.get("reflections") or [None])[0]]
    per_iter = [(text or "").strip() for text in per_iter]
    return (per_iter + [""] * n_steps)[:n_steps]


def flags_for(reflections):
    """Whether each critique asked for a correction; None once a row stopped.

    Generation stops a sample once a critique says "looks good"; its later
    iterations reuse the image and have no critique.
    """
    flagged, stopped = [], False
    for reflection in reflections:
        flagged.append(None if stopped else has_true_iteration(reflection))
        stopped = stopped or "looks good" in reflection.lower()
    return flagged


def collect_samples(results_dir, model_name, n_iters):
    """One record per (dimension, register, prompt) present in every iteration."""
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
            iter_dirs = [
                dimension_dir / f"{model_name}_iter{k}" / desc
                for k in range(1, n_iters + 1)
            ]
            register = desc.split("_")[0]
            # Sample 0 keeps the bare "<idx>.png" name; repeats are "<idx>_<k>.png".
            for first_image in sorted(
                (p for p in iter_dirs[0].glob("*.png") if p.stem.isdigit()),
                key=lambda p: int(p.stem),
            ):
                idx = first_image.stem
                images = [d / first_image.name for d in iter_dirs]
                if not all(image.is_file() for image in images[1:]):
                    continue
                key = f"{dimension}/{desc}/{idx}"
                judgement = [
                    load_judgement(
                        eval_dir / f"{model_name}_iter{k}" / dimension / register / f"{idx}.json"
                    )
                    for k in range(1, n_iters + 1)
                ]
                question_lists = {
                    tuple(qa[:2] for qa in j) for j in judgement if j is not None
                }
                if len(question_lists) > 1:
                    # Every pass must be judged against the same question list.
                    mismatched += 1
                    judgement = [None] * n_iters
                score = [
                    None if j is None else sum(qa[3] for qa in j) / len(j)
                    for j in judgement
                ]
                entry = reflections.get(key) or {}
                record_reflections = sample_reflections(entry, n_iters - 1)
                prompt = entry.get("prompt", "(missing)")
                records.append(
                    {
                        "key": key,
                        "dimension": dimension,
                        "register": register,
                        "images": images,
                        "judgement": judgement,
                        "score": score,
                        "correct": [None if j is None else all(qa[3] for qa in j) for j in judgement],
                        "reflections": record_reflections,
                        "flagged": flags_for(record_reflections),
                        "info": format_info(prompt, dimension, desc, judgement),
                        "titles": [iteration_title(s, i) for i, s in enumerate(score)],
                    }
                )
    if mismatched:
        print(
            f"Warning: dropped scores for {mismatched} prompts whose judge files "
            "disagree on the question list across iterations"
        )
    return records


def group_by(records, field):
    groups = {}
    for record in records:
        groups.setdefault(record[field], []).append(record)
    return dict(sorted(groups.items()))


def score_cells(values):
    return [format_score(value) for value in values] + [
        format_score(values[-1] - values[0])
    ]


def print_overall(scored, n_iters):
    rows = []
    for name, metric in (
        ("Question accuracy", question_accuracy),
        ("Mean prompt score", mean_score),
    ):
        rows.append([name] + score_cells([metric(scored, k) for k in range(n_iters)]))
    correct = [sum(r["correct"][k] for r in scored) for k in range(n_iters)]
    rows.append(
        ["Prompts correct"]
        + [f"{count} ({ratio(count, len(scored))})" for count in correct]
        + [f"{correct[-1] - correct[0]:+d}"]
    )
    print_table(
        "TIIF-Bench yes/no accuracy",
        ["metric"] + iteration_headers(n_iters) + [delta_header(n_iters)],
        rows,
    )


def print_movement(scored, delta_threshold, n_iters):
    steps = transitions(n_iters)
    buckets = {
        "Improved": [], "Degraded": [], "Unchanged": [],
        "Wrong -> correct": [], "Correct -> wrong": [],
    }
    for a, b in steps:
        deltas = [r["score"][b] - r["score"][a] for r in scored]
        improved = sum(delta > delta_threshold for delta in deltas)
        degraded = sum(delta < -delta_threshold for delta in deltas)
        fixed = sum(not r["correct"][a] and r["correct"][b] for r in scored)
        broken = sum(r["correct"][a] and not r["correct"][b] for r in scored)
        for name, count in (
            ("Improved", improved),
            ("Degraded", degraded),
            ("Unchanged", len(deltas) - improved - degraded),
            ("Wrong -> correct", fixed),
            ("Correct -> wrong", broken),
        ):
            buckets[name].append(f"{count} ({ratio(count, len(scored))})")
    print_table(
        f"Per-prompt movement (score delta threshold |{delta_threshold}|)",
        ["bucket"] + [transition_label(step) for step in steps],
        [[name] + cells for name, cells in buckets.items()],
    )


def print_reflection_vs_delta(scored, n_iters):
    rows = []
    for step in range(n_iters - 1):
        groups = (
            ("flagged (model edits)", [r for r in scored if r["flagged"][step]]),
            ("kept as-is", [r for r in scored if r["flagged"][step] is False]),
            ("stopped earlier", [r for r in scored if r["flagged"][step] is None]),
        )
        for name, group in groups:
            if step and name == "stopped earlier" and not group:
                continue
            delta = mean([r["score"][step + 1] - r["score"][step] for r in group])
            rows.append(
                [transition_label((step, step + 1)), name, len(group),
                 format_score(delta)]
            )
    print_table(
        "Reflection decision vs. score delta (step k->k+1: critique of iter-k)",
        ["step", "group", "prompts", "mean score delta"],
        rows,
    )


def print_group_table(title, group_header, groups, n_iters):
    rows = [
        [name, len(group)]
        + score_cells([question_accuracy(group, k) for k in range(n_iters)])
        for name, group in groups.items()
        if group
    ]
    if rows:
        print_table(
            title,
            [group_header, "prompts"] + iteration_headers(n_iters) + [delta_header(n_iters)],
            rows,
        )


def save_visualization(record, path):
    """One row per iteration: text on the left, that iteration's image on the right.

    Row 1 holds the prompt and per-question answers; row k > 1 holds the
    critique of iteration k - 1 that produced image k.
    """
    n_iters = len(record["images"])
    fig, axes = plt.subplots(n_iters, 2, figsize=(14, 5 * n_iters), squeeze=False)
    for iteration, (image_path, title) in enumerate(
            zip(record["images"], record["titles"])):
        if iteration == 0:
            add_text(axes[0, 0], "Prompt & scores", record["info"])
        else:
            step = iteration - 1
            reflection = record["reflections"][step]
            if record["flagged"][step] is None:
                reflection = "(stopped earlier: previous image reused)"
            add_text(axes[iteration, 0], f"Self-reflection on iteration {iteration}",
                     reflection or "(missing)")
        with Image.open(image_path) as image:
            axes[iteration, 1].imshow(image)
        axes[iteration, 1].set_title(title, fontsize=11)
        axes[iteration, 1].axis("off")
    fig.suptitle(record["key"], fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_bucket(bucket_dir, records, num_samples, seed):
    """Save up to ``num_samples`` random visualizations from ``records``."""
    bucket_dir.mkdir(parents=True, exist_ok=True)
    chosen = random.Random(seed).sample(records, min(num_samples, len(records)))
    for index, record in enumerate(chosen):
        name = record["key"].replace("/", "_")
        save_visualization(record, bucket_dir / f"{index:03d}_{name}.png")
    return chosen


def render_buckets(records, scored, output_dir, args):
    """Random sample, first-critique flagged, first-critique confusion buckets
    and improved/degraded (last iteration vs draft)."""
    buckets = {
        "random": records,
        "true_iteration": [r for r in records if r["flagged"][0]],
    }
    if scored:
        confusion = {"true_pos": [], "true_neg": [], "false_pos": [], "false_neg": []}
        for record in scored:
            confusion[classify_pair(record["correct"][0], record["flagged"][0])].append(record)
        buckets.update(confusion)
        deltas = {id(r): r["score"][-1] - r["score"][0] for r in scored}
        buckets["improved"] = [r for r in scored if deltas[id(r)] > args.delta_threshold]
        buckets["degraded"] = [r for r in scored if deltas[id(r)] < -args.delta_threshold]

    rows = []
    for name, bucket_records in buckets.items():
        if not bucket_records:
            rows.append([name, 0, 0, "-"])
            continue
        bucket_dir = output_dir / name
        chosen = render_bucket(bucket_dir, bucket_records, args.num_samples, args.seed)
        rows.append([name, len(bucket_records), len(chosen), bucket_dir])
    print_table(
        "Rendered visualizations",
        ["bucket", "samples", "saved", "directory"],
        rows,
    )


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    images_dir = args.results_dir / "images"
    model_name = args.model_name or infer_model_name(images_dir)
    n_iters = args.n_iters or infer_n_iters(images_dir, model_name)
    if n_iters < 2:
        raise FileNotFoundError(
            f"Need '{model_name}_iter1' and '_iter2' under {images_dir}")
    records = collect_samples(args.results_dir, model_name, n_iters)
    if not records:
        raise FileNotFoundError(f"No paired samples found in {args.results_dir}")
    scored = [r for r in records if all(s is not None for s in r["score"])]

    print_table(
        f"TIIF-Bench coverage ({model_name})",
        ["quantity", "prompts"],
        [
            ["Iterations", n_iters],
            ["Prompts with samples", len(records)],
            ["Prompts with scores", len(scored)],
        ],
    )
    if scored:
        print_overall(scored, n_iters)
        print_movement(scored, args.delta_threshold, n_iters)
        print_reflection_vs_delta(scored, n_iters)
        print_group_table(
            "Question accuracy by register", "register",
            group_by(scored, "register"), n_iters,
        )
        print_group_table(
            "Question accuracy by dimension (both registers)", "dimension",
            group_by(scored, "dimension"), n_iters,
        )
    else:
        print("No judge results found under eval_results/; skipping score stats.")

    if not args.no_images:
        render_buckets(records, scored, args.output_dir or args.results_dir / "viz", args)

    for step in range(n_iters - 1):
        counts = {"true_pos": 0, "true_neg": 0, "false_pos": 0, "false_neg": 0}
        for record in scored:
            if record["flagged"][step] is not None:
                counts[classify_pair(record["correct"][step],
                                     record["flagged"][step])] += 1
        if sum(counts.values()):
            print_stats(counts, step)


if __name__ == "__main__":
    main()
