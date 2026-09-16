"""Analyze and visualize iterative GenEval2 results.

Consumes the layout written by ``eval/iterative_generation_adhoc.py --geneval2``
and scored by GenEval2's ``evaluation.py --method soft_tifa``::

    {results_dir}/{1,2}/{prompt_index:05d}/samples/00000.png
    {results_dir}/{1,2}/{prompt_index:05d}/metadata.jsonl   # benchmark record
    {results_dir}/2/{prompt_index:05d}/self_reflect.txt
    {results_dir}/{1,2}/geneval2_results.json               # prompt -> image path
    {results_dir}/{1,2}/results_geneval2.jsonl              # [[per-question prob]]

``results_geneval2.jsonl`` is a single JSON list in benchmark order, so entry
``i`` belongs to prompt directory ``{i:05d}``; the per-question count from each
directory's ``metadata.jsonl`` is used to verify that alignment.
"""

import argparse
import json
import random
import re
import statistics
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


RESULTS_DIR = Path(
    "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/"
    "geneval2_specific_subset_200/iter_adhoc_7B_512px_repeat1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize paired outputs from iterative GenEval2 generation."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("-n", "--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=8811)
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=0.5,
        help="A prompt counts as correct when every VQA probability is >= this.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.05,
        help="Minimum AM-score change for a sample to count as improved/degraded.",
    )
    return parser.parse_args()


def geometric_mean(values):
    if any(value <= 0 for value in values):
        return 0.0
    return statistics.geometric_mean(values)


def read_text(path):
    try:
        return path.read_text().strip() or "(empty)"
    except FileNotFoundError:
        return "(missing)"


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def object_count(metadata):
    """How many things the prompt asks for, summed over its count questions.

    GenEval2 emits one "How many X are in the image?" question per noun phrase
    whose answer is the requested quantity, so "three cars on the left of two
    metal flowers, and a book" sums to six objects.
    """
    total = 0
    for vqa, skill in zip(
        metadata.get("vqa_list") or [], metadata.get("skills") or []
    ):
        if skill != "count":
            continue
        answer = str(vqa[1]).strip().lower()
        if answer.isdigit():
            total += int(answer)
        else:
            total += NUMBER_WORDS.get(answer, 0)
    return total or None


def load_metadata(prompt_dir):
    """Read the benchmark record stored alongside the samples."""
    try:
        line = (prompt_dir / "metadata.jsonl").read_text().strip().splitlines()[0]
    except (FileNotFoundError, IndexError):
        return {}
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {}


def load_scores(iteration_dir):
    """Load the soft-TIFA per-question probabilities, in benchmark order."""
    path = iteration_dir / "results_geneval2.jsonl"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def first_sample_name(samples_dir):
    samples = sorted(samples_dir.glob("*.png"))
    return samples[0].name if samples else None


def collect_samples(results_dir, correct_threshold):
    """Return one record per prompt present in both iterations.

    Each record carries the prompt metadata, the sample image name and, when the
    GenEval2 scores are available, the per-question probabilities of both passes.
    """
    first, second = results_dir / "1", results_dir / "2"
    scores = (load_scores(first), load_scores(second))
    records = []
    misaligned = 0
    for prompt_dir in sorted(p for p in first.glob("[0-9]*") if p.is_dir()):
        prompt_id = prompt_dir.name
        second_dir = second / prompt_id
        image_name = first_sample_name(prompt_dir / "samples")
        if image_name is None or not (second_dir / "samples" / image_name).is_file():
            continue
        metadata = load_metadata(prompt_dir)
        index = int(prompt_id)
        pair = []
        for score_list in scores:
            if score_list is None or index >= len(score_list):
                pair.append(None)
            else:
                pair.append(score_list[index])
        expected = len(metadata.get("vqa_list", []))
        if expected and any(s is not None and len(s) != expected for s in pair):
            # Positional alignment broke (e.g. scores from a different eval set).
            misaligned += 1
            pair = [None, None]
        records.append(
            {
                "prompt_id": prompt_id,
                "num_objects": object_count(metadata),
                "image_name": image_name,
                "metadata": metadata,
                "scores": pair,
                "am": [None if s is None else sum(s) / len(s) for s in pair],
                "gm": [None if s is None else geometric_mean(s) for s in pair],
                "correct": [
                    None if s is None else all(p >= correct_threshold for p in s)
                    for s in pair
                ],
            }
        )
    if misaligned:
        print(
            f"Warning: dropped scores for {misaligned} prompts whose question count "
            "does not match results_geneval2.jsonl"
        )
    return records


def reflection_for_sample(path, image_name):
    """Read the reflection belonging to an image sample from self_reflect.txt."""
    sample_id = Path(image_name).stem
    match = re.search(
        rf"^Sample: {re.escape(sample_id)}\nSelf-reflect:\s?(.*?)(?=^Sample:|\Z)",
        read_text(path),
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def has_true_iteration(reflection):
    """Whether first reflection/correction pass requested another iteration.

    The text may be a raw model output that starts with image tokens, or a
    reflection body that already starts right after "Self-reflect:". In both
    cases only the first self-reflect/correction pass is inspected.
    """
    # Drop anything before the first "Self-reflect:" (image tokens, etc.).
    parts = re.split(r"(?im)^[ \t]*Self-reflect:[ \t]*", reflection)
    first_pass = parts[1] if len(parts) > 1 else parts[0]

    # The label is usually "Correction:" but the model sometimes writes
    # variants such as "Correction/decision:".
    correction = re.search(
        r"(?im)^[ \t]*Correction[\w/ -]*:[ \t]*(.*?)\s*$", first_pass
    )
    if not correction:
        return False
    return "looks good" not in correction.group(1).lower()


def classify_pair(is_correct, flagged_issue):
    """Confusion-matrix bucket for one sample.

    "Positive" is the model predicting the first-pass image needs no fixing.
    """
    if is_correct:
        return "true_pos" if not flagged_issue else "false_neg"
    return "false_pos" if not flagged_issue else "true_neg"


def print_table(title, headers, rows):
    """Print ``rows`` as a bordered table; the first column is left-aligned."""
    cells = [[str(value) for value in row] for row in rows]
    widths = [
        max([len(str(headers[i]))] + [len(row[i]) for row in cells])
        for i in range(len(headers))
    ]

    def line(fill):
        return "+" + "+".join(fill * (width + 2) for width in widths) + "+"

    def row_text(values):
        parts = [
            f" {value:<{widths[i]}} " if i == 0 else f" {value:>{widths[i]}} "
            for i, value in enumerate(values)
        ]
        return "|" + "|".join(parts) + "|"

    if title:
        print(f"\n{title}")
    print(line("-"))
    print(row_text([str(header) for header in headers]))
    print(line("="))
    for cells_row in cells:
        print(row_text(cells_row))
    print(line("-"))


def ratio(numerator, denominator):
    return f"{numerator / denominator:.2%}" if denominator else "n/a"


def mean(values):
    return statistics.fmean(values) if values else None


def format_score(value):
    return "n/a" if value is None else f"{100 * value:.2f}"


def print_score_stats(records, delta_threshold):
    """Print AM/GM/strict-accuracy for both iterations and their deltas."""
    scored = [r for r in records if r["gm"][0] is not None and r["gm"][1] is not None]
    print_table(
        "GenEval2 coverage",
        ["quantity", "prompts"],
        [
            ["Prompts with samples", len(records)],
            ["Prompts with scores", len(scored)],
        ],
    )
    if not scored:
        print("No results_geneval2.jsonl scores found; skipping score stats.")
        return scored

    rows = []
    for name, key in (("AM", "am"), ("GM", "gm")):
        first = mean([r[key][0] for r in scored])
        second = mean([r[key][1] for r in scored])
        rows.append(
            [
                f"{name} score",
                format_score(first),
                format_score(second),
                format_score(second - first),
            ]
        )
    correct1 = sum(r["correct"][0] for r in scored)
    correct2 = sum(r["correct"][1] for r in scored)
    rows.append(
        [
            "All-questions correct",
            f"{correct1} ({ratio(correct1, len(scored))})",
            f"{correct2} ({ratio(correct2, len(scored))})",
            f"{correct2 - correct1:+d}",
        ]
    )
    print_table(
        "GenEval2 soft-TIFA scores",
        ["metric", "iter-1", "iter-2", "delta"],
        rows,
    )

    deltas = [r["am"][1] - r["am"][0] for r in scored]
    improved = sum(delta > delta_threshold for delta in deltas)
    degraded = sum(delta < -delta_threshold for delta in deltas)
    unchanged = len(deltas) - improved - degraded
    fixed = sum(not r["correct"][0] and r["correct"][1] for r in scored)
    broken = sum(r["correct"][0] and not r["correct"][1] for r in scored)
    print_table(
        f"Per-prompt movement (AM delta threshold |{delta_threshold}|)",
        ["bucket", "count", "share"],
        [
            ["Improved", improved, ratio(improved, len(deltas))],
            ["Degraded", degraded, ratio(degraded, len(deltas))],
            ["Unchanged", unchanged, ratio(unchanged, len(deltas))],
            ["Wrong -> correct", fixed, ratio(fixed, len(scored))],
            ["Correct -> wrong", broken, ratio(broken, len(scored))],
        ],
    )

    flagged = [r for r in scored if r["flagged"]]
    kept = [r for r in scored if not r["flagged"]]
    rows = []
    for name, group in (("flagged (model edits)", flagged), ("kept as-is", kept)):
        delta = mean([r["am"][1] - r["am"][0] for r in group])
        rows.append([name, len(group), format_score(delta)])
    print_table(
        "Reflection decision vs. AM delta",
        ["group", "prompts", "mean AM delta"],
        rows,
    )
    return scored


def print_skill_stats(records):
    """Per-skill soft-TIFA accuracy for both iterations."""
    per_skill = {}
    for record in records:
        skills = record["metadata"].get("skills") or []
        for iteration in (0, 1):
            score_list = record["scores"][iteration]
            if score_list is None or len(score_list) != len(skills):
                continue
            for skill, score in zip(skills, score_list):
                per_skill.setdefault(skill, ([], []))[iteration].append(score)
    if not per_skill:
        return
    rows = []
    for skill in sorted(per_skill):
        first, second = per_skill[skill]
        delta = (
            format_score(mean(second) - mean(first)) if first and second else "n/a"
        )
        rows.append(
            [
                skill,
                len(first),
                format_score(mean(first)),
                format_score(mean(second)),
                delta,
            ]
        )
    print_table(
        "Per-skill question scores",
        ["skill", "n", "iter-1", "iter-2", "delta"],
        rows,
    )


def print_object_count_stats(records):
    """AM/GM per iteration bucketed by how many objects the prompt asks for."""
    scored = [
        r
        for r in records
        if r["num_objects"] and r["am"][0] is not None and r["am"][1] is not None
    ]
    if not scored:
        return
    buckets = {}
    for record in scored:
        buckets.setdefault(record["num_objects"], []).append(record)

    rows = []
    for count in sorted(buckets):
        group = buckets[count]
        row = [f"{count} object{'s' if count != 1 else ''}", len(group)]
        for key in ("am", "gm"):
            first = mean([r[key][0] for r in group])
            second = mean([r[key][1] for r in group])
            row += [
                format_score(first),
                format_score(second),
                format_score(second - first),
            ]
        rows.append(row)

    row = ["all", len(scored)]
    for key in ("am", "gm"):
        first = mean([r[key][0] for r in scored])
        second = mean([r[key][1] for r in scored])
        row += [format_score(first), format_score(second), format_score(second - first)]
    rows.append(row)

    print_table(
        "Scores by number of objects requested",
        [
            "objects",
            "prompts",
            "AM iter-1",
            "AM iter-2",
            "AM delta",
            "GM iter-1",
            "GM iter-2",
            "GM delta",
        ],
        rows,
    )
    skipped = len(records) - len(scored)
    if skipped:
        print(f"({skipped} prompts without scores or count questions excluded)")


def print_stats(counts):
    """Print the self-reflection confusion matrix and derived metrics."""
    tp, tn, fp, fn = (
        counts["true_pos"],
        counts["true_neg"],
        counts["false_pos"],
        counts["false_neg"],
    )
    total = tp + tn + fp + fn

    print_table(
        "Self-reflection buckets (positive = model says 'looks good')",
        ["bucket", "count", "share"],
        [
            ["Total samples", total, ratio(total, total)],
            ["1st-pass correct images", tp + fn, ratio(tp + fn, total)],
            ["Reflections saying looks-good", tp + fp, ratio(tp + fp, total)],
            ["True positives  (correct, looks good)", tp, ratio(tp, total)],
            ["False negatives (correct, flagged)", fn, ratio(fn, total)],
            ["True negatives  (wrong, flagged)", tn, ratio(tn, total)],
            ["False positives (wrong, looks good)", fp, ratio(fp, total)],
        ],
    )
    print_table(
        "Self-reflection metrics",
        ["metric", "value"],
        [
            ["Accuracy", ratio(tp + tn, total)],
            ["Precision", ratio(tp, tp + fp)],
            ["Recall", ratio(tp, tp + fn)],
            ["Specificity", ratio(tn, tn + fp)],
            ["F1", ratio(2 * tp, 2 * tp + fp + fn)],
        ],
    )


def add_text(ax, title, value):
    wrapped = "\n".join(
        textwrap.fill(line, width=58, replace_whitespace=False)
        for line in value.splitlines()
    )
    ax.set_title(title, fontsize=11)
    ax.text(0.01, 0.99, wrapped, transform=ax.transAxes, va="top", fontsize=8)
    ax.axis("off")


def format_question_scores(record):
    """Prompt, then one line per VQA question with both iterations' scores."""
    metadata = record["metadata"]
    lines = [f"Prompt: {metadata.get('prompt', '(missing)')}", ""]
    vqa_list = metadata.get("vqa_list") or []
    first, second = record["scores"]
    for index, vqa in enumerate(vqa_list):
        question, answer = vqa[0], vqa[1]
        score1 = "n/a" if first is None else f"{first[index]:.2f}"
        score2 = "n/a" if second is None else f"{second[index]:.2f}"
        lines.append(f"[{score1} -> {score2}] {question} ({answer})")
    if not vqa_list:
        lines.append("(no vqa_list in metadata.jsonl)")
    return "\n".join(lines)


def iteration_title(record, iteration):
    am, gm = record["am"][iteration], record["gm"][iteration]
    if am is None:
        return f"Iteration {iteration + 1}"
    return (
        f"Iteration {iteration + 1} — AM {format_score(am)} / GM {format_score(gm)}"
    )


def save_visualization(results_dir, output_dir, record, index):
    prompt_id, image_name = record["prompt_id"], record["image_name"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    add_text(axes[0, 0], "Prompt & question scores", format_question_scores(record))
    with Image.open(results_dir / "1" / prompt_id / "samples" / image_name) as image:
        axes[0, 1].imshow(image)
    axes[0, 1].set_title(iteration_title(record, 0), fontsize=11)
    axes[0, 1].axis("off")

    add_text(axes[1, 0], "Self-reflection", record["reflection"] or "(missing)")
    with Image.open(results_dir / "2" / prompt_id / "samples" / image_name) as image:
        axes[1, 1].imshow(image)
    axes[1, 1].set_title(iteration_title(record, 1), fontsize=11)
    axes[1, 1].axis("off")

    fig.suptitle(f"Prompt {prompt_id} — sample {image_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{index:03d}_{prompt_id}_{image_name}", dpi=120)
    plt.close(fig)


def render_bucket(results_dir, bucket_dir, records, num_samples, seed):
    """Save up to ``num_samples`` random visualizations from ``records``."""
    bucket_dir.mkdir(parents=True, exist_ok=True)
    chosen = random.Random(seed).sample(records, min(num_samples, len(records)))
    for index, record in enumerate(chosen):
        save_visualization(results_dir, bucket_dir, record, index)
    return chosen


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    records = collect_samples(args.results_dir, args.correct_threshold)
    if not records:
        raise FileNotFoundError(f"No paired samples found in {args.results_dir}")

    for record in records:
        record["reflection"] = reflection_for_sample(
            args.results_dir / "2" / record["prompt_id"] / "self_reflect.txt",
            record["image_name"],
        )
        record["flagged"] = has_true_iteration(record["reflection"])

    output_dir = args.output_dir or args.results_dir / "viz"
    selected = render_bucket(
        args.results_dir, output_dir, records, args.num_samples, args.seed
    )
    render_rows = [["all (random sample)", len(records), len(selected), output_dir]]

    true_iteration = [record for record in records if record["flagged"]]
    if true_iteration:
        true_iteration_dir = args.results_dir / "viz_true_iteration"
        chosen = render_bucket(
            args.results_dir,
            true_iteration_dir,
            true_iteration,
            args.num_samples,
            args.seed,
        )
        render_rows.append(
            ["true_iteration", len(true_iteration), len(chosen), true_iteration_dir]
        )

    scored = print_score_stats(records, args.delta_threshold)
    print_skill_stats(records)
    print_object_count_stats(records)
    if not scored:
        print_table(
            "Rendered visualizations",
            ["bucket", "samples", "saved", "directory"],
            render_rows,
        )
        return

    buckets = {"true_pos": [], "true_neg": [], "false_pos": [], "false_neg": []}
    for record in scored:
        buckets[classify_pair(record["correct"][0], record["flagged"])].append(record)

    delta_buckets = {
        "improved": [
            r for r in scored if r["am"][1] - r["am"][0] > args.delta_threshold
        ],
        "degraded": [
            r for r in scored if r["am"][1] - r["am"][0] < -args.delta_threshold
        ],
    }

    for name, bucket_records in {**buckets, **delta_buckets}.items():
        bucket_dir = args.results_dir / name
        chosen = render_bucket(
            args.results_dir, bucket_dir, bucket_records, args.num_samples, args.seed
        )
        render_rows.append([name, len(bucket_records), len(chosen), bucket_dir])
    print_table(
        "Rendered visualizations",
        ["bucket", "samples", "saved", "directory"],
        render_rows,
    )

    print_stats({name: len(items) for name, items in buckets.items()})


if __name__ == "__main__":
    main()
