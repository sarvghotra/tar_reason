"""Analyze and visualize iterative GenEval2 results.

Consumes the layout written by ``eval/iterative_generation_adhoc.py --geneval2``
(or its ``_MDP`` variant) and scored by GenEval2's ``evaluation.py --method
soft_tifa``, for iterations k = 1..N (N is detected from the directories)::

    {results_dir}/{k}/{prompt_index:05d}/samples/00000.png
    {results_dir}/{k}/{prompt_index:05d}/metadata.jsonl   # benchmark record
    {results_dir}/{k}/{prompt_index:05d}/self_reflect.txt # k >= 2: critique of image k-1
    {results_dir}/{k}/geneval2_results.json               # prompt -> image path
    {results_dir}/{k}/results_geneval2.jsonl              # [[per-question prob]]

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
    parser.add_argument(
        "--no_images",
        "--no-images",
        action="store_true",
        help="Only print the tables and stats; do not render or save any figures.",
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


def find_iteration_dirs(results_dir):
    """Return the contiguous iteration directories ``1, 2, ..., N``."""
    dirs = []
    while (results_dir / str(len(dirs) + 1)).is_dir():
        dirs.append(results_dir / str(len(dirs) + 1))
    return dirs


def collect_samples(iteration_dirs, correct_threshold):
    """Return one record per prompt present in every iteration.

    Each record carries the prompt metadata, the sample image name and, when the
    GenEval2 scores are available, the per-question probabilities of every pass
    (lists indexed by iteration).
    """
    scores = [load_scores(iteration_dir) for iteration_dir in iteration_dirs]
    first = iteration_dirs[0]
    records = []
    misaligned = 0
    for prompt_dir in sorted(p for p in first.glob("[0-9]*") if p.is_dir()):
        prompt_id = prompt_dir.name
        image_name = first_sample_name(prompt_dir / "samples")
        if image_name is None or not all(
            (iteration_dir / prompt_id / "samples" / image_name).is_file()
            for iteration_dir in iteration_dirs[1:]
        ):
            continue
        metadata = load_metadata(prompt_dir)
        index = int(prompt_id)
        per_iter = []
        for score_list in scores:
            if score_list is None or index >= len(score_list):
                per_iter.append(None)
            else:
                per_iter.append(score_list[index])
        expected = len(metadata.get("vqa_list", []))
        if expected and any(s is not None and len(s) != expected for s in per_iter):
            # Positional alignment broke (e.g. scores from a different eval set).
            misaligned += 1
            per_iter = [None] * len(iteration_dirs)
        records.append(
            {
                "prompt_id": prompt_id,
                "num_objects": object_count(metadata),
                "image_name": image_name,
                "metadata": metadata,
                "scores": per_iter,
                "am": [None if s is None else sum(s) / len(s) for s in per_iter],
                "gm": [None if s is None else geometric_mean(s) for s in per_iter],
                "correct": [
                    None if s is None else all(p >= correct_threshold for p in s)
                    for s in per_iter
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


def iteration_headers(n_iters, prefix=""):
    return [f"{prefix}iter-{k}" for k in range(1, n_iters + 1)]


def delta_header(n_iters, prefix=""):
    return f"{prefix}delta 1->{n_iters}"


def transitions(n_iters):
    """(from, to) iteration indices: each refinement step, plus first -> last."""
    steps = [(k, k + 1) for k in range(n_iters - 1)]
    if n_iters > 2:
        steps.append((0, n_iters - 1))
    return steps


def transition_label(step):
    return f"{step[0] + 1}->{step[1] + 1}"


def print_score_stats(records, delta_threshold, n_iters):
    """Print AM/GM/strict-accuracy for every iteration and the deltas."""
    scored = [r for r in records if all(g is not None for g in r["gm"])]
    print_table(
        "GenEval2 coverage",
        ["quantity", "prompts"],
        [
            ["Iterations", n_iters],
            ["Prompts with samples", len(records)],
            ["Prompts with scores", len(scored)],
        ],
    )
    if not scored:
        print("No results_geneval2.jsonl scores found; skipping score stats.")
        return scored

    last = n_iters - 1
    rows = []
    for name, key in (("AM", "am"), ("GM", "gm")):
        values = [mean([r[key][k] for r in scored]) for k in range(n_iters)]
        rows.append(
            [f"{name} score"]
            + [format_score(value) for value in values]
            + [format_score(values[last] - values[0])]
        )
    correct = [sum(r["correct"][k] for r in scored) for k in range(n_iters)]
    rows.append(
        ["All-questions correct"]
        + [f"{count} ({ratio(count, len(scored))})" for count in correct]
        + [f"{correct[last] - correct[0]:+d}"]
    )
    print_table(
        "GenEval2 soft-TIFA scores",
        ["metric"] + iteration_headers(n_iters) + [delta_header(n_iters)],
        rows,
    )

    steps = transitions(n_iters)
    buckets = {
        "Improved": [], "Degraded": [], "Unchanged": [],
        "Wrong -> correct": [], "Correct -> wrong": [],
    }
    for a, b in steps:
        deltas = [r["am"][b] - r["am"][a] for r in scored]
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
        f"Per-prompt movement (AM delta threshold |{delta_threshold}|)",
        ["bucket"] + [transition_label(step) for step in steps],
        [[name] + cells for name, cells in buckets.items()],
    )

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
            delta = mean([r["am"][step + 1] - r["am"][step] for r in group])
            rows.append(
                [transition_label((step, step + 1)), name, len(group),
                 format_score(delta)]
            )
    print_table(
        "Reflection decision vs. AM delta (step k->k+1: critique of iter-k)",
        ["step", "group", "prompts", "mean AM delta"],
        rows,
    )
    return scored


def print_skill_stats(records, n_iters):
    """Per-skill soft-TIFA accuracy for every iteration."""
    per_skill = {}
    for record in records:
        skills = record["metadata"].get("skills") or []
        for iteration in range(n_iters):
            score_list = record["scores"][iteration]
            if score_list is None or len(score_list) != len(skills):
                continue
            for skill, score in zip(skills, score_list):
                per_skill.setdefault(skill, [[] for _ in range(n_iters)])[
                    iteration
                ].append(score)
    if not per_skill:
        return
    rows = []
    for skill in sorted(per_skill):
        values = per_skill[skill]
        first, final = values[0], values[-1]
        delta = (
            format_score(mean(final) - mean(first)) if first and final else "n/a"
        )
        rows.append(
            [skill, len(first)]
            + [format_score(mean(value)) for value in values]
            + [delta]
        )
    print_table(
        "Per-skill question scores",
        ["skill", "n"] + iteration_headers(n_iters) + [delta_header(n_iters)],
        rows,
    )


def print_object_count_stats(records, n_iters):
    """AM/GM per iteration bucketed by how many objects the prompt asks for."""
    scored = [
        r
        for r in records
        if r["num_objects"] and all(am is not None for am in r["am"])
    ]
    if not scored:
        return
    buckets = {}
    for record in scored:
        buckets.setdefault(record["num_objects"], []).append(record)

    def score_cells(group):
        cells = []
        for key in ("am", "gm"):
            values = [mean([r[key][k] for r in group]) for k in range(n_iters)]
            cells += [format_score(value) for value in values]
            cells.append(format_score(values[-1] - values[0]))
        return cells

    rows = []
    for count in sorted(buckets):
        group = buckets[count]
        rows.append(
            [f"{count} object{'s' if count != 1 else ''}", len(group)]
            + score_cells(group)
        )
    rows.append(["all", len(scored)] + score_cells(scored))

    headers = ["objects", "prompts"]
    for name in ("AM", "GM"):
        headers += iteration_headers(n_iters, f"{name} ")
        headers.append(delta_header(n_iters, f"{name} "))
    print_table("Scores by number of objects requested", headers, rows)
    skipped = len(records) - len(scored)
    if skipped:
        print(f"({skipped} prompts without scores or count questions excluded)")


def print_stats(counts, step=0):
    """Print the self-reflection confusion matrix and derived metrics.

    ``step`` k scores the critique of iteration k + 1 against that image.
    """
    tp, tn, fp, fn = (
        counts["true_pos"],
        counts["true_neg"],
        counts["false_pos"],
        counts["false_neg"],
    )
    total = tp + tn + fp + fn

    suffix = f" — critique of iter-{step + 1}"
    print_table(
        "Self-reflection buckets (positive = model says 'looks good')" + suffix,
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
        "Self-reflection metrics" + suffix,
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
    """Prompt, then one line per VQA question with every iteration's score."""
    metadata = record["metadata"]
    lines = [f"Prompt: {metadata.get('prompt', '(missing)')}", ""]
    vqa_list = metadata.get("vqa_list") or []
    for index, vqa in enumerate(vqa_list):
        question, answer = vqa[0], vqa[1]
        scores = " -> ".join(
            "n/a" if s is None else f"{s[index]:.2f}" for s in record["scores"]
        )
        lines.append(f"[{scores}] {question} ({answer})")
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


def save_visualization(iteration_dirs, output_dir, record, index):
    """One row per iteration: text on the left, that iteration's image on the right.

    Row 1 holds the prompt and question scores; row k > 1 holds the critique of
    iteration k - 1 that produced image k.
    """
    prompt_id, image_name = record["prompt_id"], record["image_name"]
    n_iters = len(iteration_dirs)
    fig, axes = plt.subplots(n_iters, 2, figsize=(14, 5 * n_iters), squeeze=False)

    for iteration, iteration_dir in enumerate(iteration_dirs):
        if iteration == 0:
            add_text(axes[0, 0], "Prompt & question scores",
                     format_question_scores(record))
        else:
            step = iteration - 1
            reflection = record["reflections"][step]
            if record["flagged"][step] is None:
                reflection = "(stopped earlier: previous image reused)"
            add_text(axes[iteration, 0], f"Self-reflection on iteration {iteration}",
                     reflection or "(missing)")
        with Image.open(iteration_dir / prompt_id / "samples" / image_name) as image:
            axes[iteration, 1].imshow(image)
        axes[iteration, 1].set_title(iteration_title(record, iteration), fontsize=11)
        axes[iteration, 1].axis("off")

    fig.suptitle(f"Prompt {prompt_id} — sample {image_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{index:03d}_{prompt_id}_{image_name}", dpi=120)
    plt.close(fig)


def render_bucket(iteration_dirs, bucket_dir, records, num_samples, seed):
    """Save up to ``num_samples`` random visualizations from ``records``."""
    bucket_dir.mkdir(parents=True, exist_ok=True)
    chosen = random.Random(seed).sample(records, min(num_samples, len(records)))
    for index, record in enumerate(chosen):
        save_visualization(iteration_dirs, bucket_dir, record, index)
    return chosen


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    iteration_dirs = find_iteration_dirs(args.results_dir)
    if len(iteration_dirs) < 2:
        raise FileNotFoundError(
            f"Need iteration directories 1 and 2 in {args.results_dir}")
    n_iters = len(iteration_dirs)
    records = collect_samples(iteration_dirs, args.correct_threshold)
    if not records:
        raise FileNotFoundError(f"No paired samples found in {args.results_dir}")

    for record in records:
        record["reflections"] = [
            reflection_for_sample(
                iteration_dir / record["prompt_id"] / "self_reflect.txt",
                record["image_name"],
            )
            for iteration_dir in iteration_dirs[1:]
        ]
        # Generation stops a sample once a critique says "looks good"; its later
        # iterations reuse the image and have no critique (flagged = None).
        record["flagged"] = []
        stopped = False
        for reflection in record["reflections"]:
            record["flagged"].append(None if stopped else has_true_iteration(reflection))
            stopped = stopped or "looks good" in reflection.lower()

    render_rows = []

    def render(name, bucket_dir, bucket_records):
        if args.no_images:
            return
        chosen = render_bucket(
            iteration_dirs, bucket_dir, bucket_records, args.num_samples, args.seed
        )
        render_rows.append([name, len(bucket_records), len(chosen), bucket_dir])

    render("all (random sample)", args.output_dir or args.results_dir / "viz",
           records)
    true_iteration = [record for record in records if record["flagged"][0]]
    if true_iteration:
        render("true_iteration", args.results_dir / "viz_true_iteration",
               true_iteration)

    scored = print_score_stats(records, args.delta_threshold, n_iters)
    print_skill_stats(records, n_iters)
    print_object_count_stats(records, n_iters)

    if scored:
        # Confusion buckets use the first critique; delta buckets compare the
        # final iteration against the draft.
        buckets = {"true_pos": [], "true_neg": [], "false_pos": [], "false_neg": []}
        for record in scored:
            buckets[classify_pair(record["correct"][0], record["flagged"][0])].append(
                record)
        delta_buckets = {
            "improved": [
                r for r in scored if r["am"][-1] - r["am"][0] > args.delta_threshold
            ],
            "degraded": [
                r for r in scored if r["am"][-1] - r["am"][0] < -args.delta_threshold
            ],
        }
        for name, bucket_records in {**buckets, **delta_buckets}.items():
            render(name, args.results_dir / name, bucket_records)

    if render_rows:
        print_table(
            "Rendered visualizations",
            ["bucket", "samples", "saved", "directory"],
            render_rows,
        )
    if not scored:
        return

    for step in range(n_iters - 1):
        counts = {"true_pos": 0, "true_neg": 0, "false_pos": 0, "false_neg": 0}
        for record in scored:
            if record["flagged"][step] is not None:
                counts[classify_pair(record["correct"][step],
                                     record["flagged"][step])] += 1
        print_stats(counts, step)


if __name__ == "__main__":
    main()
