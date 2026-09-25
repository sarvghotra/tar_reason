"""Benchmark-agnostic half of the iterative-generation analysis scripts.

Each benchmark script (viz_tiif_bench_iter_results.py,
viz_genai_bench_iter_results.py) turns its own result layout into a list of
records with these fields, and hands them to the functions below:

    key         unique, printable prompt id (used in rendered file names)
    images      (iter-1 path, iter-2 path)
    score       (iter-1, iter-2) prompt-level score in [0, 1], None if unscored
    correct     (iter-1, iter-2) bool, None if unscored
    reflection  self-reflection text written between the two passes
    flagged     whether the reflection asked for a correction
    info        text for the top-left panel of the rendered figure
    titles      (iter-1 title, iter-2 title) for the rendered figure

The table and reflection helpers are shared with viz_geneval2_iter_results.py.
"""

import random
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from viz_geneval2_iter_results import (  # noqa: F401  (re-exported)
    classify_pair,
    format_score,
    has_true_iteration,
    mean,
    print_stats,
    print_table,
    ratio,
)


def scored_records(records):
    return [r for r in records if r["score"][0] is not None and r["score"][1] is not None]


def mean_score(records, iteration):
    return mean([r["score"][iteration] for r in records])


def print_coverage(title, records, scored):
    print_table(
        title,
        ["quantity", "prompts"],
        [["Prompts with samples", len(records)], ["Prompts with scores", len(scored)]],
    )


def print_overall(title, scored, rows_before=()):
    """Mean prompt score and strict correctness for both iterations."""
    rows = [list(row) for row in rows_before]
    first, second = mean_score(scored, 0), mean_score(scored, 1)
    rows.append(
        [
            "Mean prompt score",
            format_score(first),
            format_score(second),
            format_score(second - first),
        ]
    )
    correct1 = sum(r["correct"][0] for r in scored)
    correct2 = sum(r["correct"][1] for r in scored)
    rows.append(
        [
            "Prompts correct",
            f"{correct1} ({ratio(correct1, len(scored))})",
            f"{correct2} ({ratio(correct2, len(scored))})",
            f"{correct2 - correct1:+d}",
        ]
    )
    print_table(title, ["metric", "iter-1", "iter-2", "delta"], rows)


def print_group_table(title, group_header, groups, metric=mean_score):
    """One row per group: size, ``metric`` for both iterations and the delta."""
    rows = []
    for name, group in groups.items():
        if not group:
            continue
        first, second = metric(group, 0), metric(group, 1)
        rows.append(
            [
                name,
                len(group),
                format_score(first),
                format_score(second),
                format_score(second - first),
            ]
        )
    if rows:
        print_table(title, [group_header, "prompts", "iter-1", "iter-2", "delta"], rows)


def delta_buckets(scored, delta_threshold):
    deltas = {id(r): r["score"][1] - r["score"][0] for r in scored}
    return {
        "improved": [r for r in scored if deltas[id(r)] > delta_threshold],
        "degraded": [r for r in scored if deltas[id(r)] < -delta_threshold],
    }


def print_movement(scored, delta_threshold):
    buckets = delta_buckets(scored, delta_threshold)
    improved, degraded = len(buckets["improved"]), len(buckets["degraded"])
    unchanged = len(scored) - improved - degraded
    fixed = sum(not r["correct"][0] and r["correct"][1] for r in scored)
    broken = sum(r["correct"][0] and not r["correct"][1] for r in scored)
    print_table(
        f"Per-prompt movement (score delta threshold |{delta_threshold}|)",
        ["bucket", "count", "share"],
        [
            ["Improved", improved, ratio(improved, len(scored))],
            ["Degraded", degraded, ratio(degraded, len(scored))],
            ["Unchanged", unchanged, ratio(unchanged, len(scored))],
            ["Wrong -> correct", fixed, ratio(fixed, len(scored))],
            ["Correct -> wrong", broken, ratio(broken, len(scored))],
        ],
    )


def print_reflection_vs_delta(scored):
    rows = []
    for name, flagged in (("flagged (model edits)", True), ("kept as-is", False)):
        group = [r for r in scored if r["flagged"] == flagged]
        delta = mean([r["score"][1] - r["score"][0] for r in group])
        rows.append([name, len(group), format_score(delta)])
    print_table(
        "Reflection decision vs. score delta",
        ["group", "prompts", "mean score delta"],
        rows,
    )


def add_text(ax, title, value):
    wrapped = "\n".join(
        textwrap.fill(line, width=58, replace_whitespace=False)
        for line in value.splitlines()
    )
    ax.set_title(title, fontsize=11)
    # Prompts and reflections are free text ("$50.00"); never parse them as mathtext.
    ax.text(
        0.01, 0.99, wrapped, transform=ax.transAxes, va="top", fontsize=8,
        parse_math=False,
    )
    ax.axis("off")


def save_visualization(record, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    add_text(axes[0, 0], "Prompt & scores", record["info"])
    add_text(axes[1, 0], "Self-reflection", record["reflection"] or "(missing)")
    for row, (image_path, title) in enumerate(zip(record["images"], record["titles"])):
        with Image.open(image_path) as image:
            axes[row, 1].imshow(image)
        axes[row, 1].set_title(title, fontsize=11)
        axes[row, 1].axis("off")
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


def render_and_report(records, scored, output_dir, num_samples, seed, delta_threshold):
    """Render every bucket under ``output_dir`` and print the confusion matrix.

    Buckets: a random sample of all prompts, prompts the reflection flagged,
    the four self-reflection confusion buckets and the improved/degraded sets.
    """
    buckets = {"random": records, "true_iteration": [r for r in records if r["flagged"]]}
    confusion = {"true_pos": [], "true_neg": [], "false_pos": [], "false_neg": []}
    for record in scored:
        confusion[classify_pair(record["correct"][0], record["flagged"])].append(record)
    if scored:
        buckets.update(confusion)
        buckets.update(delta_buckets(scored, delta_threshold))

    rows = []
    for name, bucket_records in buckets.items():
        if not bucket_records:
            rows.append([name, 0, 0, "-"])
            continue
        bucket_dir = output_dir / name
        chosen = render_bucket(bucket_dir, bucket_records, num_samples, seed)
        rows.append([name, len(bucket_records), len(chosen), bucket_dir])
    print_table(
        "Rendered visualizations",
        ["bucket", "samples", "saved", "directory"],
        rows,
    )
    if scored:
        print_stats({name: len(items) for name, items in confusion.items()})
