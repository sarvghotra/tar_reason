"""Create contact sheets for randomly selected iterative GenEval outputs."""

import argparse
import json
import random
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


RESULTS_DIR = Path(
    "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/geneval/"
    "iter_slf_ref_edit_t7_S0_short0.7_6K_512px_repeat1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize paired outputs from iterative GenEval generation."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("-n", "--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=8811)
    parser.add_argument("--eval_type", type=str, default='geneval')
    return parser.parse_args()


def find_sample_pairs(results_dir):
    """Return (prompt id, image name) pairs present in both iterations."""
    first, second = results_dir / "1", results_dir / "2"
    pairs = []
    for prompt_dir in first.iterdir() if first.is_dir() else ():
        if not prompt_dir.is_dir():
            continue
        second_samples = second / prompt_dir.name / "samples"
        first_samples = prompt_dir / "samples"
        if not second_samples.is_dir() or not first_samples.is_dir():
            continue
        second_names = {path.name for path in second_samples.glob("*.png")}
        pairs.extend(
            (prompt_dir.name, path.name)
            for path in first_samples.glob("*.png")
            if path.name in second_names
        )
    return pairs


def find_genai_bench_samples(results_dir):
    """Return image names present in both iterations, in prompt-id order.

    GenAI-Bench results are flat: ``<results_dir>/{1,2}/<prompt id>.jpeg`` plus a
    single top-level ``self_reflect.json``. Globs come back in arbitrary order,
    so match by name instead of zipping two directory listings.
    """
    first, second = results_dir / "1", results_dir / "2"
    second_names = {path.name for path in second.glob("*.jpeg")}
    return sorted(
        path.name for path in first.glob("*.jpeg") if path.name in second_names
    )


def load_genai_bench_reflections(results_dir):
    """Load ``self_reflect.json`` mapping prompt id -> {prompt, reflections}."""
    path = results_dir / "self_reflect.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def read_text(path):
    try:
        return path.read_text().strip() or "(empty)"
    except FileNotFoundError:
        return "(missing)"


def format_metadata(path):
    raw = read_text(path)
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw


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


def load_first_iteration_correctness(results_dir):
    """Map (prompt id, image name) -> bool from GenEval's ``1/results.jsonl``."""
    path = results_dir / "1" / "results.jsonl"
    correctness = {}
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return correctness
    for line in lines:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        image = Path(record["filename"])
        # <results dir>/1/<prompt id>/samples/<image name>
        correctness[(image.parent.parent.name, image.name)] = bool(record["correct"])
    return correctness


def classify_pair(is_correct, flagged_issue):
    """Confusion-matrix bucket for one sample.

    "Positive" is the model predicting the first-pass image needs no fixing.
    """
    if is_correct:
        return "true_pos" if not flagged_issue else "false_neg"
    return "false_pos" if not flagged_issue else "true_neg"


def print_stats(counts):
    """Print the self-reflection confusion matrix and derived metrics."""
    tp, tn, fp, fn = (
        counts["true_pos"],
        counts["true_neg"],
        counts["false_pos"],
        counts["false_neg"],
    )
    total = tp + tn + fp + fn

    def ratio(numerator, denominator):
        return f"{numerator / denominator:.2%}" if denominator else "n/a"

    print("\nSelf-reflection stats (positive = model says 'looks good')")
    print("=" * 58)
    print(f"Total samples:                {total}")
    print(f"1st-pass correct images:      {tp + fn} ({ratio(tp + fn, total)})")
    print(f"Reflections saying looks-good: {tp + fp} ({ratio(tp + fp, total)})")
    print()
    print(f"True positives  (correct, looks good): {tp} ({ratio(tp, total)})")
    print(f"False negatives (correct, flagged):    {fn} ({ratio(fn, total)})")
    print(f"True negatives  (wrong, flagged):      {tn} ({ratio(tn, total)})")
    print(f"False positives (wrong, looks good):   {fp} ({ratio(fp, total)})")
    print()
    print(f"Accuracy:    {ratio(tp + tn, total)}")
    print(f"Precision:   {ratio(tp, tp + fp)}")
    print(f"Recall:      {ratio(tp, tp + fn)}")
    print(f"Specificity: {ratio(tn, tn + fp)}")
    f1_denominator = 2 * tp + fp + fn
    print(f"F1:          {ratio(2 * tp, f1_denominator)}")


def add_text(ax, title, value):
    wrapped = "\n".join(
        textwrap.fill(line, width=58, replace_whitespace=False)
        for line in value.splitlines()
    )
    ax.set_title(title, fontsize=11)
    ax.text(0.01, 0.99, wrapped, transform=ax.transAxes, va="top", fontsize=8)
    ax.axis("off")


def save_visualization(results_dir, output_dir, prompt_id, image_name, index):
    first_dir = results_dir / "1" / prompt_id
    second_dir = results_dir / "2" / prompt_id
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    add_text(axes[0, 0], "Metadata", format_metadata(first_dir / "metadata.jsonl"))
    with Image.open(first_dir / "samples" / image_name) as image:
        axes[0, 1].imshow(image)
    axes[0, 1].set_title("Iteration 1", fontsize=11)
    axes[0, 1].axis("off")

    add_text(
        axes[1, 0],
        "Self-reflection",
        reflection_for_sample(second_dir / "self_reflect.txt", image_name),
    )
    with Image.open(second_dir / "samples" / image_name) as image:
        axes[1, 1].imshow(image)
    axes[1, 1].set_title("Iteration 2", fontsize=11)
    axes[1, 1].axis("off")

    fig.suptitle(f"Prompt {prompt_id} — sample {image_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{index:03d}_{prompt_id}_{image_name}", dpi=120)
    plt.close(fig)


def genai_bench_reflection(reflections, prompt_id):
    """First-pass reflection text for a GenAI-Bench prompt id."""
    entry = reflections.get(prompt_id) or {}
    texts = entry.get("reflections") or []
    return texts[0].strip() if texts else ""


def save_genai_bench_visualization(
    results_dir, output_dir, reflections, prompt_id, image_name, index
):
    entry = reflections.get(prompt_id) or {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    add_text(axes[0, 0], "Prompt", entry.get("prompt", "(missing)"))
    with Image.open(results_dir / "1" / image_name) as image:
        axes[0, 1].imshow(image)
    axes[0, 1].set_title("Iteration 1", fontsize=11)
    axes[0, 1].axis("off")

    add_text(
        axes[1, 0],
        "Self-reflection",
        genai_bench_reflection(reflections, prompt_id) or "(missing)",
    )
    with Image.open(results_dir / "2" / image_name) as image:
        axes[1, 1].imshow(image)
    axes[1, 1].set_title("Iteration 2", fontsize=11)
    axes[1, 1].axis("off")

    fig.suptitle(f"Prompt {prompt_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{index:03d}_{prompt_id}.png", dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    genai_bench = args.eval_type == 'genai-bench'
    if genai_bench:
        reflections = load_genai_bench_reflections(args.results_dir)
        pairs = [
            (Path(name).stem, name)
            for name in find_genai_bench_samples(args.results_dir)
        ]

        def render(output_dir, prompt_id, image_name, index):
            save_genai_bench_visualization(
                args.results_dir, output_dir, reflections, prompt_id, image_name, index
            )

        def reflection_of(prompt_id, image_name):
            return genai_bench_reflection(reflections, prompt_id)
    else:
        pairs = find_sample_pairs(args.results_dir)

        def render(output_dir, prompt_id, image_name, index):
            save_visualization(
                args.results_dir, output_dir, prompt_id, image_name, index
            )

        def reflection_of(prompt_id, image_name):
            return reflection_for_sample(
                args.results_dir / "2" / prompt_id / "self_reflect.txt", image_name
            )

    if not pairs:
        raise FileNotFoundError(f"No paired samples found in {args.results_dir}")
    selected = random.Random(args.seed).sample(pairs, min(args.num_samples, len(pairs)))
    output_dir = args.output_dir or args.results_dir / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (prompt_id, image_name) in enumerate(selected):
        render(output_dir, prompt_id, image_name, index)

    true_iteration_pairs = [
        (prompt_id, image_name)
        for prompt_id, image_name in pairs
        if has_true_iteration(reflection_of(prompt_id, image_name))
    ]
    true_iteration_output_dir = args.results_dir / "viz_true_iteration"
    true_iteration_output_dir.mkdir(parents=True, exist_ok=True)
    selected_true_iterations = random.Random(args.seed).sample(
        true_iteration_pairs, min(args.num_samples, len(true_iteration_pairs))
    )
    for index, (prompt_id, image_name) in enumerate(selected_true_iterations):
        render(true_iteration_output_dir, prompt_id, image_name, index)

    print(f"Saved {len(selected)} visualizations to {output_dir}")
    print(
        f"Saved {len(selected_true_iterations)} true-iteration visualizations to "
        f"{true_iteration_output_dir}"
    )

    correctness = {} if genai_bench else load_first_iteration_correctness(args.results_dir)
    if not correctness:
        print(
            "Skipping confusion-matrix visualizations: no 1/results.jsonl in "
            f"{args.results_dir}"
        )
        return

    buckets = {"true_pos": [], "true_neg": [], "false_pos": [], "false_neg": []}
    for prompt_id, image_name in pairs:
        is_correct = correctness.get((prompt_id, image_name))
        if is_correct is None:
            continue
        flagged = has_true_iteration(reflection_of(prompt_id, image_name))
        buckets[classify_pair(is_correct, flagged)].append((prompt_id, image_name))

    for bucket, bucket_pairs in buckets.items():
        bucket_dir = args.results_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        chosen = random.Random(args.seed).sample(
            bucket_pairs, min(args.num_samples, len(bucket_pairs))
        )
        for index, (prompt_id, image_name) in enumerate(chosen):
            render(bucket_dir, prompt_id, image_name, index)
        print(
            f"{bucket}: {len(bucket_pairs)} samples, saved {len(chosen)} "
            f"visualizations to {bucket_dir}"
        )

    print_stats({name: len(items) for name, items in buckets.items()})


if __name__ == "__main__":
    main()
