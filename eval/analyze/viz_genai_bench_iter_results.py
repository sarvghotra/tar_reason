"""Analyze and visualize iterative GenAI-Bench results.

Consumes the layout written by ``eval/iterative_genai_bench_adhoc.py`` and
scored per iteration by t2v_metrics' ``genai_bench.evaluate`` (see
scripts/eval/iter_genai_bench_adhoc.sh)::

    {results_dir}/{1,2}/{prompt_id}.jpeg
    {results_dir}/{1,2}/{score_model}_{num_prompts}_prompts.pt   # VQAScore
    {results_dir}/self_reflect.json    # prompt_id -> reflections

The ``.pt`` tensor has one row per prompt in the order of the dataset's
``genai_image.json`` keys, which is how ``genai_bench.evaluate`` built it.
A prompt counts as correct when its VQAScore is >= ``--correct-threshold``.
"""

import argparse
import json
import re
from pathlib import Path

import torch

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
    "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/genai-bench-800/"
    "iter_adhoc_slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0_512px/seed_13"
)
DATASETS_DIR = Path("/network/scratch/s/sarvjeet-singh.ghotra/git/t2v_metrics/datasets")

# Same grouping and order as genai_bench/evaluate.py's tag_groups.
SKILL_ORDER = [
    "attribute",
    "scene",
    "spatial relation",
    "action relation",
    "part relation",
    "counting",
    "comparison",
    "differentiation",
    "negation",
    "universal",
    "basic",
    "advanced",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize paired outputs from iterative GenAI-Bench generation."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--score-model",
        help="Scorer whose '<score_model>_<N>_prompts.pt' to read. "
        "Inferred from {results_dir}/1 when there is only one.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Directory holding genai_image.json and genai_skills.json "
        f"(default: {DATASETS_DIR}/GenAI-Image-<N>).",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("-n", "--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=8811)
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=0.5,
        help="A prompt counts as correct when its VQAScore is >= this.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.05,
        help="Minimum VQAScore change for a sample to count as improved/degraded.",
    )
    return parser.parse_args()


def find_score_file(iteration_dir, score_model):
    model = re.escape(score_model) if score_model else ".+"
    candidates = [
        p
        for p in iteration_dir.glob("*_prompts.pt")
        if re.fullmatch(rf"{model}_\d+_prompts\.pt", p.name)
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"Found score files {sorted(p.name for p in candidates)} in {iteration_dir}; "
            "pass --score-model"
        )
    return candidates[0] if candidates else None


def load_scores(path):
    """Per-prompt VQAScore list (mean over images of a prompt), or None."""
    if path is None or not path.is_file():
        return None
    scores = torch.load(path, map_location="cpu")
    return scores.reshape(scores.shape[0], -1).float().mean(dim=1).tolist()


def load_skills(dataset_dir):
    try:
        tags = json.loads((dataset_dir / "genai_skills.json").read_text())
    except FileNotFoundError:
        return {}
    skills = {}
    for tag, ids in tags.items():
        for prompt_id in ids:
            skills.setdefault(f"{prompt_id:05d}", []).append(tag)
    return skills


def iteration_title(score, iteration):
    if score is None:
        return f"Iteration {iteration + 1}"
    return f"Iteration {iteration + 1} — VQAScore {format_score(score)}"


def collect_samples(results_dir, dataset_dir, score_paths, correct_threshold):
    """One record per benchmark prompt whose image exists in both iterations."""
    prompts = json.loads((dataset_dir / "genai_image.json").read_text())
    skills = load_skills(dataset_dir)
    scores = [load_scores(path) for path in score_paths]
    for path, score_list in zip(score_paths, scores):
        if score_list is not None and len(score_list) != len(prompts):
            raise ValueError(
                f"{path} has {len(score_list)} rows but {dataset_dir}/genai_image.json "
                f"has {len(prompts)} prompts"
            )
    try:
        reflections = json.loads((results_dir / "self_reflect.json").read_text())
    except FileNotFoundError:
        reflections = {}

    records = []
    for index, (prompt_id, item) in enumerate(prompts.items()):
        images = tuple(results_dir / it / f"{prompt_id}.jpeg" for it in ("1", "2"))
        if not all(image.is_file() for image in images):
            continue
        score = [None if s is None else s[index] for s in scores]
        tags = skills.get(prompt_id, [])
        entry = reflections.get(prompt_id) or {}
        reflection = ((entry.get("reflections") or [None])[0] or "").strip()
        info = [f"Prompt: {item['prompt']}", "", f"Skills: {', '.join(tags) or '(none)'}"]
        records.append(
            {
                "key": prompt_id,
                "skills": tags,
                "images": images,
                "score": score,
                "correct": [None if s is None else s >= correct_threshold for s in score],
                "reflection": reflection,
                "flagged": has_true_iteration(reflection),
                "info": "\n".join(info),
                "titles": [iteration_title(s, i) for i, s in enumerate(score)],
            }
        )
    return records


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    score_paths = [
        find_score_file(args.results_dir / it, args.score_model) for it in ("1", "2")
    ]
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        found = next((p for p in score_paths if p is not None), None)
        if found is None:
            raise FileNotFoundError(
                f"No *_prompts.pt under {args.results_dir}/{{1,2}}; pass --dataset-dir "
                "to render without scores"
            )
        num_prompts = re.search(r"_(\d+)_prompts\.pt$", found.name).group(1)
        dataset_dir = DATASETS_DIR / f"GenAI-Image-{num_prompts}"

    records = collect_samples(
        args.results_dir, dataset_dir, score_paths, args.correct_threshold
    )
    if not records:
        raise FileNotFoundError(f"No paired samples found in {args.results_dir}")
    scored = scored_records(records)

    scorer = next((p.name for p in score_paths if p is not None), "no scores")
    print_coverage(f"GenAI-Bench coverage ({scorer})", records, scored)
    if scored:
        print_overall(
            f"GenAI-Bench VQAScore (correct = score >= {args.correct_threshold})",
            scored,
        )
        print_movement(scored, args.delta_threshold)
        print_reflection_vs_delta(scored)
        groups = {
            skill: [r for r in scored if skill in r["skills"]] for skill in SKILL_ORDER
        }
        groups["all"] = scored
        print_group_table("VQAScore by skill", "skill", groups)
    else:
        print("No VQAScore files found; skipping score stats.")

    output_dir = args.output_dir or args.results_dir / "viz"
    render_and_report(
        records, scored, output_dir, args.num_samples, args.seed, args.delta_threshold
    )


if __name__ == "__main__":
    main()
