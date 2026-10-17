'''
Categorize geneval2 tar_format data by "atom_count" and visualize a sample per category.

The parquet rows carry a "prompt_index" that indexes into the line order of
evaluation_metadata.jsonl, which is where "atom_count" lives. So the parquet
files are scanned for their index columns only (cheap, columnar), joined against
the metadata, sampled per atom_count, and only the row groups holding the chosen
rows are read back in full.

Output: <output_root>/<atom_count>/sample_XX.png (+ prompts.txt)
'''

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import glob
import io
import json
import os
import random
import textwrap

import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from PIL import Image

INPUT_DIR = os.path.expanduser(
    "~/scratch/data/T2I_datasets/geneval2_50K/tar_format")
METADATA = os.path.expanduser(
    "~/scratch/data/T2I_datasets/geneval2_50K/evaluation_metadata.jsonl")
OUTPUT_ROOT = ("/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/"
               "util_scripts/sft_data_convert_format/viz/output/t2i/geneval2_50K")

NUM_SAMPLES = 20
SEED = 89977
MAX_WORKERS = 32


def load_metadata(path):
    """Line i of evaluation_metadata.jsonl is prompt_index i."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def _scan_indices(path):
    """Return (path, row_group_sizes, prompt_index list, sample_index list)."""
    pf = pq.ParquetFile(path)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    table = pq.read_table(path, columns=["prompt_index", "sample_index"])
    return (path, sizes,
            table.column("prompt_index").to_pylist(),
            table.column("sample_index").to_pylist())


def build_index(input_dir, metadata, workers):
    """Group every row in the dataset by the atom_count of its prompt."""
    # shard_XXXXX.parquet without a rank suffix duplicates the rank files of the
    # same shard, so only the rank files are read.
    files = sorted(glob.glob(os.path.join(input_dir, "*.rank*.parquet")))
    if not files:
        raise FileNotFoundError(f"No rank parquet files found in {input_dir}")

    workers = max(1, min(workers, len(files)))
    if workers == 1:
        scans = map(_scan_indices, files)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        scans = executor.map(_scan_indices, files)

    by_atom = defaultdict(list)
    rg_sizes = {}
    total = 0
    try:
        for path, sizes, prompt_indices, sample_indices in scans:
            rg_sizes[path] = sizes
            total += len(prompt_indices)
            for row, (pi, si) in enumerate(zip(prompt_indices, sample_indices)):
                by_atom[metadata[pi]["atom_count"]].append((path, row, pi, si))
    finally:
        if executor is not None:
            executor.shutdown()

    print(f"Scanned {len(files)} parquet files, {total} rows")
    for atom in sorted(by_atom):
        print(f"  atom_count={atom}: {len(by_atom[atom])} rows")
    return by_atom, rg_sizes


def _row_group_of(rg_sizes, row):
    start = 0
    for rg, size in enumerate(rg_sizes):
        if row < start + size:
            return rg, row - start
        start += size
    raise IndexError(row)


def _fetch_rows(task):
    """Read one row group and pull out the selected rows."""
    path, row_group, wanted = task  # wanted: list of (offset, key)
    pf = pq.ParquetFile(path)
    offsets = {offset: key for offset, key in wanted}
    out = []
    position = 0
    for batch in pf.iter_batches(batch_size=32, row_groups=[row_group]):
        rows = batch.to_pylist()
        for row in rows:
            key = offsets.get(position)
            if key is not None:
                out.append((key, row))
            position += 1
        if len(out) == len(offsets):
            break
    return out


def fetch(selections, rg_sizes, workers):
    """selections: list of (key, path, row). Returns {key: row dict}."""
    grouped = defaultdict(list)
    for key, path, row in selections:
        rg, offset = _row_group_of(rg_sizes[path], row)
        grouped[(path, rg)].append((offset, key))

    tasks = [(path, rg, wanted) for (path, rg), wanted in grouped.items()]
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        results = map(_fetch_rows, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_fetch_rows, tasks)

    fetched = {}
    try:
        for chunk in results:
            for key, row in chunk:
                fetched[key] = row
    finally:
        if executor is not None:
            executor.shutdown()
    return fetched


def conv_to_text(convs) -> str:
    lines = []
    for turn in convs or []:
        role = turn.get("from", "?")
        val = turn.get("value", "")
        lines.append(f"[{role}] {val}")
    return "\n\n".join(lines)


def meta_to_text(meta) -> str:
    lines = [
        f"prompt: {meta['prompt']}",
        f"atom_count: {meta['atom_count']}",
        f"skills: {', '.join(meta.get('skills', []))}",
        "",
        "vqa_list:",
    ]
    for question, answer in meta.get("vqa_list", []):
        lines.append(f"  Q: {question}")
        lines.append(f"  A: {answer}")
    return "\n".join(lines)


def visualize(items, output_dir):
    """items: list of (row, meta) already ordered."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (row, meta) in enumerate(items):
        img_obj = row.get("image")
        img_bytes = img_obj.get("bytes") if isinstance(img_obj, dict) else None

        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        ax_img = axes[0]
        if img_bytes:
            ax_img.imshow(Image.open(io.BytesIO(img_bytes)))
        else:
            ax_img.text(0.5, 0.5, "(no image)", ha="center", va="center")
        ax_img.axis("off")
        ax_img.set_title(
            f"prompt_index={row.get('prompt_index')} "
            f"sample_index={row.get('sample_index')}", fontsize=9)

        for ax, label, text in [
            (axes[1], "conversations", conv_to_text(row.get("conversations"))),
            (axes[2], "metadata", meta_to_text(meta)),
        ]:
            wrapped = "\n".join(
                textwrap.fill(line, width=55) for line in text.splitlines())
            ax.text(0.01, 0.99, wrapped, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", family="monospace", wrap=True)
            ax.axis("off")
            ax.set_title(label, fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{i:02d}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    prompts_file = os.path.join(output_dir, "prompts.txt")
    with open(prompts_file, "w") as f:
        for i, (row, meta) in enumerate(items):
            f.write(f"=== sample_{i:02d} "
                    f"(prompt_index={row.get('prompt_index')}, "
                    f"sample_index={row.get('sample_index')}) ===\n")
            f.write(conv_to_text(row.get("conversations")))
            f.write("\n\n")
            f.write(meta_to_text(meta))
            f.write("\n\n")
    print(f"Saved {len(items)} samples to {output_dir}")


def main(input_dir, metadata_path, output_root, n, seed, workers):
    metadata = load_metadata(metadata_path)
    by_atom, rg_sizes = build_index(input_dir, metadata, workers)

    rng = random.Random(seed)
    selections = []
    per_atom = {}
    for atom in sorted(by_atom):
        rows = by_atom[atom]
        chosen = rng.sample(rows, min(n, len(rows)))
        per_atom[atom] = chosen
        for j, (path, row, _, _) in enumerate(chosen):
            selections.append(((atom, j), path, row))

    fetched = fetch(selections, rg_sizes, workers)

    for atom in sorted(per_atom):
        items = []
        for j, (_, _, prompt_index, _) in enumerate(per_atom[atom]):
            row = fetched.get((atom, j))
            if row is not None:
                items.append((row, metadata[prompt_index]))
        visualize(items, os.path.join(output_root, str(atom)))
    print(f"Done. Outputs in {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--metadata", default=METADATA)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--workers", type=int,
        default=min(MAX_WORKERS, os.cpu_count() or 1),
        help=f"number of parquet reader processes (default: up to {MAX_WORKERS})")
    args = parser.parse_args()
    main(args.input_dir, args.metadata, args.output_root,
         args.num_samples, args.seed, args.workers)
