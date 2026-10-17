'''
Random sample 15 datapoints from UniCoT-Self-Reflection and visualize by saving to output.

Schema difference from RLHF-V: image is a list of {bytes, path} dicts (multi-image support).
'''

import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import heapq
import io
import os
import random
import textwrap

import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from PIL import Image

NUM_SAMPLES = 50
# SEED = 339
SEED = 8811
MAX_WORKERS = 32


def _sample_parquet_row_group(task):
    """Return lightweight references to the n highest-priority rows."""
    task_index, path, row_group, n, seed = task
    rng = random.Random(seed + task_index)
    candidates = []

    # Keep image data out of worker processes.  Returning selected rows from
    # every row group duplicates large image byte strings over IPC and can OOM.
    pf = pq.ParquetFile(path)
    row_count = pf.metadata.row_group(row_group).num_rows
    for row_index in range(row_count):
        candidate = (rng.random(), task_index, row_index, path, row_group)
        if len(candidates) < n:
            heapq.heappush(candidates, candidate)
        elif candidate[0] > candidates[0][0]:
            heapq.heapreplace(candidates, candidate)

    return candidates, row_count


def _iter_rows(candidates):
    """Yield selected rows one at a time, keeping decoded image memory bounded."""
    for _, _, row_index, path, row_group in candidates:
        parquet_file = pq.ParquetFile(path)
        for index, batch in enumerate(
            parquet_file.iter_batches(batch_size=1, row_groups=[row_group])
        ):
            if index == row_index:
                yield batch.to_pylist()[0]
                break


def load_random_rows(input_dir: str, n: int, seed: int, workers=None):
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    if n <= 0:
        return []

    if workers is None:
        workers = min(MAX_WORKERS, os.cpu_count() or 1)
    tasks = []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for row_group in range(parquet_file.num_row_groups):
            tasks.append((len(tasks), path, row_group, n, seed))
    workers = max(1, min(workers, len(tasks) or 1))

    # Independent deterministic priorities make this a uniform sample. Each
    # row group only needs to return its local top n candidates.
    candidates = []
    total_rows = 0
    if workers == 1:
        partition_samples = map(_sample_parquet_row_group, tasks)
        for partition_candidates, partition_rows in partition_samples:
            total_rows += partition_rows
            for candidate in partition_candidates:
                if len(candidates) < n:
                    heapq.heappush(candidates, candidate)
                elif candidate[0] > candidates[0][0]:
                    heapq.heapreplace(candidates, candidate)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for partition_candidates, partition_rows in executor.map(
                _sample_parquet_row_group, tasks
            ):
                total_rows += partition_rows
                for candidate in partition_candidates:
                    if len(candidates) < n:
                        heapq.heappush(candidates, candidate)
                    elif candidate[0] > candidates[0][0]:
                        heapq.heapreplace(candidates, candidate)

    print("=======================")
    print("Total no. of data samples: ", total_rows)
    print("=======================")
    # Sorting makes output identical for single- and multi-process runs.
    return sorted(candidates, reverse=True)


def conv_to_text(convs) -> str:
    if not convs:
        return "(empty)"
    lines = []
    for turn in convs:
        role = turn.get("from", "?")
        val  = turn.get("value", "")
        lines.append(f"[{role}] {val}")
    return "\n\n".join(lines)


def visualize(rows, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for i, row in enumerate(rows):
        images = row.get("image") or []
        convs       = row.get("conversations", [])
        convs_short = row.get("conversations_short", [])

        n_imgs = max(1, len(images))
        fig, axes = plt.subplots(1, n_imgs + 2, figsize=(6 * (n_imgs + 2), 7))

        # Image panel(s)
        for j in range(n_imgs):
            ax = axes[j]
            if j < len(images) and images[j] and images[j].get("bytes"):
                img = Image.open(io.BytesIO(images[j]["bytes"]))
                ax.imshow(img)
                # `path` is null for images generated in-process (bytes only).
                path = images[j].get("path") or ""
                ax.set_title(f"image[{j}]\n{os.path.basename(path)}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "(no image)", ha="center", va="center")
                ax.set_title(f"image[{j}]", fontsize=8)
            ax.axis("off")

        # conversations and conversations_short panels
        for ax, label, convs_data in [
            (axes[n_imgs],     "conversations",       convs),
            (axes[n_imgs + 1], "conversations_short", convs_short),
        ]:
            text = conv_to_text(convs_data)
            wrapped = "\n".join(
                textwrap.fill(line, width=55) for line in text.splitlines()
            )
            ax.text(0.01, 0.99, wrapped, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", family="monospace")
            ax.axis("off")
            ax.set_title(label, fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"sample_{i:02d}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        if i == 0:
            print(f"Saved {out_path}")


def main(input_dir, output, workers=None):
    selected_rows = load_random_rows(input_dir, NUM_SAMPLES, SEED, workers)
    print(f"Sampled {len(selected_rows)} rows")
    visualize(_iter_rows(selected_rows), output)
    print(f"Done. Outputs in {output}")
    print("=========")


if __name__ == "__main__":
    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/UniCoT-Self-Reflection-6K_tar_format/"
    # output    = "~/edit/UniCoT-Self-Reflection-6K_tar_format/"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/HumanEdit_tar_format/"
    # output = "~/edit/HumanEdit_tar_format/"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/ImgEdit_tar_format_8K"
    # output = "~/ImgEdit_tar_format2/"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/ShareGPT-4o-Image_tar_format/ti2i"
    # output = "~/ti2i/ShareGPT-4o-Image_tar_format/"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_hqedt_slfref_tar"
    # output = "~/edit/gpt-edit-simpler_hqedt_slfref_tar/"
    # main(input_dir, output)

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_hqedt2_slfref_tar"
    # output = "~/edit/gpt-edit-simpler_hqedt2_slfref_tar/"
    # main(input_dir, output)

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_ult_slfref_tar"
    # output = "~/edit/gpt-edit-simpler_ult_slfref_tar/"
    # main(input_dir, output)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(MAX_WORKERS, os.cpu_count() or 1),
        help=f"number of parquet reader processes (default: up to {MAX_WORKERS})",
    )
    args = parser.parse_args()
    main(args.input_dir, args.output, args.workers)

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/"
    # datasets = ["gpt-edit-simpler_ult_slfref_tar"]
    # output_dir = "/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/util_scripts/sft_data_convert_format/viz/~/edit"

    # for d in datasets:
    #     main(os.path.join(input_dir, d, "train"),
    #         os.path.join(output_dir, d))
    #     print("====================\n")
