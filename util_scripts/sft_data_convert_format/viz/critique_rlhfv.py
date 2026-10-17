'''
Random sample 5 datapoints and visualize by saving to output.

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

NUM_SAMPLES = 20
SEED = 89977
MAX_WORKERS = 32


def _sample_parquet_row_group(task):
    """Return the n highest-priority rows in one parquet row group."""
    task_index, path, row_group, n, seed = task
    rng = random.Random(seed + task_index)
    candidates = []
    row_index = 0

    # A worker returns at most n rows, keeping inter-process transfers small
    # even when the row group contains many images.
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=32, row_groups=[row_group]):
        for row in batch.to_pylist():
            candidate = (rng.random(), task_index, row_index, row)
            if len(candidates) < n:
                heapq.heappush(candidates, candidate)
            elif candidate[0] > candidates[0][0]:
                heapq.heapreplace(candidates, candidate)
            row_index += 1

    return candidates


def load_random_rows(input_dir: str, n: int, seed: int, workers=None):
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    if n <= 0:
        return []

    if workers is None:
        workers = min(MAX_WORKERS, os.cpu_count() or 1)
    tasks = []
    total_rows = 0
    for path in files:
        parquet_file = pq.ParquetFile(path)
        total_rows += parquet_file.metadata.num_rows
        for row_group in range(parquet_file.num_row_groups):
            tasks.append((len(tasks), path, row_group, n, seed))
    print(f"Total dataset size: {total_rows} rows across {len(files)} files")
    workers = max(1, min(workers, len(tasks) or 1))

    # Every row receives an independent deterministic priority. The global top
    # n rows form a uniform sample; a row group only needs to return its local
    # top n because no lower-priority row from it can enter the global top n.
    if workers == 1:
        partition_samples = map(_sample_parquet_row_group, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        partition_samples = executor.map(_sample_parquet_row_group, tasks)

    candidates = []
    try:
        for partition_candidates in partition_samples:
            for candidate in partition_candidates:
                if len(candidates) < n:
                    heapq.heappush(candidates, candidate)
                elif candidate[0] > candidates[0][0]:
                    heapq.heapreplace(candidates, candidate)
    finally:
        if workers > 1:
            executor.shutdown()

    # Sorting also makes output identical for single- and multi-process runs.
    return [candidate[3] for candidate in sorted(candidates, reverse=True)]


def conv_to_text(convs) -> str:
    lines = []
    for turn in convs:
        role = turn.get("from", "?")
        val  = turn.get("value", "")
        lines.append(f"[{role}] {val}")
    return "\n\n".join(lines)


def visualize(rows, output_dir: str):
    prompts = []
    short_prompts = []
    os.makedirs(output_dir, exist_ok=True)
    for i, row in enumerate(rows):
        img_obj = row.get("image")
        img_bytes = None
        if isinstance(img_obj, dict):
            img_bytes = img_obj.get("bytes")

        convs       = row.get("conversations", [])
        convs_short = row.get("conversations_short", [])

        prompts.append(convs)
        short_prompts.append(convs_short)

        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        # Image
        ax_img = axes[0]
        if img_bytes:
            img = Image.open(io.BytesIO(img_bytes))
            ax_img.imshow(img)
        else:
            ax_img.text(0.5, 0.5, "(no image)", ha="center", va="center")
        ax_img.axis("off")
        ax_img.set_title("image", fontsize=9)

        # conversations
        for ax, label, convs_data in [
            (axes[1], "conversations",       convs),
            (axes[2], "conversations_short", convs_short),
        ]:
            text = conv_to_text(convs_data)
            wrapped = "\n".join(
                textwrap.fill(line, width=55) for line in text.splitlines()
            )
            ax.text(0.01, 0.99, wrapped, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", family="monospace",
                    wrap=True)
            ax.axis("off")
            ax.set_title(label, fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"sample_{i:02d}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        if i == 0:
            print(f"Saved {out_path}")

    all_prompts = prompts + short_prompts
    prompts_file = os.path.join(output_dir, "prompts.txt")
    with open(prompts_file, "w") as f:
        for i, prompt in enumerate(all_prompts):
            f.write(f"=== Prompt {i} ===\n")
            f.write(conv_to_text(prompt))
            f.write("\n\n")
    print(f"Saved prompts to {prompts_file}")


def main(input_dir, output, seed=None, workers=None):
    rows = load_random_rows(input_dir, NUM_SAMPLES, seed, workers)
    print(f"Sampled {len(rows)} rows")
    visualize(rows, output)
    print(f"Done. Outputs in {output}")


if __name__ == "__main__":
    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/RLHF-V-Dataset_tar_format/"
    # output    = "/tmp/RLHF-V-Dataset_tar_format/"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/RLAIF-V-Dataset_tar_format"
    # output    = "/tmp/RLAIF-V-Dataset_tar_format/"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/SeeTRUE-Feedback_tar_format/"
    # output    = "~/SeeTRUE-Feedback_tar_format/"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/VISCO_tar_format/"
    # output = "/tmp/VISCO_tar_format"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/JourneyDB_tar_format_200K_no_selfreflect"
    # output = "~/JourneyDB_tar_format_200K_no_selfreflect"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/data_1024_10K_tar_format/"
    # output = "~/data_1024_10K_tar_format"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/geneval2_400/critic_data/"
    # output = "~/geneval_400"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/text-to-image-2M_tar_format/val"
    # output = "~/text-to-image-2M_tar_format"

    # input_dir = "/tmp/text-to-image-2M_tar_format_shard_00029_0009/"
    # output = "~/text-to-image-2M_tar_format_shard_00029_0009"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/FLUX-Reason-6M_tar_format"
    # output = "~/FLUX-Reason-6M_tar_format"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/FLUX-Reason-1.4M_tar_format"
    # output = "~/FLUX-Reason-1.4M_tar_format"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/FLUX-Reason_400/critic_data/"
    # output = "~/FLUX-Reason_400_critic_data"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/midjourney-prompts-FLUX_subsets_se1_2M/critic_data/"
    # output = "~/midjourney-prompts-FLUX_subsets_se1_2M_critic_data"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/midjourney-prompts-FLUXset400K_100K/critic_data/"
    # output = "~/midjourney-prompts-FLUXset400K_100K_critic_data"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/JourneyDBval_82K/critic_data/"
    # output = "~/JourneyDBval_82K_critic_data"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/BLIP3o-60k_tar_format"
    # output = "~/BLIP3o-60k_tar_format"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/FLUX-Reason-1.2M/critic_data/"
    # output = "~/FLUX-Reason-1.2M_critic_data"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/BLIP3o-60k/critic_data/"
    # output = "~/BLIP3o-60k_critic_data/"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/geneval_10_flux.1dev"
    # output = "~/geneval_10_flux.1dev"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/FLUX-Reason-1.2M_tar_format"
    # output = "~/FLUX-Reason-1.2M_tar_format"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/BLIP3o-60k_20K/critic_data"
    # output = "~/slf_reflt/BLIP3o-60k_20K"
    # main(input_dir, output)
    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/midjourney-prompts-FLUX_subsets/critic_data"
    # output = "~/slf_reflt/midjourney-prompts-FLUX_subsets"
    # main(input_dir, output)
    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/FLUX-Reason-1.2M_10K/critic_data"
    # output = "~/slf_reflt/FLUX-Reason-1.2M_10K"
    # main(input_dir, output)

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/generated/geneval_34K/critic_data_13K"
    # output = "~/slf_reflt/gen_geneval_34K"
    # main(input_dir, output)
    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/generated/FLUX-Reason-6M_40K/critic_data"
    # output = "~/slf_reflt/gen_FLUX-Reason-6M_40K"
    # main(input_dir, output)

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/generated/t2i-2M_dalle3_40K/critic_data"
    # output = "~/slf_reflt/gen_t2i-2M_dalle3_40K"
    # main(input_dir, output)
    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/generated/t2i-2M_llava_pretrain_40K/critic_data"
    # output = "~/slf_reflt/gen_t2i-2M_llava_pretrain_40K"
    # main(input_dir, output)

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/ShareGPT-4o-Image_tar_format/t2i"
    # output = "~/t2i/ShareGPT-4o-Image_tar_format"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/mdj400K_120K/critic_data/"
    # output = "~/slf_reflt/mdj400K_120K"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/filt/FLUX-Reason-1.2M_50K/"
    # output = "~/slf_reflt/FLUX-Reason-1.2M_50K"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/filt/BLIP3o-60K/"
    # output = "~/slf_reflt/BLIP3o-60K"

    # dataset = "JDBval_20K"
    # input_dir = f"/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/{dataset}/"
    # output = f"~/slf_reflt/{dataset}"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/Tar/ImageNet1K-T2I-QwenVL-FLUX"
    # output = "~/t2i/ImageNet1K-T2I-QwenVL-FLUX"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/mdj400K_120K/critic_data"
    # output = "~/slf_reflt/mdj400K_120K"

    # input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/Tar/ImageNet1K-T2I-QwenVL-FLUX"
    # output = "~/t2i/ImageNet1K-T2I-QwenVL-FLUX"
    # main(input_dir, output)

    # input_dirs = [
    #     "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/generated/geneval_34K/critic_data_13K",
    #     "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/generated/FLUX-Reason-6M_40K/critic_data",
    #     "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/generated/t2i-2M_llava_pretrain_40K/critic_data",
    #     "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/filt/t2i-2M_llava_nxt_ft_10K",
    #     "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/mdj400K_120K/critic_data",
    #     "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/t2i-2M_llava_pt_30K/critic_data",
    #     "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/JDBval_20K"
    # ]
    # outputs = [
    #     "~/slf_reflt/gen/geneval_34K_critic_data_13K",
    #     "~/slf_reflt/gen/FLUX-Reason-6M_40K_critic_data",
    #     "~/slf_reflt/gen/t2i-2M_llava_pretrain_40K_critic_data",
    #     "~/slf_reflt/t2i-2M_llava_nxt_ft_10K",
    #     "~/slf_reflt/mdj400K_120K",
    #     "~/slf_reflt/t2i-2M_llava_pt_30K",
    #     "~/slf_reflt/JDBval_20K"
    # ]

    # for input_dir, output in zip(input_dirs, outputs):
    #     main(input_dir, output)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(MAX_WORKERS, os.cpu_count() or 1),
        help=f"number of parquet reader processes (default: up to {MAX_WORKERS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )
    args = parser.parse_args()
    main(args.input_dir, args.output, args.seed, args.workers)
