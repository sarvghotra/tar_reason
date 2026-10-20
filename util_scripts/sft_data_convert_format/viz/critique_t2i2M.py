'''
Random sample 5 datapoints and visualize by saving to output.

'''

import glob, io, os, random, textwrap

import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from PIL import Image

NUM_SAMPLES = 15
SEED = 339

# Set to one of "llava_next_finetune", "progamergov_dalle3", "gpt4o_generated"
# or None to sample across all sources.
SOURCE = "llava_next_finetune"


def load_random_rows(input_dir: str, n: int, seed: int):
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    random.seed(seed)
    random.shuffle(files)

    rows = []
    for f in files:
        # Read one RecordBatch at a time: each batch is a single contiguous
        # chunk, which avoids PyArrow's "Nested data conversions not
        # implemented for chunked array outputs" error on struct/list columns.
        i = 0
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=32):
            if random.random() > 0.3:
                continue
            batch_rows = batch.to_pylist()
            if SOURCE is not None:
                batch_rows = [r for r in batch_rows if r.get("source") == SOURCE]
            rows.extend(batch_rows)
            if i > 1:
                break
            i += 1
            # if len(rows) >= n * 10:
            #     break
        if len(rows) >= n * 10000:
            break

    return random.sample(rows, min(n, len(rows)))


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
        print(f"Saved {out_path}")

    all_prompts = prompts + short_prompts
    prompts_file = os.path.join(output_dir, "prompts.txt")
    with open(prompts_file, "w") as f:
        for i, prompt in enumerate(all_prompts):
            f.write(f"=== Prompt {i} ===\n")
            f.write(conv_to_text(prompt))
            f.write("\n\n")
    print(f"Saved prompts to {prompts_file}")


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

    input_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/T2I_datasets/text-to-image-2M_tar_format/"
    output = "~/text-to-image-2M_tar_format_llava_next_finetune"


    rows = load_random_rows(input_dir, NUM_SAMPLES, SEED)
    print(f"Sampled {len(rows)} rows")
    visualize(rows, output)
    print(f"Done. Outputs in {output}")
