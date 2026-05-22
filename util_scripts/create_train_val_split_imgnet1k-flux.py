import os
import random
import shutil
import glob


def create_split(input_dir, output_dir, val_ratio=0.05, seed=42):
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} parquet files in {input_dir}")

    rng = random.Random(seed)
    shuffled = parquet_files[:]
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_ratio))
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    print(f"Split: {len(train_files)} train files, {len(val_files)} val files")

    train_dir = os.path.join(output_dir, "ImageNet1K-T2I-QwenVL-FLUX-train")
    val_dir = os.path.join(output_dir, "ImageNet1K-T2I-QwenVL-FLUX-val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for i, src in enumerate(train_files):
        dst = os.path.join(train_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        if (i + 1) % 20 == 0 or (i + 1) == len(train_files):
            print(f"  Train: copied {i + 1}/{len(train_files)}")

    for i, src in enumerate(val_files):
        dst = os.path.join(val_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"  Val: copied {i + 1}/{len(val_files)}")

    print(f"\nDone. Train -> {train_dir}")
    print(f"      Val   -> {val_dir}")


if __name__ == "__main__":
    input_data = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/v_cot_reason/tmp/ImageNet1K-T2I-QwenVL-FLUX"
    output_dir = "/home/mila/s/sarvjeet-singh.ghotra/scratch/data/v_cot_reason/tmp/"

    create_split(input_data, output_dir, val_ratio=0.05, seed=42)
