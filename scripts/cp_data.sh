set -euo pipefail

pids=()

# Understanding
src=/home/j/jeet/links/scratch/data/Tar/LLaVA-ReCap-118K/data
dest=/tmp/LLaVA-ReCap-118K/
echo "Copying $src -> $dest"
mkdir -p "$dest"
rsync -avh "$src/" "$dest/" &
pids+=("$!")


src=/home/j/jeet/links/scratch/data/Tar/LLaVA-ReCap-558K/data
dest=/tmp/LLaVA-ReCap-558K/
echo "Copying $src -> $dest"
mkdir -p "$dest"
rsync -avh "$src/" "$dest/" &
pids+=("$!")


src=/home/j/jeet/links/scratch/data/Tar/LLaVA-SFT-665K/
dest=/tmp/LLaVA-SFT-665K
echo "Copying $src -> $dest"
mkdir -p "$dest"
rsync -avh "$src/" "$dest/" &
pids+=("$!")


# Text
src=/home/j/jeet/links/scratch/data/Tar/Magpie-Qwen2.5-Pro-1M-v0.1/data
dest=/tmp/Magpie-Qwen2.5-Pro-1M-v0.1
echo "Copying $src -> $dest"
mkdir -p "$dest"
rsync -avh "$src/" "$dest/" &
pids+=("$!")


# T2I
src=/home/j/jeet/links/scratch/data/T2I_datasets/midjourney-prompts-FLUX_subsets/set1_2M
dest=/tmp/midjourney-prompts-FLUX_subsets
echo "Copying $src -> $dest"
mkdir -p "$dest"
rsync -avh "$src/" "$dest/" &
pids+=("$!")


wait "${pids[@]}"
echo "All copies complete."
