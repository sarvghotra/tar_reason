#!/bin/bash
# Install an isolated pixel judge without changing the tar training environment.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/clusters/profile.sh"
if (( ${#TAR_REWARD_MODULES[@]} )); then
    module load "${TAR_REWARD_MODULES[@]}"
fi
QWEN_ENV="$REPO_ROOT/results/environments/qwen_reward"
export UV_CACHE_DIR="$REPO_ROOT/results/cache/uv-qwen"
export TMPDIR="$REPO_ROOT/results/tmp"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
if [[ ! -x "$QWEN_ENV/bin/python" ]]; then
    uv venv --python "$(command -v python)" "$QWEN_ENV"
fi
uv pip install --python "$QWEN_ENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0
uv pip install --python "$QWEN_ENV/bin/python" --index-url https://pypi.org/simple \
    transformers==4.57.6 accelerate==1.13.0 huggingface_hub==0.36.2 pillow==11.2.1
if [[ ! -e "$REPO_ROOT/qwen_reward" && ! -L "$REPO_ROOT/qwen_reward" ]]; then
    ln -s "$QWEN_ENV" "$REPO_ROOT/qwen_reward"
fi
uv pip freeze --python "$QWEN_ENV/bin/python" > "$REPO_ROOT/results/environments/qwen_reward.freeze.txt"
