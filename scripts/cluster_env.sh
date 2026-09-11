#!/bin/bash
# Source this file from a launcher before setting paths or running Python.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/clusters/profile.sh"
if (( ${#TAR_MODULES[@]} )); then
    module load "${TAR_MODULES[@]}"
fi
source "${TAR_ENV:-$REPO_ROOT/tar}/bin/activate"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
DATA_ROOT="$REPO_ROOT/data"
MODELS_ROOT="$REPO_ROOT/models"
SHARED_ROOT="${SHARED_ROOT:-$REPO_ROOT}"
RESULTS_ROOT="$REPO_ROOT/results"
BASE_MODEL="${BASE_MODEL:-$MODELS_ROOT/Tar-7B}"
SFT_MODEL="${SFT_MODEL:-$REPO_ROOT/sft_model}"
VISION_MODEL="${VISION_MODEL:-$MODELS_ROOT/tar/ta_tok.pth}"
AR_MODEL="${AR_MODEL:-$MODELS_ROOT/tar/ar_dtok_lp_512px.pth}"
DECODER="${DECODER:-$MODELS_ROOT/tar/vq_ds16_t2i.pt}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$REPO_ROOT/reward_model}"
REWARD_PYTHON="${REWARD_PYTHON:-$REPO_ROOT/qwen_reward/bin/python}"
export DATA_ROOT MODELS_ROOT SFT_MODEL VISION_MODEL AR_MODEL DECODER REWARD_MODEL_PATH REWARD_PYTHON
export TAR_VISION_MODEL="$VISION_MODEL"
export WANDB_PROJECT="${WANDB_PROJECT:-tar_reasoning}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_CACHE_DIR="$RESULTS_ROOT/cache/wandb"
export WANDB_CONFIG_DIR="$RESULTS_ROOT/config/wandb"
export HF_HOME="$RESULTS_ROOT/cache/huggingface"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRITON_CACHE_DIR="$RESULTS_ROOT/cache/triton"
export TORCH_EXTENSIONS_DIR="$RESULTS_ROOT/cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$RESULTS_ROOT/cache/torchinductor"
export UV_CACHE_DIR="$RESULTS_ROOT/cache/uv"
export TMPDIR="${SLURM_TMPDIR:-$RESULTS_ROOT/tmp}"

tar_require_paths() {
    python "$REPO_ROOT/scripts/check_inputs.py" "$@"
}

tar_init_run() {
    RUN_NAME="${RUN_NAME:-$1}"
    if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
        echo 'RUN_NAME must contain only letters, digits, dots, underscores and hyphens.' >&2
        return 1
    fi
    LOCAL_DIR="$RESULTS_ROOT/models/$RUN_NAME"
    EVAL_DIR="$RESULTS_ROOT/evaluations/$RUN_NAME"
    LOG_DIR="$RESULTS_ROOT/logs/$RUN_NAME"
    export WANDB_NAME="$RUN_NAME"
    export WANDB_DIR="$RESULTS_ROOT/wandb/$RUN_NAME"
    export WANDB_DATA_DIR="$WANDB_DIR/artifacts"
    if [[ "${TAR_DRY_RUN:-0}" != 1 ]]; then
        mkdir -p "$LOCAL_DIR" "$EVAL_DIR" "$LOG_DIR" "$WANDB_DIR" \
            "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
            "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
            "$TORCHINDUCTOR_CACHE_DIR" "$UV_CACHE_DIR" "$TMPDIR"
        exec > >(tee -a "$LOG_DIR/${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}.log") 2>&1
    fi
    echo "Python: $(command -v python)"
    echo "Run: $RUN_NAME; checkpoints: $LOCAL_DIR; evaluations: $EVAL_DIR"
}

tar_launch() {
    if [[ "${TAR_DRY_RUN:-0}" == 1 ]]; then
        printf 'Command: '; printf '%q ' "$@"; printf '\n'
        return
    fi
    "$@"
}
