#!/bin/bash
# Source this file from a launcher before setting paths or running Python.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/clusters/profile.sh"
# uv mode: TAR_ROOT is this cluster's synced folder ($SCRATCH/tar_reason) with
# shared/ (inputs, pushed by cluv), results/ (outputs, fetched back) and cache/.
# The uv environment needs no cvmfs modules; `uv run` already activates it.
if [[ -n "${TAR_ROOT:-}" ]]; then
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        source "${TAR_ENV:-$REPO_ROOT/.venv}/bin/activate"
    fi
else
    if (( ${#TAR_MODULES[@]} )); then
        module load "${TAR_MODULES[@]}"
    fi
    if [[ "${TAR_NVIDIA_DRIVER_SHIM:-0}" == 1 && -n "${SLURM_JOB_ID:-}" ]]; then
        # Some Fir GPU nodes lack /usr/lib64/nvidia, so the cvmfs loader cannot find
        # libcuda. Expose only the NVIDIA driver libraries (never the host glibc).
        shim="${SLURM_TMPDIR:-/tmp/$USER-$SLURM_JOB_ID}/nvidia_driver_libs"
        mkdir -p "$shim"
        for f in /usr/lib64/libcuda.so* /usr/lib64/libnvidia-*.so* /usr/lib64/libcudadebugger.so*; do
            if [[ -e "$f" ]]; then ln -sf "$f" "$shim/"; fi
        done
        export LD_LIBRARY_PATH="$shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    source "${TAR_ENV:-$REPO_ROOT/tar}/bin/activate"
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
if [[ -n "${TAR_ROOT:-}" ]]; then
    TAR_SHARED="${TAR_SHARED:-$TAR_ROOT/shared}"
    DATA_ROOT="$TAR_SHARED/data"
    MODELS_ROOT="$TAR_SHARED/models"
    SHARED_ROOT="$TAR_SHARED"
    RESULTS_ROOT="${TAR_RESULTS:-$TAR_ROOT/results}"
    CACHE_ROOT="$TAR_ROOT/cache"
    # Same filesystem as the venvs under $TAR_ROOT/env/venvs, so uv hard-links.
    UV_CACHE_DEFAULT="$TAR_ROOT/env/uv_cache"
    SFT_MODEL="${SFT_MODEL:-$MODELS_ROOT/sft/slf_ref_edit_t21/checkpoint-17000}"
    REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$MODELS_ROOT/judges/Qwen3-VL-8B-Instruct}"
    # One environment: the judge worker runs under the same interpreter.
    REWARD_PYTHON="${REWARD_PYTHON:-$(command -v python)}"
    export TAR_ROOT TAR_SHARED HF_HUB_CACHE="${HF_HUB_CACHE:-$TAR_SHARED/hf_hub}"
else
    DATA_ROOT="$REPO_ROOT/data"
    MODELS_ROOT="$REPO_ROOT/models"
    SHARED_ROOT="${SHARED_ROOT:-$REPO_ROOT}"
    RESULTS_ROOT="$REPO_ROOT/results"
    CACHE_ROOT="$RESULTS_ROOT/cache"
    UV_CACHE_DEFAULT="$CACHE_ROOT/uv"
    SFT_MODEL="${SFT_MODEL:-$REPO_ROOT/sft_model}"
    REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$REPO_ROOT/reward_model}"
    REWARD_PYTHON="${REWARD_PYTHON:-$REPO_ROOT/qwen_reward/bin/python}"
fi
BASE_MODEL="${BASE_MODEL:-$MODELS_ROOT/Tar-7B}"
VISION_MODEL="${VISION_MODEL:-$MODELS_ROOT/tar/ta_tok.pth}"
AR_MODEL="${AR_MODEL:-$MODELS_ROOT/tar/ar_dtok_lp_512px.pth}"
DECODER="${DECODER:-$MODELS_ROOT/tar/vq_ds16_t2i.pt}"
export DATA_ROOT MODELS_ROOT SFT_MODEL VISION_MODEL AR_MODEL DECODER REWARD_MODEL_PATH REWARD_PYTHON
export TAR_VISION_MODEL="$VISION_MODEL"
export WANDB_PROJECT="${WANDB_PROJECT:-tar_reasoning}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export WANDB_CONFIG_DIR="$RESULTS_ROOT/config/wandb"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$UV_CACHE_DEFAULT}"
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
