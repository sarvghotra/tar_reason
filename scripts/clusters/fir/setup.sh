#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
SHARED_ROOT="${SHARED_ROOT:-/project/rrg-bengioy-ad/jeet}"
python3 "$REPO_ROOT/scripts/clusters/setup_links.py" --cluster fir \
    --data-root "${DATA_TARGET:-$SHARED_ROOT/data}" \
    --models-root "${MODELS_TARGET:-$SHARED_ROOT/models}" \
    --sft-model "${SFT_TARGET:-$SHARED_ROOT/models/sft_ckpt/slf_ref_edit_t6_repro_w_corr_t2/checkpoint-24000}" \
    --results-root "${RESULTS_TARGET:-/scratch/$USER/tar_reason_results}" \
    --reward-model "${REWARD_TARGET:-${RESULTS_TARGET:-/scratch/$USER/tar_reason_results}/pretrained/Qwen3-VL-8B-Instruct}" "$@"
