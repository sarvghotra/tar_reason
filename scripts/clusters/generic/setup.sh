#!/bin/bash
set -euo pipefail
PROFILE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PROFILE_DIR/../../.." && pwd)"
python3 "$REPO_ROOT/scripts/clusters/setup_links.py" --cluster "$(basename "$PROFILE_DIR")" \
    --data-root "${DATA_TARGET:?Set DATA_TARGET}" \
    --models-root "${MODELS_TARGET:?Set MODELS_TARGET}" \
    --sft-model "${SFT_TARGET:?Set SFT_TARGET}" \
    --results-root "${RESULTS_TARGET:?Set RESULTS_TARGET}" \
    --reward-model "${REWARD_TARGET:?Set REWARD_TARGET}" "$@"
