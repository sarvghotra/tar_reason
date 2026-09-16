#!/bin/bash
source "${TAR_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/cluster_env.sh"
tar_init_run "vlm_reward"
REWARD_SCRIPT="${REWARD_SCRIPT:-tts/rl_w_vlm/vlm_reward.py}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$SHARED_ROOT/models/Qwen2.5-VL-7B-Instruct}"
DECOMPOSITION_FILE="${DECOMPOSITION_FILE:-tts/rl_w_vlm/decomposed.jsonl}"
# This scorer and its inputs are absent from this checkout; fail before launch.
tar_require_paths "$REWARD_SCRIPT" "$REWARD_MODEL_PATH/config.json" "$DECOMPOSITION_FILE"
tar_launch python "$REWARD_SCRIPT" \
    --images_dir "$EVAL_DIR/images" \
    --model "$REWARD_MODEL_PATH" \
    --batch_size 1 \
    --decomposition_file "$DECOMPOSITION_FILE" \
    "$@"
