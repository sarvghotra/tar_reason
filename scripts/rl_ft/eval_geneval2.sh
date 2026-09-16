#!/bin/bash
set -euo pipefail
source "${TAR_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/cluster_env.sh"
: "${RL_CHECKPOINT:?Set the exact RL adapter checkpoint}"
: "${GENEVAL2_BENCHMARK:?Set the official benchmark JSONL}"
: "${EVAL_CODE_ROOT:?Set the verified evaluation code snapshot}"
tar_init_run geneval2_full
tar_require_paths "$RL_CHECKPOINT/adapter_model.safetensors" "$RL_CHECKPOINT/state.json" \
    "$GENEVAL2_BENCHMARK" "$AR_MODEL" "$VISION_MODEL" "$DECODER" \
    "$REWARD_MODEL_PATH/config.json" "$REWARD_PYTHON" "$EVAL_CODE_ROOT/manifest.json"
export PYTHONPATH="$EVAL_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
tar_launch "${EVAL_PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="${N_GPUS:-4}" \
    "$EVAL_CODE_ROOT/llava/train/rl/eval_geneval2.py" \
    --checkpoint "$RL_CHECKPOINT" --benchmark "$GENEVAL2_BENCHMARK" \
    --output_dir "$EVAL_DIR" --code_manifest "$EVAL_CODE_ROOT/manifest.json" \
    --ar_path "$AR_MODEL" --encoder_path "$VISION_MODEL" --decoder_path "$DECODER" \
    --reward_model "$REWARD_MODEL_PATH" --reward_python "$REWARD_PYTHON" \
    --seed "${SEED:-421}" --gen_batch_size "${GEN_BATCH_SIZE:-16}" \
    --max_refinements "${MAX_REFINEMENTS:-3}" --reflect_tokens "${REFLECT_TOKENS:-128}" \
    --max_seq_len "${MAX_SEQ_LEN:-4096}" --cfg_scale 4 --report_to wandb \
    --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" "$@"
