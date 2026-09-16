#!/bin/bash
# Run generation and/or scoring in separate Python stacks on an allocated node.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
: "${CHECKPOINT:?Set CHECKPOINT to the RL adapter}"
: "${BENCHMARK:?Set BENCHMARK to the prepared benchmark JSON}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh evaluation directory}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to the benchmark judge snapshot}"
: "${GEN_PYTHON:?Set GEN_PYTHON to the working Tar evaluation Python}"
: "${GEN_PYTHONPATH:?Set GEN_PYTHONPATH to Tar dependencies}"
: "${SCORE_PYTHONPATH:?Set SCORE_PYTHONPATH to isolated judge dependencies}"
: "${AR_PATH:?Set AR_PATH}"
: "${ENCODER_PATH:?Set ENCODER_PATH}"
: "${DECODER_PATH:?Set DECODER_PATH}"
N_GPUS=${N_GPUS:-4}
SEED=${SEED:-421}
REPORT_TO=${REPORT_TO:-wandb}
STAGE=${STAGE:-all}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export WANDB_MODE=${WANDB_MODE:-online}
export HF_HUB_OFFLINE=1
if [[ "$STAGE" == all || "$STAGE" == generate ]]; then
    PYTHONPATH="$REPO_ROOT:$GEN_PYTHONPATH" "$GEN_PYTHON" -m torch.distributed.run \
        --standalone --nproc_per_node="$N_GPUS" -m llava.train.rl.generate_instruction_benchmarks \
        --checkpoint "$CHECKPOINT" --benchmark "$BENCHMARK" --output_dir "$OUTPUT_DIR" \
        --code_manifest "$REPO_ROOT/manifest.json" --ar_path "$AR_PATH" \
        --encoder_path "$ENCODER_PATH" --decoder_path "$DECODER_PATH" \
        --seed "$SEED" --gen_batch_size "${GEN_BATCH_SIZE:-16}" \
        --max_refinements 3 --reflect_tokens 128 --max_seq_len 4096 \
        --cfg_scale 4 --attn_implementation sdpa --report_to "$REPORT_TO"
fi
if [[ "$STAGE" == all || "$STAGE" == score ]]; then
    PYTHONPATH="$SCORE_PYTHONPATH:$REPO_ROOT" "${SCORE_PYTHON:-$GEN_PYTHON}" -m torch.distributed.run \
        --standalone --nproc_per_node="$N_GPUS" -m llava.train.rl.score_instruction_benchmarks \
        --output_dir "$OUTPUT_DIR" --model "$JUDGE_MODEL" --seed "$SEED" --report_to "$REPORT_TO"
    PYTHONPATH="$SCORE_PYTHONPATH:$REPO_ROOT" "${SCORE_PYTHON:-$GEN_PYTHON}" \
        -m llava.train.rl.score_instruction_benchmarks --output_dir "$OUTPUT_DIR" \
        --model "$JUDGE_MODEL" --summary_only --report_to "$REPORT_TO"
fi
if [[ "$STAGE" != all && "$STAGE" != generate && "$STAGE" != score ]]; then
    echo "Invalid STAGE: $STAGE" >&2
    exit 2
fi
