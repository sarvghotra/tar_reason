#!/bin/bash
source "${TAR_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/cluster_env.sh"
tar_init_run "sft_selfreflect"
N_GPUS=${N_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
PREV_STAGE_CHECKPOINT="${PREV_STAGE_CHECKPOINT:-$BASE_MODEL}"
DATA_PATH="${DATA_PATH:-scripts/img_gen/sft.yaml}"
MAX_STEPS=${MAX_STEPS:-240}
LR=${LR:-3e-5}
TRAIN_PARTS="${TRAIN_PARTS:-mm_language_model}"
ACCU_STEPS=${ACCU_STEPS:-32}
global_bs=${BATCH_SIZE:-256}
if (( global_bs < N_GPUS * ACCU_STEPS || global_bs % (N_GPUS * ACCU_STEPS) != 0 )); then
    echo "BATCH_SIZE must be a positive multiple of N_GPUS * ACCU_STEPS" >&2
    exit 1
fi
BS_PER_GPU=$((global_bs / N_GPUS / ACCU_STEPS))
tar_require_paths "$PREV_STAGE_CHECKPOINT/config.json" "$VISION_MODEL" "$DATA_PATH"

tar_launch torchrun \
    --nproc_per_node=$N_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT:-29505}" \
    llava/train/train.py \
    --deepspeed scripts/zero1.json \
    --num_image_tokens 65536 \
    --num_scale_tokens 3 \
    --load_embeddings_from_vision True \
    --model_name_or_path "$PREV_STAGE_CHECKPOINT" \
    --version "qwen_1_5" \
    --data_path "${DATA_PATH}" \
    --dataset_cls 'weighted_parquet' \
    --dispatch_batches False \
    --max_steps "${MAX_STEPS}" \
    --mm_tunable_parts "${TRAIN_PARTS}" \
    --vision_tower "${VISION_MODEL}" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --group_by_modality_length True \
    --image_aspect_ratio square \
    --mm_patch_merge_type flat \
    --bf16 True \
    --output_dir "${LOCAL_DIR}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size "$BS_PER_GPU" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "$ACCU_STEPS" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate "${LR}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "$RUN_NAME" \
    --dataloader_drop_last True \
    --attn_implementation "sdpa" \
    "$@"
