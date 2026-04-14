#!/bin/bash
# -*- coding: utf-8 -*-
# ---

source ~/.bashrc
mamba activate tar
module load cuda/11.8


hf download csuhan/TA-Tok ta_tok.pth --local-dir /tmp/

# RUN_NAME="tar_1.5B_finetune_viscot"
RUN_NAME="dbg"

export PYTHONPATH=$PYTHONPATH:/home/mila/s/sarvjeet-singh.ghotra/scratch/git/tar_reason/
PREV_STAGE_CHECKPOINT=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/Tar-1.5B

DATA_PATH="/home/mila/s/sarvjeet-singh.ghotra/scratch/data/v_cot_reason/Visual-CoT/vila-u_format_viscot_363k.json"
IMAGE_FOLDER="/home/mila/s/sarvjeet-singh.ghotra/scratch/data/v_cot_reason/Visual-CoT/images_vila-u"

VISION_MODEL=/tmp/ta_tok.pth

# MAX_STEPS=60000
LR=1e-5 # 5e-5
TRAIN_PARTS="mm_language_model"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

export WANDB_NAME=$RUN_NAME
export WANDB_PROJECT=tar_reasoning

LOCAL_DIR="output_dir/${RUN_NAME}"

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29505 \
    llava/train/train.py \
    --deepspeed scripts/zero1.json \
    --num_image_tokens 65536 \
    --num_scale_tokens 3 \
    --load_embeddings_from_vision True \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version "qwen_1_5" \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --dataset_cls 'custom' \
    --mm_tunable_parts ${TRAIN_PARTS} \
    --vision_tower ${VISION_MODEL} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --image_aspect_ratio square \
    --mm_patch_merge_type flat \
    --bf16 True \
    --output_dir ${LOCAL_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate ${LR} \
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
    --run_name $RUN_NAME \
    --dataloader_drop_last True \
    --attn_implementation "sdpa"
    # --attn_implementation "flash_attention_2"

    # --group_by_modality_length True \

    # --torch_compile True \
    # --torch_compile_backend inductor \

    # /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ta_tok.pth
    # --save_steps 10000 \
    # --max_steps ${MAX_STEPS} \