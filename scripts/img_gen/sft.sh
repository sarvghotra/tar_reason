#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=24
#SBATCH --job-name=llava
#SBATCH --partition short-unkillable
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/scratch/git/tar_reason/output_dir/sft_slfreflect/%j_%t.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/scratch/git/tar_reason/output_dir/sft_slfreflect/%j_%t.err

source ~/.bashrc
mamba activate tar
module load cuda/11.8

cp /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ta_tok.pth /tmp/

RUN_NAME="sft_slfreflect"

export PYTHONPATH=$PYTHONPATH:/home/mila/s/sarvjeet-singh.ghotra/scratch/git/tar_reason/
PREV_STAGE_CHECKPOINT=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/Tar-1.5B

VISION_MODEL=/tmp/ta_tok.pth

N_GPUS=4

ACCU_STEPS=32
global_bs=${BATCH_SIZE:-256}
BS_PER_GPU=$((global_bs / N_GPUS / ACCU_STEPS))
echo "Per Device Batch Size: {$BS_PER_GPU}"


DATA_PATH="scripts/img_gen/sft.yaml"
MAX_STEPS=240  # 60000
LR=3e-5
TRAIN_PARTS="mm_language_model"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

export WANDB_NAME=$RUN_NAME
export WANDB_PROJECT="tar_reasoning"
export WANDB_ENTITY="diffusion_aaron"
export WANDB_MODE="offline"

LOCAL_DIR="output_dir/${RUN_NAME}"

torchrun \
    --nproc_per_node=$N_GPUS \
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
    --dataset_cls 'weighted_parquet' \
    --dispatch_batches False \
    --max_steps ${MAX_STEPS} \
    --mm_tunable_parts ${TRAIN_PARTS} \
    --vision_tower ${VISION_MODEL} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --group_by_modality_length True \
    --image_aspect_ratio square \
    --mm_patch_merge_type flat \
    --bf16 True \
    --output_dir ${LOCAL_DIR} \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BS_PER_GPU \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $ACCU_STEPS \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
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

    # --torch_compile True \
    # --torch_compile_backend inductor \

    # /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ta_tok.pth
