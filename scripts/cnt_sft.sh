#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --gpus=h200:8
#SBATCH --mem=0G
#SBATCH -c 32
#SBATCH --ntasks=1
#SBATCH -J cnt_sft_flux6M
#SBATCH -o /home/j/jeet/links/scratch/git/Tar/output_dir/cnt_sft_flux6M/%j_%t.out
#SBATCH -e /home/j/jeet/links/scratch/git/Tar/output_dir/cnt_sft_flux6M/%j_%t.err

source /scratch/j/jeet/installs/miniforge3/bin/activate
conda activate tar
module load cuda/12.2
wandb offline

RUN_NAME="cnt_sft_flux6M"
# RUN_NAME="dbg"
LOCAL_DIR="output_dir/${RUN_NAME}"

$LOCAL_DIR/cp_data.sh

export PYTHONPATH=$PYTHONPATH:/home/j/jeet/links/scratch/git/Tar
PREV_STAGE_CHECKPOINT=/home/j/jeet/links/scratch/models/pre_train/tar/Tar-7B

cp /home/j/jeet/links/scratch/models/pre_train/tar/ta_tok.pth /tmp/
VISION_MODEL=/tmp/ta_tok.pth

DATA_PATH="${LOCAL_DIR}/cnt_sft_flux6M.yaml"
MAX_STEPS=10000
LR=5e-5
TRAIN_PARTS="mm_language_model"

N_GPUS=8

ACCU_STEPS=16
global_bs=${BATCH_SIZE:-256}
BS_PER_GPU=$((global_bs / N_GPUS / ACCU_STEPS))
echo "Per Device Batch Size: {$BS_PER_GPU}"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"



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
    --run_name $RUN_NAME \
    --output_dir ${LOCAL_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BS_PER_GPU \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $ACCU_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --weights_only_save_steps 1000 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.5}' \
    --logging_steps 8 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation "sdpa"
    # --attn_implementation "flash_attention_2"

    # --torch_compile True \
    # --torch_compile_backend inductor \

