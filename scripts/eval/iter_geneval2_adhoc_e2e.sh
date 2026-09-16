#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=geneval2_iter_adhoc
#SBATCH --partition short-unkillable
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/iter_geneval2_adhoc_7B_greedy_slf_ref_draft_S0_%j_%t.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/iter_geneval2_adhoc_7B_greedy_slf_ref_draft_S0_%j_%t.err

eval "$(mamba shell hook --shell bash)"

set -e


if [ ! -f /tmp/ta_tok.pth ]; then
    cp /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ta_tok.pth /tmp/
fi


N_GPUS=4

REPEAT=1

# EVAL_SET_NAME=geneval
# EVAL_SET=/home/mila/s/sarvjeet-singh.ghotra/scratch/git/geneval/prompts/evaluation_metadata.jsonl
# EVAL_SET_NAME=geneval_hard_cat_100
# EVAL_SET=~/tmp/geneval_hard_cat_100.jsonl

EVAL_SET_NAME=GenEval2
EVAL_SET=/home/mila/s/sarvjeet-singh.ghotra/scratch/git/GenEval2/geneval2_data.jsonl

# EVAL_SET_NAME=geneval2_specific_subset_200
# EVAL_SET=/home/mila/s/sarvjeet-singh.ghotra/tmp/geneval2_specific_subset_200.jsonl

# EVAL_SET_NAME=geneval2_data_rand300
# EVAL_SET=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/eval_sets/geneval2_data_rand300.jsonl

SEED=713

# MODEL_NAME=Tar-7B_greedy_slf_ref_draft_S0_seed${SEED} #t1.0_k1200_p0.95
# MODEL_PATH=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/Tar-7B

# MODEL_NAME=7B #t1.0_k1200_p0.95
# MODEL_PATH=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/Tar-7B

# MODEL_NAME=debug_load_save_7B
# MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/wts_load_debug/weights_only/checkpoint-2
RL_FT="_rl_ft_latent_500"
LORA_PATH=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/rl_ft_latent/checkpoint-500

MODEL_NAME=slf_ref_edit_t20_16K_greedy_slf_ref_draft_S0_seed${SEED} #t1.0_k1200_p0.95
MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t20/checkpoint-16000
# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/tamia/slf_ref_edit_t6_correction/checkpoint-16000
# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t6_repro_w_corr/checkpoint-15000

# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/tamia/slf_ref_edit_t6_cnt/checkpoint-8000
# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/tamia/slf_ref_edit_t6_cnt_correction/checkpoint-5000
# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/tamia/slf_ref_edit_t6_cnt/checkpoint-8000


# MODEL_NAME=slf_ref_edit_nolora_t5_10K
# MODEL_PATH=~/scratch/git/tar_reason/output_dir/tamia/slf_ref_edit_nolora_t5/checkpoint-10000

AR_RES=512
AR_MODEL=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ar_dtok_lp_${AR_RES}px.pth

SEEDS=(13 713 2213)
OUTPUT_DIR=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/${EVAL_SET_NAME}/


LORA_ARG=""
if [ -n "$LORA_PATH" ]; then
    LORA_ARG="--lora_path $LORA_PATH"
fi

CURR_DIR=$(pwd)
for SEED in "${SEEDS[@]}"; do

cd "$CURR_DIR"

GEN_MODEL=iter_adhoc_${MODEL_NAME}_${AR_RES}px_repeat${REPEAT}${RL_FT}/seed_${SEED}

mamba activate tar
module load cuda/12.1.1

export PYTHONPATH=$PYTHONPATH:/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/

torchrun --standalone --nproc_per_node=$N_GPUS eval/iterative_generation_adhoc.py \
    --model $MODEL_PATH \
    --out_dir $OUTPUT_DIR \
    --prompts_file $EVAL_SET \
    --repeat $REPEAT \
    --geneval2 \
    --seed $SEED \
    --temperature 1.0 \
    --top_k 1200 \
    --top_p 0.95 \
    --reflect_temperature 1.0 \
    --reflect_top_k 1200 \
    --reflect_top_p 0.95 \
    --batch_size 64 \
    --decode_batch_size 64 \
    --reflect_tokens 128 \
    --gen_seq_len 729 \
    --cfg_scale 4.0 \
    --ar_path $AR_MODEL \
    --decoder_path /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/vq_ds16_t2i.pt \
    --encoder_path /tmp/ta_tok.pth \
    --draft_img_scale 0 \
    --draft_gen_seq_len 729 \
    --verbose \
    $LORA_ARG


# # ========================================= Compute scores ===================================exit

LOG_FILE=${OUTPUT_DIR}/log.txt
touch $LOG_FILE

cd ~/scratch/git/GenEval2

mamba activate geneval2
module load cuda/12.6.0

echo " "
echo "Running ITER-1 and ITER-2 evaluations in parallel on 2 GPUs:"

CUDA_VISIBLE_DEVICES=0 python evaluation_optz.py \
    --benchmark_data $EVAL_SET \
    --image_filepath_data ${OUTPUT_DIR}/1/geneval2_results.json \
    --method soft_tifa \
    --output_file ${OUTPUT_DIR}/1/results_geneval2.jsonl > ${OUTPUT_DIR}/1/eval_log.txt 2>&1 &
PID1=$!


CUDA_VISIBLE_DEVICES=1 python evaluation_optz.py \
    --benchmark_data $EVAL_SET \
    --image_filepath_data ${OUTPUT_DIR}/2/geneval2_results.json \
    --method soft_tifa \
    --output_file ${OUTPUT_DIR}/2/results_geneval2.jsonl > ${OUTPUT_DIR}/2/eval_log.txt 2>&1 &
PID2=$!

STATUS1=0
STATUS2=0
wait $PID1 || STATUS1=$?
wait $PID2 || STATUS2=$?

{
    echo " "
    echo "ITER-1:"
    cat ${OUTPUT_DIR}/1/eval_log.txt
    echo " "
    echo "ITER-2:"
    cat ${OUTPUT_DIR}/2/eval_log.txt
} >> $LOG_FILE

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
    echo "Evaluation failed (iter-1: $STATUS1, iter-2: $STATUS2)"
    exit 1
fi


mamba activate tar

python /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/eval/analyze/viz_geneval2_iter_results.py \
    --results-dir ${OUTPUT_DIR} \
    --delta-threshold 0.1 >> $LOG_FILE \


echo "================ DONE ============="
cat $LOG_FILE

done