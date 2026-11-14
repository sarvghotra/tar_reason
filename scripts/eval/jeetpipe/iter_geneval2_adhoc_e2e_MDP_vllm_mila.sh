#!/bin/bash
# vLLM-backend variant of iter_geneval2_adhoc_e2e_MDP.sh: the Tar LM is served by an in-process vLLM
# engine per GPU (eval/lm_backends.py) instead of HF generate under torchrun.
# Scoring/judging phases are unchanged. Tunables: GEN_BATCH (default 100),
# DECODE_BATCH (32), VLLM_MEM (0.45 of the GPU for vLLM; the rest is the de-tokenizer).
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=iter_geneval2_adhoc_MDP_vllm
# cn-g005/cn-g011: CVMFS CUDA headers are broken there, Triton/flashinfer JIT fails
#SBATCH --exclude=cn-g005,cn-g011
#SBATCH --partition short-unkillable
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/iter_geneval2_adhoc_e2e_MDP_vllm_%j.out

source /home/mila/s/singhsd/CODE/LatentDCR/tar_reason/scripts/eval/jeetpipe/mila_env.sh

set -e


if [ ! -f /tmp/ta_tok.pth ]; then
    cp /network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/ta_tok.pth /tmp/
fi


N_GPUS=${N_GPUS:-4}

REPEAT=1

# Images per sample (draft + N_ITERS-1 corrections). Each correction sees only
# the prompt and the previous image; N_ITERS=2 equals iter_geneval2_adhoc_e2e.sh.
N_ITERS=${N_ITERS:-4}

# EVAL_SET_NAME=geneval
# EVAL_SET=/network/scratch/s/sarvjeet-singh.ghotra/git/geneval/prompts/evaluation_metadata.jsonl
# EVAL_SET_NAME=geneval_hard_cat_100
# EVAL_SET=~/tmp/geneval_hard_cat_100.jsonl

# EVAL_SET_NAME=FLUX-Reason-1.2M_short
# EVAL_SET=/network/scratch/s/sarvjeet-singh.ghotra/data/latentDCR_val/leakage_free/FLUX-Reason-1.2M_tar_format_short-conv_short.jsonl

# EVAL_SET_NAME=FLUX-Reason-1.2M_long_600
# EVAL_SET=/network/scratch/s/sarvjeet-singh.ghotra/data/latentDCR_val/leakage_free/FLUX-Reason-1.2M_tar_format_short-conv_long_600.jsonl

# EVAL_SET_NAME=gpt-edit-simpler_hqedt
# EVAL_SET=/network/scratch/s/sarvjeet-singh.ghotra/data/latentDCR_val/leakage_free/gpt-edit-simpler_hqedt_slfref_tar.jsonl

# EVAL_SET_NAME=gpt-edit-simpler_hqedt2
# EVAL_SET=/network/scratch/s/sarvjeet-singh.ghotra/data/latentDCR_val/leakage_free/gpt-edit-simpler_hqedt2_slfref_tar.jsonl

# EVAL_SET_NAME=gpt-edit-simpler_omni_600
# EVAL_SET=/network/scratch/s/sarvjeet-singh.ghotra/data/latentDCR_val/leakage_free/gpt-edit-simpler_omni_qwn3moe30B_short-capt_tar_600.jsonl

# EVAL_SET_NAME=i1-imagenet22k-512-resolution-1m
# EVAL_SET=/network/scratch/s/sarvjeet-singh.ghotra/data/latentDCR_val/leakage_free/i1-imagenet22k-512-resolution-1m-tar_format.jsonl


EVAL_SET_NAME=${EVAL_SET_NAME:-GenEval2}
EVAL_SET=${EVAL_SET:-/network/scratch/s/sarvjeet-singh.ghotra/git/GenEval2/geneval2_data.jsonl}

# EVAL_SET_NAME=geneval2_specific_subset_200
# EVAL_SET=/home/mila/s/sarvjeet-singh.ghotra/tmp/geneval2_specific_subset_200.jsonl

# EVAL_SET_NAME=geneval2_data_rand300
# EVAL_SET=/network/scratch/s/singhsd/tar_reason/results_jeetpipe/eval_sets/geneval2_data_rand300.jsonl

# MODEL_NAME=Tar-7B_greedy_slf_ref_draft_S0_seed${SEED} #t1.0_k1200_p0.95
# MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/Tar-7B

# MODEL_NAME=7B #t1.0_k1200_p0.95
# MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/Tar-7B

# MODEL_NAME=debug_load_save_7B
# MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/wts_load_debug/weights_only/checkpoint-2
# RL_FT=""
# LORA_PATH=${5:-}
# MODEL_NAME/MODEL_PATH/RL_FT/LORA_PATH can be overridden from the environment
# (sbatch --export=ALL,MODEL_NAME=...); an empty LORA_PATH means no adapter.
RL_FT=${RL_FT-""}
LORA_PATH=${LORA_PATH-""}

# MODEL_NAME=slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0 #t1.0_k1200_p0.95
# MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t21/checkpoint-17000
MODEL_NAME=${MODEL_NAME:-slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0} #t1.0_k1200_p0.95
MODEL_PATH=${MODEL_PATH:-/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t21/checkpoint-17000}

# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t20/checkpoint-16000
# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/tamia/slf_ref_edit_t6_correction/checkpoint-16000
# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t6_repro_w_corr/checkpoint-15000
# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t21/checkpoint-17000

# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/tamia/slf_ref_edit_t6_cnt/checkpoint-8000
# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/tamia/slf_ref_edit_t6_cnt_correction/checkpoint-5000
# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/tamia/slf_ref_edit_t6_cnt/checkpoint-8000


# MODEL_NAME=slf_ref_edit_nolora_t5_10K
# MODEL_PATH=~/scratch/git/tar_reason/output_dir/tamia/slf_ref_edit_nolora_t5/checkpoint-10000

AR_RES=512
AR_MODEL=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/ar_dtok_lp_${AR_RES}px.pth

SEEDS=(${SEEDS:-2213 713 13})
OUTPUT_DIR=/network/scratch/s/singhsd/tar_reason/results_jeetpipe/${EVAL_SET_NAME}


echo "MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} RL_FT=${RL_FT} LORA_PATH=${LORA_PATH}"

LORA_ARG=""
if [ -n "$LORA_PATH" ]; then
    LORA_ARG="--lora_path $LORA_PATH"
fi

CURR_DIR=$(pwd)
for SEED in "${SEEDS[@]}"; do

cd "$CURR_DIR"

GEN_MODEL=iter_adhoc_MDP_n${N_ITERS}_${MODEL_NAME}_${AR_RES}px_repeat${REPEAT}${RL_FT}/seed_${SEED}
RESULT_DIR=${OUTPUT_DIR}/${GEN_MODEL}

source ${CURR_DIR}/scripts/eval/vllm_gen_env.sh
vllm_gen_activate

run_vllm_ranks ${RESULT_DIR}/.sync_${SLURM_JOB_ID:-$$} ${RESULT_DIR}/gen_logs eval/iterative_generation_adhoc_MDP.py \
    --n_iters $N_ITERS \
    --model $MODEL_PATH \
    --out_dir $RESULT_DIR \
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
    --batch_size ${GEN_BATCH:-100} \
    --decode_batch_size ${DECODE_BATCH:-32} \
    --vllm_gpu_memory_utilization ${VLLM_MEM:-0.45} \
    --reflect_tokens ${REFLECT_TOKENS:-128} \
    --gen_seq_len 729 \
    --cfg_scale 4.0 \
    --ar_path $AR_MODEL \
    --decoder_path /network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/vq_ds16_t2i.pt \
    --encoder_path /tmp/ta_tok.pth \
    --draft_img_scale 0 \
    --draft_gen_seq_len 729 \
    --verbose \
    $LORA_ARG \
    --prompt_format ${PROMPT_FORMAT:-sft}

# --verbose \

# # ========================================= Compute scores ===================================exit

LOG_FILE=${RESULT_DIR}/log.txt
touch $LOG_FILE

cd /network/scratch/s/sarvjeet-singh.ghotra/git/GenEval2

mamba activate geneval2
module load cuda/12.6.0

echo " "
echo "Running ITER-1..ITER-${N_ITERS} evaluations in parallel on ${N_GPUS} GPUs:"

# Score every iteration, in batches of N_GPUS (one GPU each).
FAILED=""
for (( START=1; START<=N_ITERS; START+=N_GPUS )); do
    declare -A PIDS=()
    for (( ITER=START; ITER<START+N_GPUS && ITER<=N_ITERS; ITER++ )); do
        CUDA_VISIBLE_DEVICES=$(( ITER - START )) python evaluation_optz.py \
            --benchmark_data $EVAL_SET \
            --image_filepath_data ${RESULT_DIR}/${ITER}/geneval2_results.json \
            --method soft_tifa \
            --output_file ${RESULT_DIR}/${ITER}/results_geneval2.jsonl > ${RESULT_DIR}/${ITER}/eval_log.txt 2>&1 &
        PIDS[$ITER]=$!
    done
    for ITER in "${!PIDS[@]}"; do
        STATUS=0
        wait ${PIDS[$ITER]} || STATUS=$?
        if [ $STATUS -ne 0 ]; then
            FAILED="$FAILED iter-${ITER}:${STATUS}"
        fi
    done
    unset PIDS
done

for ITER in $(seq 1 $N_ITERS); do
    {
        echo " "
        echo "ITER-${ITER}:"
        cat ${RESULT_DIR}/${ITER}/eval_log.txt
    } >> $LOG_FILE
done

if [ -n "$FAILED" ]; then
    echo "Evaluation failed:$FAILED"
    exit 1
fi


mamba activate tar

python /network/scratch/s/singhsd/jeet_snapshot/tar_reason_20261113/eval/analyze/viz_geneval2_iter_results.py \
    --results-dir ${RESULT_DIR} \
    --delta-threshold 0.1 >> $LOG_FILE \
    --no_images \


# FLOPs per iteration (cumulative) next to soft-TIFA AM/GM -> ${RESULT_DIR}/flops.json
echo " " >> $LOG_FILE
python /network/scratch/s/singhsd/jeet_snapshot/tar_reason_20261113/eval/compute_flops.py summarize \
    --output_dir ${RESULT_DIR} \
    --n_iters ${N_ITERS} >> $LOG_FILE

echo "================ DONE ============="
cat $LOG_FILE

done
