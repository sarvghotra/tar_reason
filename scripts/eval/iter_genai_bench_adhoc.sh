#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=iter_genai_bench_adhoc
#SBATCH --partition short-unkillable
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/iter_genai_bench_adhoc/%j_%t.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/iter_genai_bench_adhoc/%j_%t.err

source ~/.bashrc
eval "$(mamba shell hook --shell bash)"

set -e

if [ ! -f /tmp/ta_tok.pth ]; then
    cp /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ta_tok.pth /tmp/
fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
    N_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi
echo "Using ${N_GPUS} GPUs"

REPEAT=1
NUM_PROMPTS=800           # 1600 (GenAI-Bench paper) or 527 (VQAScore paper)
EVAL_SET_DIR=GenAI-Image-${NUM_PROMPTS}   # dir under ROOT_DIR with genai_image.json

EVAL_SET_NAME=genai-bench-${NUM_PROMPTS}

MODEL_NAME=slf_ref_edit_t20_16K_greedy_slf_ref_draft_S0
MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t20/checkpoint-16000

# MODEL_NAME=7B
# MODEL_PATH=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/Tar-7B

T2V_METRICS_DIR=/network/scratch/s/sarvjeet-singh.ghotra/git/t2v_metrics
ROOT_DIR=${T2V_METRICS_DIR}/datasets
SCORE_MODEL=qwen3.5-27b    # gemma-3-27b-it  # gemma-3-4b-it

AR_RES=512
AR_MODEL=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ar_dtok_lp_${AR_RES}px.pth

SEEDS=(13 17 91)
OUTPUT_DIR=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/${EVAL_SET_NAME}/

RL="_rl_ft_oracle_t2_ckpt_500"
LORA_PATH=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/rl_ft_oracle_t2/checkpoint-500
# ${5:-}

LORA_ARG=""
if [ -n "$LORA_PATH" ]; then
    LORA_ARG="--lora_path $LORA_PATH"
fi

CURR_DIR=$(pwd)
for SEED in "${SEEDS[@]}"; do

cd "$CURR_DIR"
# GEN_MODEL=iter_adhoc_${MODEL_NAME}_${AR_RES}px_repeat${REPEAT}${RL}/seed_${SEED}
GEN_MODEL=iter_adhoc_${MODEL_NAME}_${AR_RES}px${RL}/seed_${SEED}

############################ 1. Generate images ############################
mamba activate tar
module load cuda/12.1.1

export LD_LIBRARY_PATH="/network/scratch/s/sarvjeet-singh.ghotra/installs/miniforge3/envs/tar/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=$(pwd):$PYTHONPATH

torchrun --standalone --nproc_per_node=$N_GPUS \
    eval/iterative_genai_bench_adhoc.py \
    --model ${MODEL_PATH} \
    --gen_model ${GEN_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --root_dir ${ROOT_DIR} \
    --eval_set_dir ${EVAL_SET_DIR} \
    --repeat $REPEAT \
    --seed $SEED \
    --temperature 1.0 \
    --top_k 1200 \
    --top_p 0.95 \
    --reflect_temperature 1.0 \
    --reflect_top_k 1200 \
    --reflect_top_p 0.95 \
    --batch_size 32 \
    --decode_batch_size 32 \
    --reflect_tokens 128 \
    --gen_seq_len 729 \
    --cfg_scale 4.0 \
    --ar_path $AR_MODEL \
    --encoder_path /tmp/ta_tok.pth \
    --decoder_path /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/vq_ds16_t2i.pt \
    --verbose \
    --draft_img_scale 0 \
    --draft_gen_seq_len 729 \
    $LORA_ARG


############################ 2. VQAScore (per iteration) ############################
mamba activate t2v_metric
module load cuda/12.4.0

# Drop tar_reason from PYTHONPATH so `import llava` resolves inside t2v_metrics,
# not to tar_reason/llava (which needs the `tar` env's transformers).
unset PYTHONPATH

RESULT_FILE=${OUTPUT_DIR}/${GEN_MODEL}/${SCORE_MODEL}_log.txt
cd ${T2V_METRICS_DIR}

echo " "
echo "Running ITER-1 and ITER-2 scoring in parallel on 2 GPUs:"

ITERS=(1 2)
PIDS=()
for I in "${!ITERS[@]}"; do
    ITER=${ITERS[$I]}
    ITER_LOG=${OUTPUT_DIR}/${GEN_MODEL}/${ITER}/eval_log.txt
    mkdir -p $(dirname $ITER_LOG)
    CUDA_VISIBLE_DEVICES=$I python -m genai_bench.evaluate \
        --model ${SCORE_MODEL} \
        --root_dir ${ROOT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --gen_model ${GEN_MODEL}/${ITER} \
        --num_prompts ${NUM_PROMPTS} \
        --result_dir ${OUTPUT_DIR} \
        > ${ITER_LOG} 2>&1 &
    PIDS+=($!)
done

STATUS=0
for I in "${!ITERS[@]}"; do
    wait ${PIDS[$I]} || STATUS=$?
done

for ITER in "${ITERS[@]}"; do
    {
        echo " "
        echo "ITER-${ITER}:"
        cat ${OUTPUT_DIR}/${GEN_MODEL}/${ITER}/eval_log.txt
    } | tee -a $RESULT_FILE
done

if [ $STATUS -ne 0 ]; then
    echo "ERROR: at least one genai_bench.evaluate run failed (exit $STATUS)"
    exit $STATUS
fi

echo "OUTPUT DIR: ${OUTPUT_DIR}/${GEN_MODEL}"

done
