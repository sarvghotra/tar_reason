#!/bin/bash
# vLLM-backend variant of iter_genai_bench_adhoc_MDP.sh: the Tar LM is served by an in-process vLLM
# engine per GPU (eval/lm_backends.py) instead of HF generate under torchrun.
# Scoring/judging phases are unchanged. Tunables: GEN_BATCH (default 100),
# DECODE_BATCH (32), VLLM_MEM (0.45 of the GPU for vLLM; the rest is the de-tokenizer).
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=iter_genai_bench_adhoc_MDP_vllm
# cn-g005/cn-g011: CVMFS CUDA headers are broken there, Triton/flashinfer JIT fails
#SBATCH --exclude=cn-g005,cn-g011
#SBATCH --partition short-unkillable
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/iter_genai_bench_adhoc_MDP_vllm_%j.out

# End-to-end GenAI-Bench for the forced-schedule iterative model, Markov (MDP)
# variant: every correction is conditioned only on the prompt, the previous
# image and its critique (eval/iterative_genai_bench_adhoc_MDP.py). With
# N_ITERS=2 it equals iter_genai_bench_adhoc.sh.

source ~/.bashrc
source /home/mila/s/singhsd/CODE/LatentDCR/tar_reason/scripts/eval/jeetpipe/mila_env.sh

set -e

if [ ! -f /tmp/ta_tok.pth ]; then
    cp /network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/ta_tok.pth /tmp/
fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
    N_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi
echo "Using ${N_GPUS} GPUs"

REPEAT=1
# Images per sample: draft + N_ITERS-1 corrections, each seeing only the
# previous image and its critique.
N_ITERS=4
NUM_PROMPTS=800           # 1600 (GenAI-Bench paper) or 527 (VQAScore paper)
EVAL_SET_DIR=GenAI-Image-${NUM_PROMPTS}   # dir under ROOT_DIR with genai_image.json

EVAL_SET_NAME=genai-bench-${NUM_PROMPTS}

# MODEL_NAME=slf_ref_edit_t20_16K_greedy_slf_ref_draft_S0
# MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t20/checkpoint-16000
# MODEL_NAME/MODEL_PATH/RL_FT/LORA_PATH can be overridden from the environment
# (sbatch --export=ALL,MODEL_NAME=...); an empty LORA_PATH means no adapter.
MODEL_NAME=${MODEL_NAME:-slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0}
MODEL_PATH=${MODEL_PATH:-/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t21/checkpoint-17000}

# MODEL_NAME=7B
# MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/Tar-7B

T2V_METRICS_DIR=/network/scratch/s/sarvjeet-singh.ghotra/git/t2v_metrics
ROOT_DIR=${T2V_METRICS_DIR}/datasets
SCORE_MODEL=${SCORE_MODEL:-clip-flant5-xxl}     # qwen3.5-27b    # gemma-3-27b-it  # gemma-3-4b-it

# t2v_metrics v3.1 dropped CLIP-FlanT5, so it is scored from a second checkout
# pinned to the last release that has it (cdbd5e4, + local patches), in its own
# env with the older transformers it needs. Weights are local:
# ~/scratch/models/pre_train/{clip-flant5-xxl,clip-vit-large-patch14-336}.
if [[ "$SCORE_MODEL" == clip-flant5-* ]]; then
    SCORE_T2V_METRICS_DIR=/network/scratch/s/singhsd/jeet_snapshot/t2v_metrics_clipt5
    SCORE_ENV=t2v_clipt5
else
    SCORE_T2V_METRICS_DIR=${T2V_METRICS_DIR}
    SCORE_ENV=t2v_metric
fi

AR_RES=512
AR_MODEL=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/ar_dtok_lp_${AR_RES}px.pth

SEEDS=(${SEEDS:-13 17 91})
OUTPUT_DIR=/network/scratch/s/singhsd/tar_reason/results_jeetpipe/${EVAL_SET_NAME}/

# RL=""
# LORA_PATH=${5:-}
RL_FT=${RL_FT-""}
LORA_PATH=${LORA_PATH-""}

echo "MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} RL_FT=${RL_FT} LORA_PATH=${LORA_PATH}"

LORA_ARG=""
if [ -n "$LORA_PATH" ]; then
    LORA_ARG="--lora_path $LORA_PATH"
fi

CURR_DIR=$(pwd)
for SEED in "${SEEDS[@]}"; do

cd "$CURR_DIR"
GEN_MODEL=iter_adhoc_MDP_n${N_ITERS}_${MODEL_NAME}_${AR_RES}px_repeat${REPEAT}${RL_FT}/seed_${SEED}
# GEN_MODEL=iter_adhoc_${MODEL_NAME}_${AR_RES}px${RL}/seed_${SEED}

############################ 1. Generate images ############################
# Skip generation when every iteration dir already holds all NUM_PROMPTS*REPEAT
# images (e.g. re-scoring with a new judge); a partial run is resumed.
GEN_DONE=1
for ITER in $(seq 1 $N_ITERS); do
    N_IMG=$(find ${OUTPUT_DIR}/${GEN_MODEL}/${ITER} -maxdepth 1 -name '*.jpeg' 2>/dev/null | wc -l || true)
    if [ "$N_IMG" -lt $(( NUM_PROMPTS * REPEAT )) ]; then GEN_DONE=0; fi
done

if [ "$GEN_DONE" -eq 1 ]; then
    echo "Images already complete for ${GEN_MODEL}; skipping generation."
else
source ${CURR_DIR}/scripts/eval/vllm_gen_env.sh
vllm_gen_activate

run_vllm_ranks ${OUTPUT_DIR}/${GEN_MODEL}/.sync_${SLURM_JOB_ID:-$$} ${OUTPUT_DIR}/${GEN_MODEL}/gen_logs \
    eval/iterative_genai_bench_adhoc_MDP.py \
    --model ${MODEL_PATH} \
    --gen_model ${GEN_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --root_dir ${ROOT_DIR} \
    --eval_set_dir ${EVAL_SET_DIR} \
    --repeat $REPEAT \
    --n_iters $N_ITERS \
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
    --encoder_path /tmp/ta_tok.pth \
    --decoder_path /network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/vq_ds16_t2i.pt \
    --draft_img_scale 0 \
    --draft_gen_seq_len 729 \
    --skip_existing \
    $LORA_ARG \
    --prompt_format ${PROMPT_FORMAT:-sft}

fi

# --verbose \
############################ 2. VQAScore (per iteration) ############################
mamba activate ${SCORE_ENV}
module load cuda/12.4.0

# Drop tar_reason from PYTHONPATH so `import llava` resolves inside t2v_metrics,
# not to tar_reason/llava (which needs the `tar` env's transformers).
unset PYTHONPATH

RESULT_FILE=${OUTPUT_DIR}/${GEN_MODEL}/${SCORE_MODEL}_log.txt
cd ${SCORE_T2V_METRICS_DIR}

ITERS=($(seq 1 $N_ITERS))
echo " "
echo "Running ITER-1..ITER-${N_ITERS} scoring, one iteration per GPU (${N_GPUS} at a time):"

# One scorer per GPU; when every GPU is busy, wait for that wave to finish.
STATUS=0
PIDS=()
for I in "${!ITERS[@]}"; do
    ITER=${ITERS[$I]}
    ITER_LOG=${OUTPUT_DIR}/${GEN_MODEL}/${ITER}/eval_log.txt
    mkdir -p $(dirname $ITER_LOG)
    CUDA_VISIBLE_DEVICES=$((I % N_GPUS)) python -m genai_bench.evaluate \
        --model ${SCORE_MODEL} \
        --root_dir ${ROOT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --gen_model ${GEN_MODEL}/${ITER} \
        --num_prompts ${NUM_PROMPTS} \
        --result_dir ${OUTPUT_DIR} \
        > ${ITER_LOG} 2>&1 &
    PIDS+=($!)
    if [ ${#PIDS[@]} -eq $N_GPUS ]; then
        for PID in "${PIDS[@]}"; do wait $PID || STATUS=$?; done
        PIDS=()
    fi
done
for PID in "${PIDS[@]}"; do wait $PID || STATUS=$?; done

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

############################ 3. Per-prompt analysis ############################
mamba activate tar

python ${CURR_DIR}/eval/analyze/viz_genai_bench_iter_results.py \
    --results-dir ${OUTPUT_DIR}/${GEN_MODEL} \
    --score-model ${SCORE_MODEL} \
    --dataset-dir ${ROOT_DIR}/${EVAL_SET_DIR} \
    --delta-threshold 0.1 \
    --no_images | tee -a $RESULT_FILE

echo "OUTPUT DIR: ${OUTPUT_DIR}/${GEN_MODEL}"

done
