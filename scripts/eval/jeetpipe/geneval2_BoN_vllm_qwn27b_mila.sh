#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=geneval2_bon_qwn27b
# cn-g005/cn-g011: CVMFS CUDA headers are broken there, Triton JIT fails
#SBATCH --exclude=cn-g005,cn-g011
#SBATCH --partition short-unkillable
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/geneval2_bon_qwn27b_%j.out

# geneval2_BoN.sh with the BoN picks graded by a Qwen3.5-27B judge instead of
# GenEval2's own Qwen3-VL-8B (same soft-TIFA questions/prompt as evaluation_optz.py,
# via eval/geneval2_rescore.py --in_place, as iter_geneval2_adhoc_e2e_MDP_vllm_qwn27b.sh).
# Best-of-N: sample N images per prompt, pick one with a value model (Qwen3-VL-8B
# vqascore, unchanged), score the pick with the 27B judge. One generation pass gives
# every BoN-n for n in N_LIST (best of the first n samples).
# Prompt pre-processing matches iter_geneval2_adhoc_e2e_MDP_vllm_qwn27b.sh's draft
# image: PROMPT_PREFIX=iterative (default) puts ITERATIVE_PROMPT_PREFIX ("Generate an
# image iteratively by self-reflecting and correcting.\n") before the prompt in the
# user turn (eval_geneval.py --prompt_prefix). Those runs write to a _pfx-iterative
# GEN_DIR; PROMPT_PREFIX=none is the old bare-prompt BoN, sharing GEN_DIR with
# geneval2_BoN.sh. Generation is skipped when a seed's images already exist; the
# selection is reused from ${GEN_DIR}/select_${SELECT_METHOD} when present.
# Results go to ${GEN_DIR}/select_${SELECT_METHOD}_judge-${JUDGE_TAG}.
# Generation runs the LM on vLLM (eval_geneval.py --lm_backend vllm, one process per
# GPU via run_vllm_ranks, as the *_MDP_vllm.sh launchers). Same prompt ids, sampling
# params, HF generation_config eos + repetition_penalty (1.05) and fp32 de-tokenizer
# as the torchrun/HF path, but per-row vLLM seeds, so images are not bit-identical.
# Tunables: GEN_BATCH (32 prompts = 128 rows per vLLM call), DECODE_BATCH (32),
# VLLM_MEM (0.35 of the GPU for vLLM; the rest is the fp32 de-tokenizer),
# LM_BACKEND (vllm; hf = the old torchrun path), JUDGE, JUDGE_TAG, JUDGE_BATCH (32).

source /home/mila/s/singhsd/CODE/LatentDCR/tar_reason/scripts/eval/jeetpipe/mila_env.sh

set -e


if [ ! -f /tmp/ta_tok.pth ]; then
    cp /network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/ta_tok.pth /tmp/
fi


N_GPUS=4

N=4
N_LIST=(1 2 3 4)

# vqascore: P(Yes | 'Does this image show "<prompt>"?').
# soft_tifa: the benchmark's own questions -> oracle upper bound, not a fair baseline.
SELECT_METHOD=vqascore
VALUE_MODEL=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/Qwen3-VL-8B-Instruct

# GenEval2 judge (grading only).
JUDGE=${JUDGE:-/network/scratch/s/singhsd/hf_home/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654}   # Mila: same snapshot as his
JUDGE_TAG=${JUDGE_TAG:-Qwen3.5-27B}
JUDGE_BATCH=${JUDGE_BATCH:-32}

EVAL_SET_NAME=GenEval2
EVAL_SET=/network/scratch/s/sarvjeet-singh.ghotra/git/GenEval2/geneval2_data.jsonl

# MODEL_NAME=Tar-7B
# MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/Tar-7B
MODEL_NAME=${MODEL_NAME:-slf_ref_edit_t21_17K_greedy_slf_ref_draft_S0} #t1.0_k1200_p0.95
MODEL_PATH=${MODEL_PATH:-/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t21/checkpoint-17000}
# RL_FT=""
RL_FT=${RL_FT-""}
LORA_PATH=${LORA_PATH-""}

# iterative: same user-turn prefix as the iterative scripts (--prompt_format sft);
# none: bare prompt (geneval2_BoN.sh).
PROMPT_PREFIX=${PROMPT_PREFIX:-iterative}
PFX_TAG=""
if [ "$PROMPT_PREFIX" != none ]; then
    PFX_TAG=_pfx-${PROMPT_PREFIX}
fi

AR_RES=512
AR_MODEL=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/ar_dtok_lp_${AR_RES}px.pth

LM_BACKEND=${LM_BACKEND:-vllm}
# Prompts per LM call (BATCH_SIZE * N rows). vLLM has no padding and a paged KV cache,
# so it takes far more rows than HF generate (64 rows there).
if [ "$LM_BACKEND" = vllm ]; then
    BATCH_SIZE=${GEN_BATCH:-32}
else
    BATCH_SIZE=$(( 64 / N > 0 ? 64 / N : 1 ))
fi
# Decoding is capped at 32 images, as in geneval2_e2e.sh: 64 OOMs the 512px AR
# decoder's CFG KV cache next to the 7B model.
DECODE_BATCH_SIZE=${DECODE_BATCH:-${DECODE_BATCH_SIZE:-32}}
# vLLM share of the GPU: 7B bf16 weights ~15 GB + KV for 128 rows (~6 GB) fit in 0.35;
# the fp32 de-tokenizer at decode batch 32 needs the rest.
VLLM_MEM=${VLLM_MEM:-0.35}

SEEDS=(${SEEDS:-2213 713 13})
OUTPUT_DIR=/network/scratch/s/singhsd/tar_reason/results_jeetpipe/${EVAL_SET_NAME}_Qwen3.5-27B


LORA_ARG=""
if [ -n "$LORA_PATH" ]; then
    LORA_ARG="--lora_path $LORA_PATH"
fi

CURR_DIR=$(pwd)
for SEED in "${SEEDS[@]}"; do

cd "$CURR_DIR"

GEN_MODEL=bon_${MODEL_NAME}_${AR_RES}px_N${N}${RL_FT}${PFX_TAG}/seed_${SEED}
GEN_DIR=${OUTPUT_DIR}/${GEN_MODEL}
# geneval2_BoN.sh's selection dir (8B-judged); its picks are reused when complete.
SELECT_DIR_8B=${GEN_DIR}/select_${SELECT_METHOD}
RESULT_DIR=${GEN_DIR}/select_${SELECT_METHOD}_judge-${JUDGE_TAG}

N_PROMPTS=$(wc -l < $EVAL_SET)
N_IMAGES=$(find ${GEN_DIR} -path '*/samples/*.png' 2>/dev/null | wc -l)

GENERATED=0
if [ "$N_IMAGES" -lt $(( N_PROMPTS * N )) ]; then
GENERATED=1

# Same sampling as geneval2_e2e.sh; no --geneval2 since it only allows --repeat 1.
GEN_ARGS=(
    --model $MODEL_PATH
    --save_dir $GEN_DIR
    --prompts $EVAL_SET
    --repeat $N
    --seed $SEED
    --temperature 1.0
    --top_k 1200
    --top_p 0.95
    --batch_size $BATCH_SIZE
    --decode_batch_size $DECODE_BATCH_SIZE
    --seq_len 729
    --cfg_scale 4.0
    --ar_path $AR_MODEL
    --decoder_path /network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/vq_ds16_t2i.pt
    --encoder_path /tmp/ta_tok.pth
    --prompt_prefix $PROMPT_PREFIX
    $LORA_ARG
)

if [ "$LM_BACKEND" = vllm ]; then
    source ${CURR_DIR}/scripts/eval/vllm_gen_env.sh
    vllm_gen_activate
    # LoRA adapters are served by vLLM directly (no merge).
    run_vllm_ranks ${GEN_DIR}/.sync_${SLURM_JOB_ID:-$$} ${GEN_DIR}/gen_logs eval/eval_geneval.py \
        "${GEN_ARGS[@]}" \
        --vllm_gpu_memory_utilization $VLLM_MEM
else
    mamba activate tar
    module load cuda/12.1.1
    export PYTHONPATH=$PYTHONPATH:${CURR_DIR}
    torchrun --standalone --nproc_per_node=$N_GPUS eval/eval_geneval.py \
        "${GEN_ARGS[@]}" --attn sdpa
fi

fi

# # ========================================= Select ===================================

mkdir -p $RESULT_DIR
LOG_FILE=${RESULT_DIR}/log.txt
: > $LOG_FILE

# Reuse geneval2_BoN.sh's picks when every BoN-n file is there and the images were
# not just (re)generated; otherwise select here with the same value model.
REUSE=1
[ "$GENERATED" -eq 0 ] || REUSE=0
for n in "${N_LIST[@]}"; do
    [ -f ${SELECT_DIR_8B}/geneval2_results_bon${n}.json ] || REUSE=0
done

if [ "$REUSE" -eq 1 ]; then
    echo "Reusing picks from ${SELECT_DIR_8B}" | tee -a $LOG_FILE
    for n in "${N_LIST[@]}"; do
        cp ${SELECT_DIR_8B}/geneval2_results_bon${n}.json ${RESULT_DIR}/
    done
    cp ${SELECT_DIR_8B}/bon_scores_${SELECT_METHOD}.json ${RESULT_DIR}/ 2>/dev/null || true
    grep "^BoN-" ${SELECT_DIR_8B}/select_log.txt >> $LOG_FILE 2>/dev/null || true
else
    mamba activate geneval2
    module load cuda/12.6.0

    # Selection only (no --eval_method): grading is done by the 27B judge below.
    # torchrun defaults OMP_NUM_THREADS to 1, which starves the image preprocessing threads.
    OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=$N_GPUS eval/bon_select_geneval2.py \
        --benchmark_data $EVAL_SET \
        --gen_dir $GEN_DIR \
        --n_candidates $N \
        --n_list "${N_LIST[@]}" \
        --value_model $VALUE_MODEL \
        --geneval2_dir /network/scratch/s/sarvjeet-singh.ghotra/git/GenEval2 \
        --method $SELECT_METHOD \
        --out_dir $RESULT_DIR > ${RESULT_DIR}/select_log.txt 2>&1 \
        || { echo "Selection failed:"; cat ${RESULT_DIR}/select_log.txt; exit 1; }

    grep "^BoN-" ${RESULT_DIR}/select_log.txt >> $LOG_FILE
fi

# # ========================================= Grade with the 27B judge ===================================

mamba activate t2v_metric
module load cuda/12.6.0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# Qwen3.5's linear-attention layers JIT-compile Triton kernels, and Triton's gcc
# step reads Python.h from $SCRATCH, which intermittently fails. Compile against a
# node-local copy of this env's headers through a CC wrapper.
JUDGE_PY_INC=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
JUDGE_LOCAL_INC=/tmp/py_include_judge_${SLURM_JOB_ID:-$$}
if [ ! -f ${JUDGE_LOCAL_INC}/Python.h ]; then
    for TRY in 1 2 3; do
        rm -rf ${JUDGE_LOCAL_INC}
        cp -r ${JUDGE_PY_INC} ${JUDGE_LOCAL_INC} \
            && [ "$(find ${JUDGE_LOCAL_INC} -type f | wc -l)" -eq "$(find ${JUDGE_PY_INC} -type f | wc -l)" ] \
            && break
        sleep 10
    done
    [ -f ${JUDGE_LOCAL_INC}/Python.h ] || { echo "ERROR: could not copy ${JUDGE_PY_INC} to ${JUDGE_LOCAL_INC}"; exit 1; }
fi
export CC=/tmp/triton_cc_judge_${SLURM_JOB_ID:-$$}.sh
cat > ${CC} <<EOS
#!/bin/bash
ARGS=()
for A in "\$@"; do
    [ "\$A" = "-I${JUDGE_PY_INC}" ] && A="-I${JUDGE_LOCAL_INC}"
    ARGS+=("\$A")
done
exec /usr/bin/gcc "\${ARGS[@]}"
EOS
chmod +x ${CC}
export TRITON_CACHE_DIR=/tmp/triton_cache_judge_${SLURM_JOB_ID:-$$}

echo " "
echo "Grading BoN-{${N_LIST[*]}} picks with ${JUDGE_TAG} on ${N_GPUS} GPUs:"

# One judge process per GPU; geneval2_rescore.py finds every geneval2_results_bon<n>.json
# in RESULT_DIR and splits them round-robin over the ranks (4 n, 4 GPUs: one each),
# writing results_geneval2_bon<n>.jsonl + eval_log_bon<n>.txt next to them.
JUDGE_LOG_DIR=${RESULT_DIR}/judge_logs
mkdir -p ${JUDGE_LOG_DIR}
# run_judges <extra flag> <log suffix>: sets FAILED.
run_judges() {
    FAILED=""
    declare -A PIDS=()
    for (( R=0; R<N_GPUS; R++ )); do
        CUDA_VISIBLE_DEVICES=${R} python ${CURR_DIR}/eval/geneval2_rescore.py \
            --src_dirs ${RESULT_DIR} \
            --in_place $1 \
            --benchmark_data $EVAL_SET \
            --model_path $JUDGE \
            --tag ${JUDGE_TAG} \
            --method soft_tifa \
            --batch_size ${JUDGE_BATCH} \
            --no_bucket \
            --rank ${R} --world_size ${N_GPUS} > ${JUDGE_LOG_DIR}/rank${R}$2.log 2>&1 &
        PIDS[$R]=$!
    done
    for R in "${!PIDS[@]}"; do
        STATUS=0
        wait ${PIDS[$R]} || STATUS=$?
        if [ $STATUS -ne 0 ]; then
            FAILED="$FAILED rank-${R}:${STATUS}"
        fi
    done
}
# --overwrite: the picks were just (re)written, so never keep scores left from an
# earlier run. One retry without it re-scores only what is still missing.
run_judges --overwrite ""
if [ -n "$FAILED" ]; then
    echo "Judge failed:$FAILED, retrying the missing BoN-n once"
    run_judges "" _retry
fi

for n in "${N_LIST[@]}"; do
    {
        echo "BoN-${n} soft TIFA (${JUDGE_TAG}):"
        # A crashed judge rank writes no eval log: keep going so the failure
        # report below still runs (set -e would exit on grep).
        grep "Score:" ${RESULT_DIR}/eval_log_bon${n}.txt 2>/dev/null || echo "(no eval_log_bon${n}.txt: judge failed)"
    } >> $LOG_FILE
done

if [ -n "$FAILED" ]; then
    echo "Evaluation failed:$FAILED (judge logs in ${JUDGE_LOG_DIR})"
    tail -n 20 ${JUDGE_LOG_DIR}/rank*.log
    exit 1
fi

echo "================ DONE ============="
cat $LOG_FILE

done
