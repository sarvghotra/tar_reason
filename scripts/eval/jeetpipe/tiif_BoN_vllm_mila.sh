#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=tiif_bon_vllm
# cn-g005/cn-g011: CVMFS CUDA headers are broken there, Triton/flashinfer JIT fails
#SBATCH --exclude=cn-g005,cn-g011
#SBATCH --partition short-unkillable
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/tiif_bon_vllm_%j.out

# vLLM-backend variant of tiif_BoN.sh: the Tar LM is served by an in-process
# vLLM engine per GPU (eval/lm_backends.py, one process per GPU via
# run_vllm_ranks in scripts/eval/vllm_gen_env.sh) instead of HF generate under
# torchrun, as in iter_tiif_bench_adhoc_e2e_MDP_vllm.sh. Selection, judging and
# summaries are unchanged. Tunables: GEN_BATCH (default 100), DECODE_BATCH (32),
# VLLM_MEM (0.45 of the GPU for vLLM; the rest is the de-tokenizer). Per-row
# seeds differ from the HF path, so samples are not bit-identical to tiif_BoN.sh.
#
# Best-of-N text-to-image on TIIF-Bench:
#   1. sample N draft images per (prompt, register) with the same sampling as the
#      iter-1 draft of iter_tiif_bench_adhoc_e2e.sh;
#   2. pick one with a value model (eval/bon_select_tiif.py), which links the pick
#      of every BoN-n into a '<MODEL_NAME>_bon<n>' model dir;
#   3. judge each BoN-n with a local Qwen2.5-VL and summarise with TIIF-Bench's
#      own scripts, so '<MODEL_NAME>_bon1' .. '_bon<N>' are rows of one table.
#      With the 72B judge and --tiif_protocol this is TIIF-Bench's own
#      Qwen2.5-VL-72B recipe (see iter_tiif_bench_adhoc_e2e_MDP.sh).
# All seeds are generated and selected first, then judged together by one judge
# load (the 72B takes minutes to start), then summarised.
# One generation pass gives every BoN-n for n in N_LIST (best of the first n samples).
# BoN-1 is sample 0, i.e. the plain one-shot baseline.
# Generation resumes from the images already on disk, so a timed-out job can be resubmitted.

source ~/.bashrc
source /home/mila/s/singhsd/CODE/LatentDCR/tar_reason/scripts/eval/jeetpipe/mila_env.sh

set -euo pipefail

# Copy via a temp name + mv, with retries: a $SCRATCH read that fails midway
# must not leave a truncated /tmp/ta_tok.pth that later jobs on this node trust.
if [ ! -f /tmp/ta_tok.pth ]; then
    TA_TOK_SRC=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/ta_tok.pth
    for TRY in 1 2 3; do
        cp ${TA_TOK_SRC} /tmp/ta_tok.pth.$$ \
            && [ "$(stat -c %s /tmp/ta_tok.pth.$$)" -eq "$(stat -c %s ${TA_TOK_SRC})" ] \
            && mv /tmp/ta_tok.pth.$$ /tmp/ta_tok.pth && break
        rm -f /tmp/ta_tok.pth.$$
        sleep 10
    done
    [ -f /tmp/ta_tok.pth ] || { echo "ERROR: could not copy ${TA_TOK_SRC} to /tmp"; exit 1; }
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
    N_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi
echo "Using ${N_GPUS} GPUs"

N=4
N_LIST=(1 2 3 4)

# vqascore: P(Yes | 'Does this image show "<prompt>"?'), prompt in the judged register.
SELECT_METHOD=vqascore
VALUE_MODEL=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/Qwen3-VL-8B-Instruct

TIIF_DIR=/network/scratch/s/sarvjeet-singh.ghotra/git/TIIF-Bench
PROMPTS_DIR=${TIIF_DIR}/data/testmini_prompts
EVAL_PROMPTS_DIR=${TIIF_DIR}/data/testmini_eval_prompts
EVAL_SET_NAME=tiif-bench-testmini_eval

# MODEL_NAME=Tar-7B
# MODEL_PATH=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/Tar-7B
# /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t21/checkpoint-17000
MODEL_NAME=${MODEL_NAME:-""}
MODEL_PATH=${MODEL_PATH:-""}

# Optional LoRA adapter on MODEL_PATH; RL_FT tags the output dir.
RL_FT=${RL_FT-""}
LORA_PATH=${LORA_PATH-""}

LORA_ARG=""
if [ -n "$LORA_PATH" ]; then
    LORA_ARG="--lora_path $LORA_PATH"
fi

AR_RES=512
AR_MODEL=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/ar_dtok_lp_${AR_RES}px.pth

# Every job is one image here (--repeat expands into jobs). Decoding is capped at 32
# images: peak GPU memory is ~72 GB at VLLM_MEM 0.45 + DECODE_BATCH 32, so do not
# raise one without lowering the other.
GEN_BATCH=${GEN_BATCH:-100}
DECODE_BATCH=${DECODE_BATCH:-32}
VLLM_MEM=${VLLM_MEM:-0.45}

# Qwen2.5-VL judge (local HF snapshot; nothing is downloaded at run time).
# JUDGE_GPUS is the number of GPUs one judge instance shards the model over:
# the 72B (~146 GB of bf16 weights) needs all four A100-80GB, the 7B fits on
# one, so four 7B judges run side by side.
# JUDGE_BACKEND=vllm serves the judge with `vllm serve` (tensor parallel over
# JUDGE_GPUS, one server per GPU block) and queries it the way TIIF-Bench's
# eval_with_vlm.py does; JUDGE_BACKEND=hf loads it with transformers
# (device_map over JUDGE_GPUS), which is several times slower.
# JUDGE_PATH=$(ls -d /network/scratch/s/sarvjeet-singh.ghotra/hf_home/models/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*/ | head -1)
# JUDGE_GPUS=1
JUDGE_PATH=${JUDGE_PATH:-/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/Qwen2.5-VL-72B-Instruct}
JUDGE_GPUS=${JUDGE_GPUS:-4}
JUDGE_BACKEND=${JUDGE_BACKEND:-vllm}
# hf: prompts per generate() call. vllm: tasks submitted per round (results are
# written after each round), with JUDGE_CONCURRENCY requests in flight per seed.
JUDGE_BATCH=${JUDGE_BATCH:-$( [ "${JUDGE_BACKEND}" = vllm ] && echo 512 || echo 16 )}
JUDGE_CONCURRENCY=${JUDGE_CONCURRENCY:-96}
# TIIF-Bench's protocol (eval_with_vlm.py against `vllm serve`): sampled at
# T=1.0 with vLLM's defaults, random template per attempt, text before image.
# It retries a mis-counted answer forever; allow more re-rolls than the default.
JUDGE_ARGS="--tiif_protocol --temperature 1.0 --max_retries 10"
VLLM_PYTHON=/network/scratch/s/sarvjeet-singh.ghotra/installs/miniforge3/envs/vllm/bin/python

if [ ! -f ${JUDGE_PATH}/model.safetensors.index.json ]; then
    echo "ERROR: ${JUDGE_PATH} has no model.safetensors.index.json (download incomplete?)"
    exit 1
fi
if (( N_GPUS % JUDGE_GPUS != 0 )); then
    echo "ERROR: N_GPUS=${N_GPUS} is not a multiple of JUDGE_GPUS=${JUDGE_GPUS}"
    exit 1
fi

SEEDS=(${SEEDS:-17 91 13})
OUTPUT_DIR=/network/scratch/s/singhsd/tar_reason/results_jeetpipe/${EVAL_SET_NAME}

# Each prompt line is generated (and judged) in both the short and the long register.
N_RECORDS=$(cat ${PROMPTS_DIR}/*_prompts.jsonl | grep -c .)
N_RECORDS=$(( 2 * N_RECORDS ))

CURR_DIR=$(pwd)
EXPECTED=$(grep -c . ${EVAL_PROMPTS_DIR}/*.jsonl | awk -F: '{n+=$2} END {print 2*n}')
ALL_MODELS=()
for n in "${N_LIST[@]}"; do
    ALL_MODELS+=(${MODEL_NAME}_bon${n})
done

set_seed_dirs() {
    GEN_MODEL=bon_${MODEL_NAME}_${AR_RES}px_N${N}${RL_FT}/seed_$1
    SAVE_DIR=${OUTPUT_DIR}/${GEN_MODEL}
    IMAGE_DIR=${SAVE_DIR}/images
    RESULT_DIR=${SAVE_DIR}/eval_results
    SELECT_DIR=${SAVE_DIR}/select_${SELECT_METHOD}
}

############################ 1. Generate N samples ############################
# All seeds first, in the vllm env, so the env is switched once.
# --n_iters 1 is the draft only, with the draft sampling of iter_tiif_bench_adhoc_e2e.sh.
# --skip_existing resumes a timed-out run instead of wiping its images.
declare -A GENERATED=()
GEN_ENV_READY=0
for SEED in "${SEEDS[@]}"; do
set_seed_dirs ${SEED}
N_IMAGES=$( (find ${IMAGE_DIR} -path "*/${MODEL_NAME}_iter1/*" -name '*.png' 2>/dev/null || true) | wc -l)
GENERATED[$SEED]=0
if [ "$N_IMAGES" -ge $(( N_RECORDS * N )) ]; then
    echo "seed ${SEED}: all $(( N_RECORDS * N )) samples exist; skipping generation"
    continue
fi
GENERATED[$SEED]=1

if [ "$GEN_ENV_READY" -eq 0 ]; then
    source ${CURR_DIR}/scripts/eval/vllm_gen_env.sh
    vllm_gen_activate
    GEN_ENV_READY=1
fi

# vllm_gen_activate already compiles Triton against a node-local copy of the
# Python headers. Other transient $SCRATCH read failures (an ImportError on a .so
# in the env, a checkpoint read) still kill a rank, and run_vllm_ranks then stops
# all of them. Re-run: --skip_existing resumes from the images already saved.
GEN_ATTEMPTS=3
for TRY in $(seq 1 ${GEN_ATTEMPTS}); do
if [ ${TRY} -gt 1 ]; then
    echo "seed ${SEED}: generation attempt ${TRY}/${GEN_ATTEMPTS}"
    sleep 30
fi
run_vllm_ranks ${SAVE_DIR}/.sync_${SLURM_JOB_ID:-$$} ${SAVE_DIR}/gen_logs/try${TRY} \
    ${CURR_DIR}/eval/iterative_tiif_bench_adhoc.py \
    --n_iters 1 \
    --model ${MODEL_PATH} \
    --gen_model ${GEN_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --prompts_dir ${PROMPTS_DIR} \
    --eval_model_name ${MODEL_NAME} \
    --repeat $N \
    --seed $SEED \
    --temperature 1.0 \
    --top_k 1200 \
    --top_p 0.95 \
    --batch_size $GEN_BATCH \
    --decode_batch_size $DECODE_BATCH \
    --vllm_gpu_memory_utilization $VLLM_MEM \
    --gen_seq_len 729 \
    --cfg_scale 4.0 \
    --ar_path $AR_MODEL \
    --encoder_path /tmp/ta_tok.pth \
    --decoder_path /network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/tar/vq_ds16_t2i.pt \
    --draft_img_scale 0 \
    --draft_gen_seq_len 729 \
    --skip_existing \
    $LORA_ARG \
    && break
if [ ${TRY} -eq ${GEN_ATTEMPTS} ]; then
    echo "ERROR: seed ${SEED}: generation failed ${GEN_ATTEMPTS} times"
    exit 1
fi
done

N_IMAGES=$( (find ${IMAGE_DIR} -path "*/${MODEL_NAME}_iter1/*" -name '*.png' 2>/dev/null || true) | wc -l)
if [ "$N_IMAGES" -lt $(( N_RECORDS * N )) ]; then
    echo "ERROR: seed ${SEED}: only ${N_IMAGES}/$(( N_RECORDS * N )) samples on disk after generation"
    exit 1
fi
done
# The generation env's Triton compiler wrapper is not meant for the other envs.
unset CC TRITON_CACHE_DIR

for SEED in "${SEEDS[@]}"; do

cd "$CURR_DIR"
set_seed_dirs ${SEED}

############################ 2. Select ############################
mamba activate geneval2
module load cuda/12.6.0
unset PYTHONPATH

mkdir -p ${SELECT_DIR}

# torchrun defaults OMP_NUM_THREADS to 1, which starves the image preprocessing threads.
OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=$N_GPUS ${CURR_DIR}/eval/bon_select_tiif.py \
    --prompts_dir ${PROMPTS_DIR} \
    --image_dir ${IMAGE_DIR} \
    --eval_model_name ${MODEL_NAME} \
    --n_candidates $N \
    --n_list "${N_LIST[@]}" \
    --value_model $VALUE_MODEL \
    --geneval2_dir /network/scratch/s/sarvjeet-singh.ghotra/git/GenEval2 \
    --method $SELECT_METHOD \
    --out_dir ${SELECT_DIR} > ${SELECT_DIR}/select_log.txt 2>&1 \
    || { echo "Selection failed:"; cat ${SELECT_DIR}/select_log.txt; exit 1; }

grep "^BoN-" ${SELECT_DIR}/select_log.txt

mkdir -p ${RESULT_DIR}

# The judge skips any prompt whose result json exists. New images mean new picks,
# so drop the old verdicts; otherwise keep them and resume.
if [ "${GENERATED[$SEED]}" -eq 1 ]; then
    for n in "${N_LIST[@]}"; do
        rm -rf "${RESULT_DIR:?}"/"${MODEL_NAME:?}"_bon"${n:?}"
    done
fi

# Results from two judges would silently mix in one directory for the same
# reason. Pin the directory to one judge.
JUDGE_STAMP=${RESULT_DIR}/judge_model.txt
if [ -s ${JUDGE_STAMP} ] && [ "$(cat ${JUDGE_STAMP})" != "${JUDGE_PATH}" ]; then
    echo "ERROR: ${RESULT_DIR} was judged by $(cat ${JUDGE_STAMP}), not ${JUDGE_PATH}"
    exit 1
fi
echo "${JUDGE_PATH}" > ${JUDGE_STAMP}

done

############################ 3. Qwen2.5-VL judge ############################
mamba activate vllm
module load cuda/12.6.0

# TIIF-Bench's summary script writes an .xlsx.
$VLLM_PYTHON -c "import openpyxl" 2>/dev/null || $VLLM_PYTHON -m pip install --quiet openpyxl

# Retries a transient $SCRATCH read failure; see iter_tiif_bench_adhoc_e2e.sh.
JUDGE_ATTEMPTS=3
run_judge() {
    local GPUS=$1 JUDGE_LOG=$2 SEED=$3
    shift 3
    local EVAL_MODELS=("$@")
    local TRY
    set_seed_dirs ${SEED}
    for TRY in $(seq 1 ${JUDGE_ATTEMPTS}); do
        if [ ${TRY} -gt 1 ]; then
            echo "=== retry ${TRY}/${JUDGE_ATTEMPTS} ===" >> ${JUDGE_LOG}
        fi
        CUDA_VISIBLE_DEVICES=${GPUS} $VLLM_PYTHON ${CURR_DIR}/eval/tiif_bench_vlm_judge.py \
            --jsonl_dir ${EVAL_PROMPTS_DIR} \
            --image_dir ${IMAGE_DIR} \
            --eval_model "${EVAL_MODELS[@]}" \
            --output_dir ${RESULT_DIR} \
            --model ${JUDGE_PATH} \
            --batch_size ${JUDGE_BATCH} \
            --seed ${SEED} \
            ${JUDGE_ARGS} \
            >> ${JUDGE_LOG} 2>&1 && return 0
        sleep 30
    done
    return 1
}

N_JUDGES=$(( N_GPUS / JUDGE_GPUS ))
STATUS=0
PIDS=()
JUDGE_LOGS=()

if [ "${JUDGE_BACKEND}" = vllm ]; then
    # N_JUDGES servers, each tensor-parallel over its own block of JUDGE_GPUS
    # GPUs, loaded once for every seed. --no-mm-device-do-normalize sidesteps
    # this build's FusedInputNorm crash on Qwen2.5-VL (F.batch_norm eps=0.0):
    # the processor normalises on the CPU instead, as transformers does.
    # Everything else is `vllm serve --dtype bfloat16`, as in TIIF-Bench's
    # README; sampling defaults come from the checkpoint's generation_config.
    # flashinfer JIT-builds its sampler at warm-up and needs the env's ninja.
    export PATH=$(dirname ${VLLM_PYTHON}):${PATH}
    export TRITON_CACHE_DIR=/tmp/triton_cache_tiif_judge_${SLURM_JOB_ID:-$$}
    # Per job: concurrent jobs share OUTPUT_DIR and would overwrite each other's logs.
    SERVER_LOG_DIR=${OUTPUT_DIR}/judge_server_logs/${SLURM_JOB_ID:-$$}
    mkdir -p ${SERVER_LOG_DIR}
    BASE_PORT=${BASE_PORT:-$(( 20000 + ${SLURM_JOB_ID:-$$} % 20000 ))}
    SERVER_PIDS=()
    URLS=()
    cleanup() {
        for PID in "${SERVER_PIDS[@]}"; do
            kill -- "-${PID}" 2>/dev/null || true  # the whole group: engine and TP workers too
        done
        sleep 10
        for PID in "${SERVER_PIDS[@]}"; do  # a server still loading ignores SIGTERM
            kill -9 -- "-${PID}" 2>/dev/null || true
        done
    }
    trap cleanup EXIT

    # Every TP worker gcc-compiles Triton launchers at startup against the
    # env's Python headers on $SCRATCH, and a transient failed read of one
    # header kills that worker and hangs the server. Compile against a
    # node-local copy instead: Triton takes its compiler from $CC.
    PY_INC=$($VLLM_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
    LOCAL_PY_INC=/tmp/py_include_${SLURM_JOB_ID:-$$}
    for TRY in 1 2 3; do
        rm -rf ${LOCAL_PY_INC}
        cp -r ${PY_INC} ${LOCAL_PY_INC} \
            && [ "$(find ${LOCAL_PY_INC} -type f | wc -l)" -eq "$(find ${PY_INC} -type f | wc -l)" ] \
            && break
        sleep 10
    done
    [ -f ${LOCAL_PY_INC}/Python.h ] || { echo "ERROR: could not copy ${PY_INC} to ${LOCAL_PY_INC}"; exit 1; }
    export CC=/tmp/triton_cc_${SLURM_JOB_ID:-$$}.sh
    cat > ${CC} <<EOF
#!/bin/bash
ARGS=()
for A in "\$@"; do
    [ "\$A" = "-I${PY_INC}" ] && A="-I${LOCAL_PY_INC}"
    ARGS+=("\$A")
done
exec /usr/bin/gcc "\${ARGS[@]}"
EOF
    chmod +x ${CC}

    # The custom all-reduce kernel fails CUDA graph capture on these nodes;
    # NCCL is used instead (as in enrich_captions_vllm.sh).
    SERVER_EXTRA_ARGS=$( (( JUDGE_GPUS > 1 )) && echo --disable-custom-all-reduce || true)
    start_server() {
        local S=$1 TRY=$2
        local GPUS PORT
        GPUS=$(seq -s, $(( S * JUDGE_GPUS )) $(( (S + 1) * JUDGE_GPUS - 1 )))
        PORT=$(( BASE_PORT + S ))
        CUDA_VISIBLE_DEVICES=${GPUS} setsid $VLLM_PYTHON -m vllm.entrypoints.cli.main serve ${JUDGE_PATH} \
            --port ${PORT} \
            --dtype bfloat16 \
            --tensor-parallel-size ${JUDGE_GPUS} \
            --max-model-len 8192 \
            --limit-mm-per-prompt '{"image": 1, "video": 0}' \
            --no-mm-device-do-normalize \
            --seed 0 \
            ${SERVER_EXTRA_ARGS} > ${SERVER_LOG_DIR}/server${S}_try${TRY}.log 2>&1 &
        SERVER_PIDS[$S]=$!
        URLS[$S]=http://localhost:${PORT}
        echo "judge server ${S} (try ${TRY}): GPUs ${GPUS}, log ${SERVER_LOG_DIR}/server${S}_try${TRY}.log"
    }
    for (( S=0; S<N_JUDGES; S++ )); do
        start_server ${S} 1
    done
    SERVER_RETRIES=3
    STARTED=${SECONDS}
    for (( S=0; S<N_JUDGES; S++ )); do
        TRY=1
        until curl -sf ${URLS[$S]}/health >/dev/null 2>&1; do
            # A TP worker that dies at startup (e.g. Triton's gcc failing on a
            # transient $SCRATCH read of Python.h) leaves the server hung, not
            # exited; restart on either.
            if ! kill -0 ${SERVER_PIDS[$S]} 2>/dev/null \
                    || grep -q "WorkerProc hit an exception" ${SERVER_LOG_DIR}/server${S}_try${TRY}.log; then
                echo "judge server ${S} failed during startup (try ${TRY}/${SERVER_RETRIES}):"
                grep -v shm_broadcast ${SERVER_LOG_DIR}/server${S}_try${TRY}.log | tail -20
                kill -9 -- "-${SERVER_PIDS[$S]}" 2>/dev/null || true
                if (( TRY >= SERVER_RETRIES )); then
                    exit 1
                fi
                sleep 30
                TRY=$(( TRY + 1 ))
                start_server ${S} ${TRY}
            fi
            if (( SECONDS - STARTED > 2400 )); then
                echo "ERROR: judge server ${S} not healthy after $(( SECONDS - STARTED )) s"
                tail -30 ${SERVER_LOG_DIR}/server${S}_try${TRY}.log
                exit 1
            fi
            sleep 10
        done
    done
    echo "judge server(s) healthy after $(( SECONDS - STARTED )) s"

    JUDGE_ARGS="${JUDGE_ARGS} --concurrency ${JUDGE_CONCURRENCY} --base_url $(printf '%s/v1 ' "${URLS[@]}")"
    # One client per seed, all at once; the servers batch across them.
    for SEED in "${SEEDS[@]}"; do
        set_seed_dirs ${SEED}
        JUDGE_LOG=${SAVE_DIR}/judge_log.txt
        : > ${JUDGE_LOG}
        echo "judging seed ${SEED}: ${ALL_MODELS[*]} (log ${JUDGE_LOG})"
        run_judge "" ${JUDGE_LOG} ${SEED} "${ALL_MODELS[@]}" &
        PIDS+=($!)
        JUDGE_LOGS+=(${JUDGE_LOG})
    done
    for PID in "${PIDS[@]}"; do
        wait ${PID} || STATUS=$?
    done
    cleanup
    trap - EXIT
else
    # N_JUDGES transformers processes, each on its own block of JUDGE_GPUS
    # GPUs, per seed; BoN-n are dealt round-robin, and each process loads the
    # judge once for all of its BoN-n.
    for SEED in "${SEEDS[@]}"; do
        set_seed_dirs ${SEED}
        PIDS=()
        for (( J=0; J<N_JUDGES && J<${#ALL_MODELS[@]}; J++ )); do
            MODELS=()
            for (( I=J; I<${#ALL_MODELS[@]}; I+=N_JUDGES )); do
                MODELS+=(${ALL_MODELS[$I]})
            done
            GPUS=$(seq -s, $(( J * JUDGE_GPUS )) $(( (J + 1) * JUDGE_GPUS - 1 )))
            JUDGE_LOG=${SAVE_DIR}/judge_log_${J}.txt
            : > ${JUDGE_LOG}
            echo "seed ${SEED}, judge ${J} on GPUs ${GPUS}: ${MODELS[*]} (log ${JUDGE_LOG})"
            run_judge ${GPUS} ${JUDGE_LOG} ${SEED} "${MODELS[@]}" &
            PIDS+=($!)
            JUDGE_LOGS+=(${JUDGE_LOG})
        done
        for PID in "${PIDS[@]}"; do
            wait ${PID} || STATUS=$?
        done
    done
fi

if [ $STATUS -ne 0 ]; then
    echo "ERROR: at least one judging run failed (exit $STATUS)"
    for JUDGE_LOG in "${JUDGE_LOGS[@]}"; do
        echo "--- ${JUDGE_LOG} tail ---"
        tail -30 ${JUDGE_LOG}
    done
    exit $STATUS
fi

for SEED in "${SEEDS[@]}"; do
set_seed_dirs ${SEED}

# A missing per-prompt json silently drops that prompt from the accuracy denominator.
for n in "${N_LIST[@]}"; do
    JUDGED=$(find ${RESULT_DIR}/${MODEL_NAME}_bon${n} -name '*.json' 2>/dev/null | wc -l || true)
    echo "seed ${SEED} BoN-${n}: judged ${JUDGED}/${EXPECTED} (prompt, register) pairs"
done

############################ 4. Summarise ############################
RESULT_FILE=${SAVE_DIR}/tiif_bench_log.txt
grep "^BoN-" ${SELECT_DIR}/select_log.txt > ${RESULT_FILE}

(cd ${TIIF_DIR} && $VLLM_PYTHON eval/summary_results.py --input_dir ${RESULT_DIR}) \
    | tee -a ${RESULT_FILE}

# summary_dimension_results.py exits 120 even on success; see iter_tiif_bench_adhoc_e2e.sh.
DIM_TXT=${RESULT_DIR}/result_summary_dimension.txt
rm -f ${DIM_TXT}
(cd ${TIIF_DIR} && $VLLM_PYTHON eval/summary_dimension_results.py \
    --input_excel ${RESULT_DIR}/result_summary.xlsx \
    --output_txt ${DIM_TXT}) 2>&1 \
    | tee -a ${RESULT_FILE} || true
if [ ! -s ${DIM_TXT} ]; then
    echo "ERROR: summary_dimension_results.py produced no report"
    exit 1
fi

echo "================ DONE ============="
echo "OUTPUT DIR: ${SAVE_DIR}"
echo "  per-dimension accuracy : ${RESULT_DIR}/result_summary.xlsx"
echo "  grouped TIIF scores    : ${RESULT_DIR}/result_summary_dimension.txt"
echo "  rows '${MODEL_NAME}_bon1' .. '${MODEL_NAME}_bon${N}' are the BoN-n picks"

done
