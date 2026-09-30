#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=critique_sensitivity
#SBATCH --partition long
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/critique_sensitivity_%j_%t.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/critique_sensitivity_%j_%t.err

# Critique-sensitivity probe (eval/analyze/critique_sensitivity.py): one draft per
# prompt, refined under real / shuffled / generic / oracle critiques, plus the
# fresh-draft and re-decode controls, all scored by GenEval2's own judge.
#
# One Qwen3-VL-8B oracle judge is started per GPU, next to that GPU's policy
# rank (~17 G judge + ~15 G policy + de-tokenizer fits an 80 G A100), and rank r
# talks to judge r, so scoring never queues behind a single server.
#
# Cost per prompt with K=4, M=4: 33 sampled images, 37 decodes, 37 judged
# images; ~1 h for 256 prompts on 4 GPUs. The run resumes from records/ if it
# hits the time limit: resubmit with the same settings.
#
# Override from the environment, e.g.
#   MODEL_NAME=rl_t4_500 LORA_PATH=<adapter dir> MAX_PROMPTS=128 sbatch eval/analyze/critique_sensitivity.sh

eval "$(mamba shell hook --shell bash)"
set -e

if [ ! -f /tmp/ta_tok.pth ]; then
    cp /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ta_tok.pth /tmp/
fi

# Judge g must sit on the same physical GPU as torchrun's local rank g, so map
# through CUDA_VISIBLE_DEVICES when it is set (e.g. CUDA_VISIBLE_DEVICES=2,3).
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPU_IDS < <(seq 0 $(($(nvidia-smi --list-gpus | wc -l) - 1)))
fi
N_GPUS=${#GPU_IDS[@]}

# ---- What to probe ----------------------------------------------------------
EVAL_SET_NAME=GenEval2
EVAL_SET=/home/mila/s/sarvjeet-singh.ghotra/scratch/git/GenEval2/geneval2_data.jsonl

MODEL_NAME=${MODEL_NAME:-slf_ref_edit_t20_16K}
MODEL_PATH=${MODEL_PATH:-/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t20/checkpoint-16000}
LORA_PATH=${LORA_PATH:-}            # e.g. an RL adapter checkpoint on top of MODEL_PATH

MAX_PROMPTS=${MAX_PROMPTS:-256}     # random subset (GenEval2 is sorted by atom count)
NUM_REAL=${NUM_REAL:-4}             # K real critiques per draft
IMAGES_PER_CRITIQUE=${IMAGES_PER_CRITIQUE:-4}   # M refine images per critique
CONDITIONS=${CONDITIONS:-real,shuffled,generic,oracle,fresh,redecode}
SEED=${SEED:-713}

AR_RES=512
AR_MODEL=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ar_dtok_lp_${AR_RES}px.pth

OUTPUT_DIR=${OUTPUT_DIR:-/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/critique_sensitivity/${EVAL_SET_NAME}/${MODEL_NAME}_${AR_RES}px_K${NUM_REAL}_M${IMAGES_PER_CRITIQUE}_n${MAX_PROMPTS}_seed${SEED}}
mkdir -p "${OUTPUT_DIR}"

# ---- Oracle judge: GenEval2's own Qwen3-VL-8B, benchmark answer ids -----------
JUDGE_MODEL=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/Qwen3-VL-8B-Instruct
JUDGE_PY=/home/mila/s/sarvjeet-singh.ghotra/scratch/installs/miniforge3/envs/geneval2/bin/python
JUDGE_QUESTION_BATCH=32
JUDGE_LOAD_TIMEOUT=1800
# Offset by job id so two jobs sharing a node do not collide.
BASE_PORT=$((8850 + (${SLURM_JOB_ID:-0} % 50) * 8))

mamba activate tar
module load cuda/12.1.1
export PYTHONPATH=$PYTHONPATH:/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The judge env's torch is built against a newer CUDA than the `module load
# cuda/...` the tar env needs, and that module puts its own lib64 first on
# LD_LIBRARY_PATH; libcusparse then picks up the module's older libnvJitLink and
# the import dies on `undefined symbol: __nvJitLinkComplete_12_4`. Put the judge
# env's own libnvJitLink first, for the judge process only.
judge_ld_path() {   # $1 = the judge env's python
    local nvjit
    nvjit=$("$1" -c 'import os,sysconfig;print(os.path.join(sysconfig.get_paths()["purelib"],"nvidia","nvjitlink","lib"))' 2>/dev/null)
    if [ -n "${nvjit}" ] && [ -d "${nvjit}" ]; then
        printf '%s' "${nvjit}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    else
        printf '%s' "${LD_LIBRARY_PATH}"
    fi
}

READY_DIR=$(mktemp -d)
PIDS=()
URLS=""
for ((g=0; g<N_GPUS; g++)); do
    PORT=$((BASE_PORT + g))
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$g]} \
        LD_LIBRARY_PATH="$(judge_ld_path "${JUDGE_PY}")" \
        ${JUDGE_PY} \
        llava/train/rl/pixel_reward_server.py \
        --model "${JUDGE_MODEL}" \
        --host 127.0.0.1 --port ${PORT} \
        --batch_size ${JUDGE_QUESTION_BATCH} \
        --answer_suffix geneval2 \
        --answer_id_mode geneval2 \
        --attn_implementation sdpa \
        --no-verbose \
        --ready_file "${READY_DIR}/ready_${g}" \
        > "${OUTPUT_DIR}/judge_${g}.log" 2>&1 &
    PIDS+=($!)
    URLS="${URLS}${URLS:+,}http://127.0.0.1:${PORT}"
done
trap 'for pid in "${PIDS[@]}"; do kill $pid 2>/dev/null; done; rm -rf "${READY_DIR}"' EXIT INT TERM

echo "Loading ${N_GPUS} oracle judges; logs in ${OUTPUT_DIR}/judge_*.log"
waited=0
while [ "$(ls "${READY_DIR}" 2>/dev/null | wc -l)" -lt "${N_GPUS}" ]; do
    for pid in "${PIDS[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            echo "A judge died while loading; see ${OUTPUT_DIR}/judge_*.log" >&2
            tail -20 "${OUTPUT_DIR}"/judge_*.log >&2
            exit 1
        fi
    done
    if [ ${waited} -ge ${JUDGE_LOAD_TIMEOUT} ]; then
        echo "Judges not ready after ${JUDGE_LOAD_TIMEOUT}s" >&2
        exit 1
    fi
    sleep 10
    waited=$((waited + 10))
done
echo "All judges ready after ${waited}s"

LORA_ARG=""
if [ -n "$LORA_PATH" ]; then
    LORA_ARG="--lora_path $LORA_PATH"
fi

# Sampling matches iter_geneval2_adhoc_e2e.sh.
torchrun --standalone --nproc_per_node=$N_GPUS eval/analyze/critique_sensitivity.py \
    --model $MODEL_PATH \
    --out_dir $OUTPUT_DIR \
    --prompts_file $EVAL_SET \
    --max_prompts $MAX_PROMPTS \
    --reward_server_url "${URLS}" \
    --conditions $CONDITIONS \
    --num_real $NUM_REAL \
    --images_per_critique $IMAGES_PER_CRITIQUE \
    --fail_threshold 0.5 \
    --seed $SEED \
    --temperature 1.0 \
    --top_k 1200 \
    --top_p 0.95 \
    --reflect_temperature 1.0 \
    --reflect_top_k 1200 \
    --reflect_top_p 0.95 \
    --reflect_tokens 128 \
    --prompt_chunk 16 \
    --batch_size 32 \
    --decode_batch_size 32 \
    --gen_seq_len 729 \
    --scale 0 \
    --cfg_scale 4.0 \
    --save_images first \
    --ar_path $AR_MODEL \
    --decoder_path /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/vq_ds16_t2i.pt \
    --encoder_path /tmp/ta_tok.pth \
    $LORA_ARG

echo "================ DONE ============="
echo "Summary: ${OUTPUT_DIR}/summary.txt"


exit 0

The probe is implemented and a small real run worked end to end: two GPUs on your idle interactive job, 4 prompts, K=2 critiques, M=2 images each, about 10 minutes, exit 0. That run also contradicts something I told you earlier.

It contradicts my earlier claim. I said a refine is basically a fresh resample. In this run it isn't: refine images keep the draft's layout, objects and style. On the flamingo/bicycle prompt I compared the draft, real, oracle, generic and fresh images side by side. The three refines are clearly edits of the draft; the fresh sample is a different picture. The numbers agree:

┌──────────────────────────────────────────┬───────────────────────────────┐
│                                          │ Spread of child AM (variance) │
├──────────────────────────────────────────┼───────────────────────────────┤
│ Refine (real / oracle / generic)         │ 0.001–0.003                   │
├──────────────────────────────────────────┼───────────────────────────────┤
│ Fresh sample, no draft in context        │ 0.028                         │
├──────────────────────────────────────────┼───────────────────────────────┤
│ Same codes decoded again (decoder noise) │ 0.0005                        │
└──────────────────────────────────────────┴───────────────────────────────┘

Picking up the draft's composition is how Tar behaves. The problem is that the refine rarely applies what the critique asks for. For example, an oracle critique of "make sure there are exactly five bicycles" still left two bicycles. That fits your earlier finding that named atoms get fixed only 14% of the time. This is only 4 prompts, so treat it as a hint until the full run is done.

How to run it

sbatch eval/analyze/critique_sensitivity.sh                      # SFT t20-16K, 256 prompts, K=4, M=4
MODEL_NAME=rl_t4_500 LORA_PATH=<adapter> sbatch eval/analyze/critique_sensitivity.sh

Settings can be overridden from the environment: MAX_PROMPTS, NUM_REAL, IMAGES_PER_CRITIQUE, CONDITIONS, SEED and OUTPUT_DIR. I estimate 256 prompts take about an hour on 4 GPUs. I haven't timed it at that size. If a run hits the 3-hour limit, resubmit with the same settings and it continues from where it stopped.

What the probe does

- Draft: one per prompt, scored per question by GenEval2's own Qwen3-VL judge. Each GPU gets its own judge, placed next to its policy rank.
- Critique conditions: each is followed by M refine images, using the same splice and sampling settings as iter_geneval2_adhoc_e2e.sh.
  - real: K critiques sampled from the policy.
  - shuffled: a real critique written for a different prompt, never a "looks good" one.
  - generic: a fixed critique with no specific content.
  - oracle: built from the questions the draft actually failed, in the SFT critique style (for example "there are no backpacks in the image\nCorrection: add three backpacks"). Every question in GenEval2's 800 prompts maps to a template, with no fallbacks.
- Controls: fresh is a new draft with no draft in context. redecode decodes the same draft codes again, which measures decoder noise.
- Always refines: every critique gets images even when it says "looks good", so conditions differ only in the text. The summary also has a real_policy row, which scores "looks good" critiques the way eval does, by keeping the draft.

summary.txt reports the following, for all drafts and separately for drafts that fail at least one question ("wrong") and drafts that pass everything:

- Child AM and change from the draft for each condition.
- Paired differences over drafts with 95% confidence intervals. The ones that decide your three suspicions are oracle−generic, real−generic and real−shuffled.
- Per-skill fix and break rates on wrong drafts: how often a failed question gets fixed, and how often a passing one gets broken.
- A split of the child-score variance into between-critique and within-critique parts. This shows how much of the GRPO signal on reflection tokens is real.

Changes along the way

- Subset sampling: GenEval2 is sorted by atom count, so the first 256 prompts would all be 3–5-atom prompts. --max_prompts now takes a fixed-seed random sample.
- Judge placement bug: the launcher inherited a bug from check_reward_pixel.sh. When CUDA_VISIBLE_DEVICES was a subset, the judges landed on the wrong GPUs. It's fixed in the new script. check_reward_pixel.sh still has the bug.

The test output is in the scratchpad (smoke/); nothing was written to results/. Both files are new and not yet committed.