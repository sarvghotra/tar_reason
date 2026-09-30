#!/bin/bash
#SBATCH --time=2:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --partition long
#SBATCH --job-name=reward_noise_pixdec
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/tmp/reward_noise_pixdec_%j.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/tmp/reward_noise_pixdec_%j.err

# Reward noise from the stochastic AR de-tokenizer: sample GRPO-style groups
# with the policy, decode + judge every image DECODE_REPEATS times, and compare
# the re-decode spread of identical codes with the within-group spread GRPO
# sees. See reward_noise_to_pixel_decoding.py for the statistics.
#
# Layout: one judge on the last GPU, one shard (policy + de-tokenizer) on every
# GPU including the judge's (~15 G policy + ~19 G judge fits an 80 G A100). Shards
# write their own files; the analysis merges them at the end.
#
# Cost at the defaults: 64 prompts x 8 drafts x 8 decodes = 4096 judged images,
# ~0.45 s/image on the single judge -> ~35 min; decoding is spread over 4 GPUs.
#
# Re-running with the same OUT_DIR reuses the sampled codes and finished decode
# repeats, so a new decode setting on the *same* codes is: copy codes_shard*.jsonl
# into a fresh OUT_DIR and change DECODE_TEMPERATURE / CFG_SCALE.
#
#   sbatch eval/analyze/reward_noise_to_pixel_decoding.sh
#   BRANCH=1,8 TAG=refine sbatch eval/analyze/reward_noise_to_pixel_decoding.sh
#   ADAPTER_DIR=output_dir/rl_ft_oracle_t4/checkpoint-500 TAG=t4_500 sbatch ...

# ===================== Config params ========================
MODEL=${MODEL:-/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/output_dir/fir/slf_ref_edit_t20/checkpoint-16000}
ADAPTER_DIR=${ADAPTER_DIR:-}             # optional LoRA checkpoint, merged into MODEL
PROMPTS=${PROMPTS:-/home/mila/s/sarvjeet-singh.ghotra/tmp/geneval2_data_64.jsonl}
NUM_PROMPTS=${NUM_PROMPTS:-64}

# "8": groups are the 8 drafts of a prompt. "1,8": groups are the refined
# children of one draft (the t4 refine groups).
BRANCH=${BRANCH:-8}
DECODE_REPEATS=${DECODE_REPEATS:-8}

# De-tokenizer sampling. Training uses cfg 4.0, temperature 1.0, no top-k/p.
CFG_SCALE=${CFG_SCALE:-4.0}
DECODE_TEMPERATURE=${DECODE_TEMPERATURE:-1.0}
DECODE_TOP_K=${DECODE_TOP_K:-0}
DECODE_TOP_P=${DECODE_TOP_P:-1.0}

N_GPUS=${N_GPUS:-4}
JUDGE_GPU=$((N_GPUS - 1))
JUDGE_MODEL=/network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/Qwen3-VL-8B-Instruct
JUDGE_PY=/home/mila/s/sarvjeet-singh.ghotra/scratch/installs/miniforge3/envs/geneval2/bin/python
JUDGE_ARGS="--answer_id_mode geneval2"
JUDGE_LOAD_TIMEOUT=1800

AR_MODEL=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ar_dtok_lp_512px.pth
DECODER=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/vq_ds16_t2i.pt
ENCODER=/tmp/ta_tok.pth

TAG=${TAG:-draft}
SEED=${SEED:-421}
# ===================== Config params END =====================

REPO=/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason
OUT_DIR=${OUT_DIR:-${REPO}/results/reward_noise/${TAG}_b${BRANCH//,/-}_r${DECODE_REPEATS}_cfg${CFG_SCALE}_t${DECODE_TEMPERATURE}}
JOB=${SLURM_JOB_ID:-local}
PORT=$((8500 + ${SLURM_JOB_ID:-0} % 1000))
URL="http://127.0.0.1:${PORT}"

eval "$(mamba shell hook --shell bash)"
mamba activate tar
module load cuda/12.1.1
export PYTHONPATH=$PYTHONPATH:${REPO}
cd ${REPO}
mkdir -p "${OUT_DIR}"

echo "MODEL: ${MODEL}  ADAPTER_DIR: ${ADAPTER_DIR:-none}"
echo "BRANCH: ${BRANCH}  prompts: ${NUM_PROMPTS}  decode repeats: ${DECODE_REPEATS}"
echo "decode: cfg=${CFG_SCALE} temperature=${DECODE_TEMPERATURE} top_k=${DECODE_TOP_K} top_p=${DECODE_TOP_P}"
echo "OUT_DIR: ${OUT_DIR}"

if [ ! -f "${ENCODER}" ]; then
    cp /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/tar/ta_tok.pth /tmp/
fi

# Same libnvJitLink fix as output_dir/rl_ft_oracle_t4/bash_multi_gpus.sh: the
# cuda module's older copy otherwise shadows the judge env's own.
judge_ld_path() {
    local nvjit
    nvjit=$("$1" -c "import os,sysconfig;print(os.path.join(sysconfig.get_paths()['purelib'],'nvidia','nvjitlink','lib'))" 2>/dev/null)
    if [ -n "${nvjit}" ] && [ -d "${nvjit}" ]; then
        printf "%s" "${nvjit}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    else
        printf "%s" "${LD_LIBRARY_PATH}"
    fi
}

# ---------------- Judge ----------------
JUDGE_LOG="${OUT_DIR}/judge_${JOB}.log"
READY_FILE="${OUT_DIR}/.judge_ready_${JOB}"
rm -f "${READY_FILE}"
CUDA_VISIBLE_DEVICES=${JUDGE_GPU} \
    LD_LIBRARY_PATH="$(judge_ld_path "${JUDGE_PY}")" \
    ${JUDGE_PY} llava/train/rl/pixel_reward_server.py \
    --model "${JUDGE_MODEL}" --host 127.0.0.1 --port ${PORT} \
    --batch_size 32 --answer_suffix geneval2 ${JUDGE_ARGS} \
    --attn_implementation sdpa --ready_file "${READY_FILE}" \
    > "${JUDGE_LOG}" 2>&1 &
JUDGE_PID=$!
trap "kill ${JUDGE_PID} 2>/dev/null" EXIT INT TERM

waited=0
until [ -f "${READY_FILE}" ]; do
    if ! kill -0 ${JUDGE_PID} 2>/dev/null; then
        echo "judge died while loading; tail of ${JUDGE_LOG}:" >&2
        tail -40 "${JUDGE_LOG}" >&2
        exit 1
    fi
    if [ ${waited} -ge ${JUDGE_LOAD_TIMEOUT} ]; then
        echo "judge not ready after ${JUDGE_LOAD_TIMEOUT}s" >&2
        tail -40 "${JUDGE_LOG}" >&2
        exit 1
    fi
    sleep 10
    waited=$((waited + 10))
done
echo "judge ready after ${waited}s on GPU ${JUDGE_GPU} -> ${URL}"

# ---------------- Shards ----------------
COMMON_ARGS="--model_name_or_path ${MODEL} \
    --prompts ${PROMPTS} --num_prompts ${NUM_PROMPTS} \
    --branch ${BRANCH} --decode_repeats ${DECODE_REPEATS} \
    --cfg_scale ${CFG_SCALE} --decode_temperature ${DECODE_TEMPERATURE} \
    --decode_top_k ${DECODE_TOP_K} --decode_top_p ${DECODE_TOP_P} \
    --ar_path ${AR_MODEL} --encoder_path ${ENCODER} --decoder_path ${DECODER} \
    --reward_server_url ${URL} --output_dir ${OUT_DIR} --seed ${SEED}"
if [ -n "${ADAPTER_DIR}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --adapter_dir ${ADAPTER_DIR}"
fi

PIDS=()
for ((s = 0; s < N_GPUS; s++)); do
    CUDA_VISIBLE_DEVICES=${s} python eval/analyze/reward_noise_to_pixel_decoding.py \
        ${COMMON_ARGS} --shard_id ${s} --num_shards ${N_GPUS} \
        > "${OUT_DIR}/shard${s}_${JOB}.log" 2>&1 &
    PIDS+=($!)
done
echo "launched ${N_GPUS} shards; logs: ${OUT_DIR}/shard*_${JOB}.log"

FAILED=0
for ((s = 0; s < N_GPUS; s++)); do
    if ! wait ${PIDS[$s]}; then
        echo "shard ${s} failed; tail of its log:" >&2
        tail -30 "${OUT_DIR}/shard${s}_${JOB}.log" >&2
        FAILED=1
    fi
done
if [ ${FAILED} -ne 0 ]; then
    exit 1
fi

# ---------------- Analysis ----------------
python eval/analyze/reward_noise_to_pixel_decoding.py --output_dir ${OUT_DIR} --analyze_only



exit 0


What it measures

For each prompt, the script samples a GRPO group the same way training does. It then decodes and judges every image 8 times, going through PixelVQAReward.score_images exactly like training. That splits the score spread within a group into:

- Seen by GRPO: the spread within a group when each image is scored once.
- Noise: the spread across re-decodes of the same codes.
- Signal: the part that depends on the codes (the first minus the second).

The main output is noise_frac, the share of the advantage spread that is decoding noise.

How to run

sbatch eval/analyze/reward_noise_to_pixel_decoding.sh                       # draft groups (BRANCH=8)
BRANCH=1,8 TAG=refine sbatch eval/analyze/reward_noise_to_pixel_decoding.sh  # t4-style refine groups

The launcher runs one judge on GPU 3 and one shard on each of the 4 GPUs, then merges the shards and prints a summary. The default is 64 prompts × 8 images × 8 decodes = 4,096 judged images, which should take roughly 35–50 minutes. You can add ADAPTER_DIR=output_dir/rl_ft_oracle_t4/checkpoint-500 to measure the RL checkpoint instead of the SFT policy.

Reading the result

- noise_frac below about 0.2: decoding noise isn't what's holding RL back.
- Around 0.5 or higher: half or more of each advantage is noise. That fits the flat t4 run, and it's worth fixing before tuning anything else. The reliability line shows how much averaging k decodes per image would help.

Other outputs:
- Pair order flip rate: how often a single decode ranks two siblings the wrong way.
- Noise by score bucket: whether the noise is concentrated around a score of 0.5.
- Judge recheck: scores the same PNGs twice, the second time in reversed order, so judge or batching noise isn't blamed on the decoder.
- strips/: all 8 decodes of a few images side by side, so you can see what the noise looks like.
- summary.json and per_node.jsonl hold the numbers for later analysis.

Things to know

- Refine mode drops "looks good" children. Training doesn't decode them. The parent's decode noise also cancels out in child minus parent, so only the refined siblings are scored.
- You can compare de-tokenizer settings on the same images. The sampled codes are saved and reused if you rerun into the same output directory. Copy codes_shard*.jsonl into a new OUT_DIR and change DECODE_TEMPERATURE or CFG_SCALE to see if a gentler decode lowers the noise. Training currently uses temperature 1.0 and CFG 4.0.
- I created results/reward_noise/, because SLURM needs the log directory to exist before submission.


=== reward noise from pixel decoding: /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/reward_noise/draft_b8_r8_cfg4.0_t1.0 ===
groups 64  nodes 512  mean group size 8.0  decodes/node 8  mean score 0.609
within-group sd, single decode (what GRPO sees)  0.1439
re-decode sd of identical codes (noise)          0.0377
code-dependent sd (signal)                       0.1388
noise_frac = V_dec / V_obs                       0.069
reliability (true-vs-observed adv corr^2) when averaging k decodes: k=1: 0.931  k=2: 0.964  k=4: 0.982  k=8: 0.991  k=16: 0.995
test-retest advantage corr (1 decode vs 1 decode) 0.932
advantage sign agreement vs other decodes        0.935
sibling pair order flip rate                     0.094
questions whose prob crosses 0.5 across decodes  0.090
decode sd by mean score: [0.00,0.25]: 0.023 (n=37)  [0.25,0.50]: 0.039 (n=130)  [0.50,0.75]: 0.042 (n=176)  [0.75,1.00]: 0.034 (n=169)
judge on identical PNGs: mean |dAM| 1.58e-05, max |dAM| 1.28e-03 over 128 images
wrote /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/reward_noise/draft_b8_r8_cfg4.0_t1.0/summary.json