#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --open-mode=append
#SBATCH -J tiif_self
#SBATCH -o /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.err
# Self-verified selection on Fir TIIF runs (paper repo results/self_verified_selection/, score am_pol):
# eval/tiif_self_judge.py --questions on every (model, seed) in PAIRS, spread over the node's 4 GPUs (pairs on the same
# GPU run one after another; resumable, re-submit the same command). TIIF's questions are the same for every model and
# seed, so they are read from one already-judged run (QSRC) through QDIR symlinks; their answers are never read.
#   PAIRS="t4_e2eC200:17 t4_e2eC200:91 t4_e2eC200:13 t4_e2eC300:17 t4_e2eC300:91 t4_e2eC300:13" \
#   LORAS="t4_e2eC200=<run>/checkpoint-200 t4_e2eC300=<run>/checkpoint-300" sbatch scripts/eval/tiif_self_judge_fir.sh
# Then (login node, after the 72B judge) eval/analyze/tiif_rollback.py --score am_pol --judge_dir eval_results_qwen72b.
set -eu
cd /home/dars11/scratch/git/tar_reason
: "${PAIRS:?}" "${LORAS:?}"
BASE=${BASE:-/home/dars11/scratch/models/tar/sft/darshan_E450_orc500_crisft_t4_merged}
RES=/scratch/dars11/tar_reason/results_jeet/tiif-bench-testmini_eval
QSRC=${QSRC:-$RES/iter_adhoc_MDP_n4_t4_e2eC200_512px_repeat1/seed_17/eval_results_qwen72b/t4_e2eC200_iter1}
QDIR=/scratch/dars11/tar_reason/self_judge/tiif/questions
OUTROOT=/scratch/dars11/tar_reason/self_judge/tiif
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
declare -A LORA; for kv in $LORAS; do LORA[${kv%%=*}]=${kv#*=}; done
echo "$(date) job=${SLURM_JOB_ID:-} PAIRS=$PAIRS LORAS=$LORAS BASE=$BASE QSRC=$QSRC"
NG=$(nvidia-smi -L | wc -l); i=0; declare -a Q
for p in $PAIRS; do Q[$((i % NG))]+="$p "; i=$((i + 1)); done
for g in "${!Q[@]}"; do
  ( for p in ${Q[$g]}; do
      m=${p%%:*}; s=${p#*:}; out=$OUTROOT/$m; mkdir -p $out $QDIR
      ln -sfn $QSRC $QDIR/${m}_iter1
      CUDA_VISIBLE_DEVICES=$g ~/envs/tar/bin/python eval/tiif_self_judge.py --questions \
        --run_dir $RES/iter_adhoc_MDP_n4_${m}_512px_repeat1 --seeds $s --iter_tag $m --n_iters 4 \
        --judge_dir $QDIR --model $BASE --lora_path ${LORA[$m]} --ta_tok /home/dars11/scratch/models/tar/ta_tok.pth \
        --siglip2 /home/dars11/scratch/models/tar/siglip2-so400m-patch14-384 --out_dir $out >> $out/log_seed_$s.txt 2>&1 \
        && echo "$(date) done $m seed $s: $(wc -l < $out/seed_${s}_am.jsonl) am rows" || echo "$(date) FAILED $m seed $s"
    done ) &
done
wait
