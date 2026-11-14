#!/bin/bash
# Make Mila (singhsd) copies of jeet's vLLM eval launchers from the frozen snapshot of his checkout, changing only
# paths / env activation / output root, so generation and judging run exactly as his runs (e.g. t4's):
#   - `eval "$(mamba shell hook ...)"` -> source mila_env.sh (mamba activate = his env on PATH)
#   - /home/mila/s/sarvjeet-singh.ghotra/scratch/ -> /network/scratch/s/sarvjeet-singh.ghotra/ (same files)
#   - SBATCH logs -> our slurm_logs; results -> $OUT_ROOT (default results_jeetpipe/) instead of his results/
#   - CLIP-FlanT5 scorer -> our copy of his t2v_metrics_clipt5 (_PRETRAIN_DIR repointed off his /home)
#   - viz / flops helpers -> the snapshot's copies
# Run the outputs from the snapshot dir (they use $(pwd) as the repo root):
#   cd $SNAP && sbatch --export=ALL,MODEL_NAME=...,MODEL_PATH=...,LORA_PATH=... <repo>/scripts/eval/jeetpipe/<x>_mila.sh
set -e
HERE=$(cd $(dirname $0) && pwd)
SNAP=/network/scratch/s/singhsd/jeet_snapshot/tar_reason_20261113
OUT_ROOT=/network/scratch/s/singhsd/tar_reason/results_jeetpipe
for SRC in iter_geneval2_adhoc_e2e_MDP_vllm iter_tiif_bench_adhoc_e2e_MDP_vllm iter_genai_bench_adhoc_MDP_vllm; do
  sed -e "s|^eval \"\$(mamba shell hook --shell bash)\"|source ${HERE}/mila_env.sh|" \
      -e "s|^#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/.*|#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/${SRC}_%j.out|" \
      -e "/^#SBATCH -e \/home\/mila\/s\/sarvjeet-singh.ghotra\/tmp\/tmp\//d" \
      -e "s|/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/|${OUT_ROOT}/|g" \
      -e "s|/network/scratch/s/sarvjeet-singh.ghotra/git/t2v_metrics_clipt5|/network/scratch/s/singhsd/jeet_snapshot/t2v_metrics_clipt5|g" \
      -e "s|/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/eval/|${SNAP}/eval/|g" \
      -e "s|/home/mila/s/sarvjeet-singh.ghotra/.cache/vllm|/network/scratch/s/singhsd/.cache/vllm|g" \
      -e "s|/home/mila/s/sarvjeet-singh.ghotra/scratch/|/network/scratch/s/sarvjeet-singh.ghotra/|g" \
      ${SNAP}/scripts/eval/${SRC}.sh > ${HERE}/${SRC}_mila.sh
  chmod +x ${HERE}/${SRC}_mila.sh
  echo "${SRC}: $(diff ${SNAP}/scripts/eval/${SRC}.sh ${HERE}/${SRC}_mila.sh | grep -c '^>') lines changed"
  grep -n "sarvjeet-singh.ghotra/\(tmp\|git/tar_reason/results\)\|/home/mila/s/sarvjeet" ${HERE}/${SRC}_mila.sh || true
done
