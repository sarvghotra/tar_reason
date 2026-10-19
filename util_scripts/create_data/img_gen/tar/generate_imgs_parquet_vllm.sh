#!/bin/bash
#SBATCH --time=2:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0G
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --job-name=genimgs_13K_2
#SBATCH --array=0-1            # one 4-GPU node per task; any count resumes any other
#SBATCH -o /home/jeet/scratch/data/syn_data_critique/conceptmix/prompts_13K_vqa_unverif_2/%j_%t_MN.out
#SBATCH -e /home/jeet/scratch/data/syn_data_critique/conceptmix/prompts_13K_vqa_unverif_2/%j_%t_MN.err

type module >/dev/null 2>&1 || source /etc/profile
eval "$(/scratch/jeet/installs/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate vllm
module load cuda/12.9
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [ ! -f /tmp/ta_tok.pth ]; then
    cp /scratch/jeet/models/pre_train/tar/ta_tok.pth /tmp/
fi

set -e

cd /home/jeet/scratch/git/tar_reason
export PYTHONPATH=$PYTHONPATH:/home/jeet/scratch/git/tar_reason/
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-$$}   # local disk, not NFS


N_GPUS=4
MODEL=/home/jeet/scratch/git/tar_reason/output_dir/slf_ref_edit_t21/checkpoint-17000
PROMPT_FILE=/home/jeet/scratch/data/syn_data_critique/conceptmix/prompts_38K.jsonl
# /home/jeet/scratch/data/syn_data_critique/conceptmix/prompts_13K_vqa_unverif_2.jsonl
OUTPUT_DIR=/home/jeet/scratch/data/syn_data_critique/conceptmix/prompts_38K

# Fixed shard layout: keep LOGICAL_WORLD_SIZE (and --chunk_rows, --seed,
# --samples, prompts) unchanged across relaunches so finished shards are skipped,
# whatever the number of array tasks or GPUs. 4 matches earlier 1x4-GPU runs.
LOGICAL_WORLD_SIZE=4
# Without --array these fall back to a single task (the whole input). Override
# NUM_TASKS when resubmitting only some indices (e.g. --array=1 of 0-1).
NUM_TASKS=${NUM_TASKS:-${SLURM_ARRAY_TASK_COUNT:-1}}
TASK_ID=${TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}


python util_scripts/create_data/img_gen/tar/generate_imgs_parquet_vllm.py \
        --ar_path /scratch/jeet/models/pre_train/tar/ar_dtok_lp_512px.pth \
        --decoder_path /scratch/jeet/models/pre_train/tar/vq_ds16_t2i.pt \
        --encoder_path /tmp/ta_tok.pth \
        --num_gpus $N_GPUS \
        --batch_size 256 \
        --decode_batch_size 32 \
        --bf16 \
        --model $MODEL \
        --prompts_file $PROMPT_FILE \
        --output_dir $OUTPUT_DIR \
        --gpu_memory_utilization 0.5 \
        --logical_world_size $LOGICAL_WORLD_SIZE \
        --num_tasks $NUM_TASKS \
        --task_id $TASK_ID


# --samples 16
