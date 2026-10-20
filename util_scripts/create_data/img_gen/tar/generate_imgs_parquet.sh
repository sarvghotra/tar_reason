#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=48
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH -J conceptmix_20K
#SBATCH -o /home/jeet/scratch/data/syn_data_critique/conceptmix/20K/gen_imgs_%j_%t.out
#SBATCH -e /home/jeet/scratch/data/syn_data_critique/conceptmix/20K/gen_imgs_%j_%t.err


source /scratch/jeet/installs/miniforge3/bin/activate tar
module load cuda/12.2


if [ ! -f /tmp/ta_tok.pth ]; then
    cp /scratch/jeet/models/pre_train/tar/ta_tok.pth /tmp/
fi

set -e

cd /home/jeet/scratch/git/tar_reason
export PYTHONPATH=$PYTHONPATH:/home/jeet/scratch/git/tar_reason/


N_GPUS=4
MODEL=/home/jeet/scratch/git/tar_reason/output_dir/slf_ref_edit_t21/checkpoint-17000
PROMPT_FILE=/home/jeet/scratch/data/conceptmix/prompts_train_20K.jsonl
OUTPUT_DIR=/home/jeet/scratch/data/syn_data_critique/conceptmix/20K


torchrun --nproc_per_node=$N_GPUS util_scripts/create_data/img_gen/tar/generate_imgs_parquet.py \
        --ar_path /scratch/jeet/models/pre_train/tar/ar_dtok_lp_512px.pth \
        --decoder_path /scratch/jeet/models/pre_train/tar/vq_ds16_t2i.pt \
        --encoder_path /tmp/ta_tok.pth \
        --batch_size 32 \
        --bf16 \
        --model $MODEL \
        --prompts_file $PROMPT_FILE \
        --output_dir $OUTPUT_DIR \
        # --samples 16



mamba activate tar
python util_scripts/sft_data_convert_format/viz/edit.py \
        $OUTPUT_DIR \
        /home/jeet/scratch/git/tar_reason/util_scripts/sft_data_convert_format/viz/edit/conceptmix_20K


# mamba activate vllm
# module load cuda/12.6.0



# export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID   # local disk, not NFS
# export TRITON_HOME=/tmp/triton_home_$SLURM_JOB_ID
# mkdir -p "$TRITON_CACHE_DIR" "$TRITON_HOME"

# PYTHON_INCLUDE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
# LOCAL_PY_INCLUDE="/tmp/python_include_${SLURM_JOB_ID}"

# cp -a "$PYTHON_INCLUDE" "$LOCAL_PY_INCLUDE"
# export CPATH="$LOCAL_PY_INCLUDE${CPATH:+:$CPATH}"

# printf '#include <Python.h>\n' |
#     gcc -E -I"$LOCAL_PY_INCLUDE" -x c - >/dev/null || exit 1


# # Triton JIT-compiles a small CUDA helper the first time vLLM touches the GPU.
# # Fail fast here if the conda Python headers are incomplete, instead of after
# # loading the full VL checkpoint.
# PYTHON_INCLUDE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
# if [ ! -f "$PYTHON_INCLUDE/Python.h" ] || [ ! -f "$PYTHON_INCLUDE/cpython/dictobject.h" ]; then
#     echo "ERROR: incomplete Python headers in active env: $PYTHON_INCLUDE" >&2
#     echo "Expected Python.h and cpython/dictobject.h. Reinstall/repair python in the vllm env." >&2
#     exit 1
# fi
# printf '#include <Python.h>\n' | gcc -E -I"$PYTHON_INCLUDE" -x c - >/dev/null || {
#     echo "ERROR: gcc cannot preprocess Python.h from $PYTHON_INCLUDE" >&2
#     exit 1
# }




# python util_scripts/create_data/generate_critic_vllm.py \
#     --images_dir /home/jeet/scratch/data/critique/geneval_critique_val/imgs_test12/ \
#     --output_jsonl /home/jeet/scratch/data/critique/geneval_critique_val/critic_test12.jsonl \
#     --output_parquet /home/jeet/scratch/data/critique/geneval_critique_val/critic_test12/ \
#     --max_pixels 262144 \
#     --num_gpus 1 \
#     --tensor_parallel_size 1 \
#     --chunk_size 128 \
#     --max_tokens 512 \
#     --safetensors_load_strategy lazy



# mamba activate tar
# python /home/jeet/scratch/git/tar_reason/util_scripts/sft_data_convert_format/viz/critique_rlhfv.py \
#     /home/jeet/scratch/data/syn_data_critique/generated/10/t2i_2M/critic_data \
#     /home/jeet/scratch/data/syn_data_critique/generated/10/t2i_2M/critic_data_viz

