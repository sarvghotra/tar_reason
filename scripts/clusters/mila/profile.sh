# Mila: 29 nodes x 4 A100-80GB (a100l), 2 x 8 H100, 91 x 4 L40S and older GPUs
# (RTX8000/V100 are not supported by flash-attn 2). No account is needed.
# uv mode only.
TAR_MODULES=()
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
TAR_ACCOUNT_REQUIRED=0
GPU_TYPE="${GPU_TYPE:-a100l}"
