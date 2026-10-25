# Trillium GPU (Alliance): 60 nodes x 4 H100-80GB; jobs take whole nodes or one
# GPU, are submitted from $SCRATCH, and compute nodes have no internet.
# uv mode only. Confirm the account with `sshare -U` and set it in config.local.sh.
TAR_MODULES=()
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
GPU_TYPE="${GPU_TYPE:-h100}"
TAR_GPU_REQUEST_STYLE="${TAR_GPU_REQUEST_STYLE:-whole_node}"
