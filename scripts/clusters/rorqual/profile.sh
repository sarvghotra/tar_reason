# Rorqual (Alliance): 81 nodes x 4 H100-80GB; compute nodes have no internet.
# uv mode only (TAR_ROOT=$SCRATCH/tar_reason). Confirm the account with `sshare -U`
# and set it in config.local.sh (Mila members: an rrg-bengioy-ad allocation).
TAR_MODULES=()
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
GPU_TYPE="${GPU_TYPE:-h100}"
