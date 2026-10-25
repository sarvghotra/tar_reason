# Narval (Alliance): 159 nodes x 4 A100-40GB; compute nodes have no internet.
# Evaluation stages fit in 40 GB; RL training (68-77 GB/GPU on H100) does not.
# uv mode only. Confirm the account with `sshare -U` (def-bengioy; Narval is
# not in the Mila allocation) and set it in config.local.sh.
TAR_MODULES=()
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
GPU_TYPE="${GPU_TYPE:-a100}"
