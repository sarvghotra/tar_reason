# Shared profile selection; does not activate environments or allocate resources.
if [[ -z "${TAR_CLUSTER:-}" ]]; then
    # Inside a job the scheduler names the cluster; otherwise the checkout's .cluster.
    if [[ "${SLURM_CLUSTER_NAME:-}" =~ ^[a-zA-Z0-9_-]+$ && -f "$REPO_ROOT/scripts/clusters/$SLURM_CLUSTER_NAME/profile.sh" ]]; then
        TAR_CLUSTER="$SLURM_CLUSTER_NAME"
    elif [[ -f "$REPO_ROOT/.cluster" ]]; then
        read -r TAR_CLUSTER < "$REPO_ROOT/.cluster"
    else
        TAR_CLUSTER=generic
    fi
fi
[[ "$TAR_CLUSTER" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid TAR_CLUSTER' >&2; return 1; }
TAR_PROFILE_DIR="$REPO_ROOT/scripts/clusters/$TAR_CLUSTER"
[[ -f "$TAR_PROFILE_DIR/profile.sh" ]] || { echo "Missing cluster profile: $TAR_CLUSTER" >&2; return 1; }
source "$TAR_PROFILE_DIR/profile.sh"
if [[ -f "$TAR_PROFILE_DIR/config.local.sh" ]]; then
    source "$TAR_PROFILE_DIR/config.local.sh"
fi
export TAR_CLUSTER
