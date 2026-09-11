# Shared profile selection; does not activate environments or allocate resources.
if [[ -z "${TAR_CLUSTER:-}" ]]; then
    if [[ -f "$REPO_ROOT/.cluster" ]]; then
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
