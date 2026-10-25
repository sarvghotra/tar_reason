#!/bin/bash
# Create this checkout's uv environment under the cluster's synced folder.
#
#   TAR_ROOT=$SCRATCH/tar_reason bash scripts/clusters/setup_uv.sh
#
# The venv lives at $TAR_ROOT/env/venvs/<uv.lock sha256[:12]> (immutable per lock,
# so queued jobs never see their environment change) and .venv points to it.
# $HOME quotas are small, so the uv cache goes to $TAR_ROOT/env/uv_cache: the
# user-level ~/.config/uv/uv.toml is written once, because `cluv sync` runs
# `uv sync` in a login shell that does not see this script's environment.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TAR_ROOT="${TAR_ROOT:-${SCRATCH:?Set TAR_ROOT or SCRATCH}/tar_reason}"
CACHE="$TAR_ROOT/env/uv_cache"
mkdir -p "$CACHE" "$TAR_ROOT/env/venvs"

CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml"
WANT="cache-dir = \"$CACHE\""
if [[ ! -f "$CONFIG" ]]; then
    mkdir -p "$(dirname "$CONFIG")"
    printf '# Written by tar_reason scripts/clusters/setup_uv.sh: keep the uv cache off $HOME.\n%s\n' "$WANT" > "$CONFIG"
    echo "Wrote $CONFIG ($WANT)"
elif ! grep -qxF "$WANT" "$CONFIG"; then
    echo "$CONFIG exists without '$WANT'; leaving it unchanged. Set UV_CACHE_DIR or edit it." >&2
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE}"

HASH="$(sha256sum "$REPO_ROOT/uv.lock" | cut -c1-12)"
VENV="$TAR_ROOT/env/venvs/$HASH"
UV_PROJECT_ENVIRONMENT="$VENV" uv --directory "$REPO_ROOT" sync --frozen
if [[ -e "$REPO_ROOT/.venv" && ! -L "$REPO_ROOT/.venv" ]]; then
    echo "$REPO_ROOT/.venv is a real directory; not replacing it with a link to $VENV" >&2
    exit 1
fi
ln -sfn "$VENV" "$REPO_ROOT/.venv"
echo ".venv -> $VENV"
"$VENV/bin/python" -c 'import torch, transformers, flash_attn; print("torch", torch.__version__, "transformers", transformers.__version__)'
