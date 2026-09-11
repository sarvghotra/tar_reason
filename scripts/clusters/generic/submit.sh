#!/bin/bash
set -euo pipefail
PROFILE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TAR_CLUSTER="$(basename "$PROFILE_DIR")"
REPO_ROOT="$(cd "$PROFILE_DIR/../../.." && pwd)"
exec bash "$REPO_ROOT/scripts/submit.sh" "$@"
