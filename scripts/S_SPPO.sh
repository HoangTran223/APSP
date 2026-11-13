#!/bin/bash
# S-SPPO (Semantic-Calibrated SPPO).
#   bash scripts/S_SPPO.sh [qwen2.5|gemma2|qwen3]    qwen2.5 by default
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/models.sh" "${1:-qwen2.5}"
export METHOD=sspppo
export RUN_TAG="${RUN_TAG:-${TAG}-sspppo-uc50k}"
exec bash "${SCRIPT_DIR}/sppo/run_iterations.sh"
