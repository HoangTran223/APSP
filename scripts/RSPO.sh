#!/bin/bash
# RSPO (Regularized SPO, reverse-KL variant).
#   bash scripts/RSPO.sh [qwen2.5|gemma2|qwen3]    qwen2.5 by default
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/models.sh" "${1:-qwen2.5}"
export METHOD=rspo
export RUN_TAG="${RUN_TAG:-${TAG}-rspo-uc50k}"
# Iteration 1 trains on the SPPO iteration-1 round (same SFT policy).
export ITER1_DATA_TAG="${ITER1_DATA_TAG:-${TAG}-sppo-uc50k}"
exec bash "${SCRIPT_DIR}/sppo/run_iterations.sh"
