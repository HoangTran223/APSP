#!/bin/bash
# SPACE (Self-Play via Noise Contrastive Estimation).
#   bash scripts/SPACE.sh [qwen2.5|gemma2|qwen3]    qwen2.5 by default
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/models.sh" "${1:-qwen2.5}"
export METHOD=space
exec bash "${SCRIPT_DIR}/spin/run_iterations.sh"
