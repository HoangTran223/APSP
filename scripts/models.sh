#!/bin/bash
# Per-model settings, sourced by the method scripts: source models.sh qwen2.5|gemma2|qwen3

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HUB="${PROJECT_ROOT}/model_hub"

case "${1:-}" in
qwen2.5)
    TAG=qwen2.5-1.5b
    export SFT_MODEL="${SFT_MODEL:-${HUB}/Qwen2.5-1.5B-ultrachat200k}"
    export MODEL_ROOT="${MODEL_ROOT:-${SFT_MODEL}}"
    export INIT_POLICY="${INIT_POLICY:-${HUB}/Qwen2.5-1.5B}"
    export TAU="${TAU:-0.4}"
    export MODEL_CFG=qwen
    ;;
gemma2)
    TAG=gemma2-2b
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="${PROJECT_ROOT}/scripts/sppo/tf_torchao_site${PYTHONPATH:+:$PYTHONPATH}"
    # torch.compile during vLLM startup crashes on this stack.
    export VLLM_USE_STANDALONE_COMPILE=0
    export VLLM_USE_AOT_COMPILE=0
    export TORCHDYNAMO_DISABLE=1
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    # Gemma-2 logit soft-capping is only applied in eager; SDPA skips it.
    export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
    export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
    export WANDB_ENABLED="${WANDB_ENABLED:-false}"
    GEMMA_HUB="${GEMMA_HUB:-${HUB}/gemma-2-2b-ultrachat200k}"
    export SFT_MODEL="${SFT_MODEL:-${GEMMA_HUB}/final}"
    export MODEL_ROOT="${MODEL_ROOT:-${GEMMA_HUB}}"
    export INIT_POLICY="${INIT_POLICY:-${HUB}/gemma-2-2b}"
    export TAU="${TAU:-0.03}"
    export MODEL_CFG=gemma2
    ;;
qwen3)
    TAG=qwen3-4b
    export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
    export SFT_MODEL="${SFT_MODEL:-${HUB}/Qwen3-4B-ultrachat200k}"
    export MODEL_ROOT="${MODEL_ROOT:-${SFT_MODEL}}"
    export INIT_POLICY="${INIT_POLICY:-${HUB}/Qwen3-4B-Base}"
    export TAU="${TAU:-0.1}"
    export MODEL_CFG=qwen3
    ;;
*)
    echo "usage: bash $0 [qwen2.5|gemma2|qwen3]" >&2
    exit 1
    ;;
esac
