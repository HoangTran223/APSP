#!/bin/bash
# SFT on UltraChat200k: bash scripts/sft.sh qwen2.5|gemma2|qwen3
# Writes output/<name>_sft_ultrachat200k_<date>/final. GPU_ID picks the GPU; SMOKE=1 runs a short check.
set -euo pipefail

MODEL="${1:-}"
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$PROJECT_ROOT"

case "$MODEL" in
qwen2.5)
    MODEL_PATH="${MODEL_PATH:-model_hub/Qwen2.5-1.5B}"; NAME=qwen2.5_1.5b
    NUM_EPOCHS="${NUM_EPOCHS:-1}"; LR="${LR:-2e-5}"; OPTIM="${OPTIM:-adamw_torch}"; ATTN_IMPL="${ATTN_IMPL:-sdpa}"
    ;;
gemma2)
    MODEL_PATH="${MODEL_PATH:-model_hub/gemma-2-2b}"; NAME=gemma2_2b
    NUM_EPOCHS="${NUM_EPOCHS:-1}"; LR="${LR:-2e-5}"; OPTIM="${OPTIM:-paged_adamw_8bit}"; ATTN_IMPL="${ATTN_IMPL:-eager}"
    ;;
qwen3)
    MODEL_PATH="${MODEL_PATH:-model_hub/Qwen3-4B-Base}"; NAME=qwen3_4b
    NUM_EPOCHS="${NUM_EPOCHS:-1}"; LR="${LR:-2e-5}"; OPTIM="${OPTIM:-paged_adamw_8bit}"; ATTN_IMPL="${ATTN_IMPL:-sdpa}"
    ;;
*)
    echo "usage: bash scripts/sft.sh qwen2.5|gemma2|qwen3" >&2
    exit 1
    ;;
esac

CONDA_ENV="${CONDA_ENV:-apsp-qwen3}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
    if [ -z "${CONDA_SH:-}" ]; then
        for _cs in "${HOME}/miniconda3/etc/profile.d/conda.sh" \
                   "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"; do
            if [ -n "$_cs" ] && [ -f "$_cs" ]; then CONDA_SH="$_cs"; break; fi
        done
    fi
    # shellcheck disable=SC1090
    source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
    conda activate "$CONDA_ENV"
fi

DATASET_PATH="${DATASET_PATH:-data/ultrachat_200k}"
OUTPUT_DIR="${OUTPUT_DIR:-output/${NAME}_sft_ultrachat200k_$(date +%Y%m%d_%H%M%S)}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LR_SCHED="${LR_SCHED:-cosine}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_PROC="${NUM_PROC:-8}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
LOG_STEPS="${LOG_STEPS:-10}"
SEED="${SEED:-42}"

EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "[SFT] SMOKE mode: 256 train / 64 eval samples."
    OUTPUT_DIR="output/${NAME}_sft_SMOKE_$(date +%Y%m%d_%H%M%S)"
    EXTRA_ARGS+=(--max_train_samples 256 --max_eval_samples 64
                 --gradient_accumulation_steps 2 --save_steps 50 --logging_steps 2)
else
    EXTRA_ARGS+=(--gradient_accumulation_steps "$GRAD_ACCUM" --save_steps "$SAVE_STEPS"
                 --logging_steps "$LOG_STEPS")
fi

echo "[SFT] MODEL=$MODEL_PATH"
echo "[SFT] DATA=$DATASET_PATH"
echo "[SFT] OUT=$OUTPUT_DIR"
echo "[SFT] GPU=$CUDA_VISIBLE_DEVICES  bs=$PER_DEVICE_BS x accum=$GRAD_ACCUM  lr=$LR ($LR_SCHED)  epochs=$NUM_EPOCHS  seq=$MAX_SEQ_LEN  attn=$ATTN_IMPL  optim=$OPTIM"

python -u sft.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --per_device_eval_batch_size "$PER_DEVICE_BS" \
    --learning_rate "$LR" \
    --lr_scheduler_type "$LR_SCHED" \
    --warmup_ratio "$WARMUP_RATIO" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --optim "$OPTIM" \
    --attn_implementation "$ATTN_IMPL" \
    --gradient_checkpointing \
    --num_proc "$NUM_PROC" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --seed "$SEED" \
    --run_name "${NAME}-sft-ultrachat200k" \
    "${EXTRA_ARGS[@]}"

echo "[SFT] Done. Final model: $OUTPUT_DIR/final"
