#!/bin/bash
# One SPPO / S-SPPO / RSPO data round, run from SPPO/: sample K responses, rank them with PairRM, build pairs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$SCRIPT_DIR/../../SPPO"
export WANDB_MODE="${WANDB_MODE:-offline}"
# vLLM inspects the model in a child interpreter; sitecustomize there disables torchao too.
export PYTHONPATH="${SCRIPT_DIR}/tf_torchao_site${PYTHONPATH:+:$PYTHONPATH}"

GPU_IDS="${GPU_IDS:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
MODEL="${MODEL:-${PROJECT_ROOT}/model_hub/Qwen2.5-1.5B-ultrachat200k}"
OUTDIR="${OUTDIR:-data-qwen2.5-1.5b-sppo-uc50k-iter1}"
PROMPTS="${PROMPTS:-${PROJECT_ROOT}/data/Ultrachat50k/train.jsonl}"
PAIRS="${PAIRS:-5}"
SIM_MODEL="${SIM_MODEL:-}"
SIM_BATCH_SIZE="${SIM_BATCH_SIZE:-64}"
MAXLEN="${MAXLEN:-256}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-1536}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --pairs) PAIRS="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --out_path) OUTDIR="$2"; shift ;;
    --prompt) PROMPTS="$2"; shift ;;
    --sim_model) SIM_MODEL="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

GEN_DIR="generated/${OUTDIR}"
RANK_DIR="ranking/${OUTDIR}"
SCORE_DIR="synthetic_data_${OUTDIR#*-}_score"
LOG_DIR="logs/${OUTDIR}"
SIM_ARGS=()
[[ -n "$SIM_MODEL" ]] && SIM_ARGS=(--sim_model "$SIM_MODEL")
FORCE_ARGS=()
[[ "$FORCE_REGENERATE" == "1" ]] && FORCE_ARGS=(--force)

STATE="$(python3 "$SCRIPT_DIR/dataset_spec.py" prepare \
    --model "$MODEL" --prompts "$PROMPTS" --pairs "$PAIRS" --maxlen "$MAXLEN" \
    --max_model_len "$VLLM_MAX_MODEL_LEN" --num_gpus "${#GPUS[@]}" \
    --gen_dir "$GEN_DIR" --rank_dir "$RANK_DIR" --score_dir "$SCORE_DIR" \
    ${SIM_ARGS[@]+"${SIM_ARGS[@]}"} ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"})"
read -r ACTION FRAC_LEN NUM_SHARDS <<< "$STATE"
if [[ "$ACTION" == "done" ]]; then
    echo "[generate] ${SCORE_DIR} is up to date for ${MODEL}"
    exit 0
fi
echo "[generate] ${OUTDIR}: frac_len=${FRAC_LEN} shards=${NUM_SHARDS} gpus=${GPU_IDS} logs=${LOG_DIR}"
mkdir -p "$LOG_DIR"

shard_job() {
    local sid="$1" gpu="$2"
    CUDA_VISIBLE_DEVICES="$gpu" python3 "$SCRIPT_DIR/generate.py" \
        --model "$MODEL" --prompts "$PROMPTS" --output_dir "$GEN_DIR" --pairs "$PAIRS" --maxlen "$MAXLEN" \
        --frac_len "$FRAC_LEN" --data_frac "$sid" \
        --gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" --max_num_seqs "$VLLM_MAX_NUM_SEQS" \
        --max_model_len "$VLLM_MAX_MODEL_LEN" --max_num_batched_tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
        > "${LOG_DIR}/generate_${sid}.log" 2>&1 || return 1
    CUDA_VISIBLE_DEVICES="$gpu" python3 "$SCRIPT_DIR/rank.py" \
        --model "$MODEL" --prompts "$PROMPTS" --gen_dir "$GEN_DIR" --rank_dir "$RANK_DIR" --pairs "$PAIRS" \
        --frac_len "$FRAC_LEN" --data_frac "$sid" \
        > "${LOG_DIR}/rank_${sid}.log" 2>&1 || return 1
}

sid=0
while (( sid < NUM_SHARDS )); do
    pids=()
    for gpu in "${GPUS[@]}"; do
        (( sid < NUM_SHARDS )) || break
        echo "[generate] shard ${sid} on GPU ${gpu}"
        shard_job "$sid" "$gpu" &
        pids+=("$!")
        sid=$((sid + 1))
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if (( failed )); then
        echo "[generate] a shard failed; see ${LOG_DIR}/" >&2
        exit 1
    fi
done

python3 "$SCRIPT_DIR/compute_prob.py" \
    --prompts "$PROMPTS" --gen_dir "$GEN_DIR" --rank_dir "$RANK_DIR" --score_dir "$SCORE_DIR" \
    --pairs "$PAIRS" --frac_len "$FRAC_LEN"
if [[ -n "$SIM_MODEL" ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" python3 "$SCRIPT_DIR/compute_sim.py" \
        --input "${SCORE_DIR}/train.parquet" --model "$SIM_MODEL" --batch_size "$SIM_BATCH_SIZE"
fi
python3 "$SCRIPT_DIR/dataset_spec.py" finalize --gen_dir "$GEN_DIR" --score_dir "$SCORE_DIR" ${SIM_ARGS[@]+"${SIM_ARGS[@]}"}
echo "[generate] wrote ${SCORE_DIR}/train.parquet"
