#!/bin/bash
# Self-play driver for SPIN, SPACE, T-SPIN and APSP, called by scripts/{APSP,SPIN,SPACE,T_SPIN}.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"
export WANDB_MODE="${WANDB_MODE:-offline}"
# torch.compile during vLLM startup crashes on this stack.
export VLLM_USE_STANDALONE_COMPILE=0
export VLLM_USE_AOT_COMPILE=0
export TORCHDYNAMO_DISABLE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export GPU_IDS="${GPU_IDS:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
IFS=',' read -r -a _gpus <<< "$GPU_IDS"
VLLM_WORLD_SIZE="${VLLM_WORLD_SIZE:-${#_gpus[@]}}"

METHOD="${METHOD:?set METHOD to spin, space, tspin or apsp}"
SFT_MODEL="${SFT_MODEL:?set SFT_MODEL}"
MODEL_ROOT="${MODEL_ROOT:?set MODEL_ROOT}"
MODEL_CFG="${MODEL_CFG:-qwen}"
TRAINER="${TRAINER:-BasicTrainer}"
PROMPT_FILE="${PROMPT_FILE:-${PROJECT_ROOT}/data/Ultrachat50k/train.jsonl}"
START_ITE="${START_ITE:-0}"
MAX_ITE="${MAX_ITE:-4}"
LR_PEAK="${LR_PEAK:-5e-7}"
LR_DECAY="${LR_DECAY:-1e-7}"
LR_DECAY_FROM_ITE="${LR_DECAY_FROM_ITE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-false}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
BETA="${BETA:-0.1}"
MU="${MU:-1.0}"
ALPHA="${ALPHA:-1.0}"

MAX_NEW="${MAX_NEW:-256}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-1536}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
VLLM_NO_ENFORCE_EAGER="${VLLM_NO_ENFORCE_EAGER:-1}"

case "$METHOD" in
spin)  RUN_TAG="${RUN_TAG:-SPIN}";   LOSS_ARGS=(loss=dpo "loss.beta=$BETA") ;;
space) RUN_TAG="${RUN_TAG:-SPACE}";  LOSS_ARGS=(loss=space "loss.mu=$MU") ;;
tspin) RUN_TAG="${RUN_TAG:-T_SPIN}"; LOSS_ARGS=(loss=tspin "loss.alpha=$ALPHA") ;;
apsp)  RUN_TAG="${RUN_TAG:-APSP}";   LOSS_ARGS=(loss=apsp "loss.beta=$BETA") ;;
*)
    echo "Unknown METHOD: $METHOD (expected spin, space, tspin or apsp)" >&2
    exit 1
    ;;
esac
if [[ "$METHOD" == apsp ]]; then
    INIT_POLICY="${INIT_POLICY:?set INIT_POLICY}"
    TAU="${TAU:?set TAU}"
else
    INIT_POLICY="$SFT_MODEL"
fi
RUN_DIR="${MODEL_ROOT}/${RUN_TAG}"

if [[ "${ENABLE_AUTO_EVAL:-0}" == 1 ]]; then
    export EVAL_SCRIPT="${EVAL_SCRIPT:-${PROJECT_ROOT}/llm_evaluation/lm-evaluation-harness/run_leaderboard_spin.sh}"
else
    unset EVAL_SCRIPT
fi

VLLM_ARGS=(--world_size "$VLLM_WORLD_SIZE" --max_new_tokens "$MAX_NEW"
    --gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" --max_model_len "$VLLM_MAX_MODEL_LEN"
    --max_num_seqs "$VLLM_MAX_NUM_SEQS" --max_num_batched_tokens "$VLLM_MAX_NUM_BATCHED_TOKENS")
if [[ "$VLLM_NO_ENFORCE_EAGER" == 1 ]]; then
    VLLM_ARGS+=(--no_enforce_eager)
fi

# sample MODEL OUT: one response per prompt -> OUT.jsonl
sample () {
    python generate_vllm.py --model "$1" --input_dir "$PROMPT_FILE" --output_dir "$2" "${VLLM_ARGS[@]}"
}

echo "[${METHOD}] RUN_DIR=${RUN_DIR} SFT_MODEL=${SFT_MODEL} MODEL_CFG=${MODEL_CFG} TRAINER=${TRAINER}"
echo "[${METHOD}] PROMPT_FILE=${PROMPT_FILE} ite${START_ITE}..ite$((MAX_ITE - 1)) GPU_IDS=${GPU_IDS} (vLLM TP=${VLLM_WORLD_SIZE})"
echo "[${METHOD}] lr=${LR_PEAK} (ite < ${LR_DECAY_FROM_ITE}), ${LR_DECAY} after; ${LOSS_ARGS[*]}"
if [[ "$METHOD" == apsp ]]; then
    echo "[${METHOD}] INIT_POLICY=${INIT_POLICY} p_base=${SFT_MODEL} tau=${TAU}"
fi
[[ -f "$PROMPT_FILE" ]] || { echo "[${METHOD}] prompt file not found: ${PROMPT_FILE}" >&2; exit 1; }

if [[ "$METHOD" == tspin ]]; then
    PROTO="${RUN_DIR}/proto/train.jsonl"
    echo "[${METHOD}] proto y_0 from ${SFT_MODEL}"
    sample "$SFT_MODEL" "${PROTO%.jsonl}"
fi

if (( START_ITE > 0 )); then
    policy="${RUN_DIR}/ite$((START_ITE - 1))"
    if [[ ! -f "${policy}/config.json" || ! -f "${policy}/train.jsonl" ]]; then
        echo "[${METHOD}] START_ITE=${START_ITE} needs the model and round data of ${policy}" >&2
        exit 1
    fi
else
    policy="$INIT_POLICY"
fi

for (( t = START_ITE; t < MAX_ITE; t++ )); do
    ite_dir="${RUN_DIR}/ite${t}"
    mkdir -p "$ite_dir"
    echo "[${METHOD}] ite${t}: round from ${policy}"
    iter_args=()
    if [[ "$METHOD" == tspin ]]; then
        if (( t == 0 )); then
            # y' at ite0 comes from the SFT model, so it is y_0 and the historical term is dropped.
            generated="$PROTO"
            iter_args=(loss.beta=0.0)
        else
            sample "$policy" "${ite_dir}/generated"
            generated="${ite_dir}/generated.jsonl"
            iter_args=("loss.beta=$BETA")
        fi
        python scripts/spin/merge_proto.py --generated "$generated" --proto "$PROTO" --output "${ite_dir}/train.jsonl"
    elif [[ "$METHOD" == apsp ]]; then
        sample "$policy" "${ite_dir}/generated"
        python score_apsp.py --base_model "$SFT_MODEL" --policy_model "$policy" \
            --input_jsonl "${ite_dir}/generated.jsonl" --output_jsonl "${ite_dir}/train.jsonl" \
            --tau "$TAU" --max_length "$MAX_LENGTH" --max_prompt_length "$MAX_PROMPT_LENGTH"
    else
        sample "$policy" "${ite_dir}/train"
    fi

    if (( t == 0 )); then
        datasets="[\"${RUN_TAG}/ite0\"]"
    else
        datasets="[\"${RUN_TAG}/ite$((t - 1))\",\"${RUN_TAG}/ite${t}\"]"
    fi
    if (( t < LR_DECAY_FROM_ITE )); then lr="$LR_PEAK"; else lr="$LR_DECAY"; fi

    echo "[${METHOD}] ite${t}: training ${ite_dir} (lr=${lr}) on ${datasets}"
    python -u train.py \
        model="$MODEL_CFG" \
        trainer="$TRAINER" \
        model.name_or_path="$policy" \
        "${LOSS_ARGS[@]}" ${iter_args[@]+"${iter_args[@]}"} \
        lr="$lr" \
        base_data_dir="$MODEL_ROOT" \
        datasets="$datasets" \
        ckpt_dir="${ite_dir}/" \
        batch_size="$TRAIN_BATCH_SIZE" \
        gradient_accumulation_steps="$GRAD_ACCUM_STEPS" \
        max_length="$MAX_LENGTH" \
        max_prompt_length="$MAX_PROMPT_LENGTH" \
        activation_checkpointing="$ACTIVATION_CHECKPOINTING" \
        wandb.enabled="$WANDB_ENABLED"
    policy="$ite_dir"
done

echo "[${METHOD}] done: ${policy}"
