#!/bin/bash
# Self-play driver for SPPO, S-SPPO and RSPO, called by scripts/{SPPO,S_SPPO,RSPO}.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
# AF_UNIX socket paths are capped at 108 chars; datasets.map workers create theirs under TMPDIR.
export TMPDIR="${TMPDIR:-/tmp/apsp}"
mkdir -p "$TMPDIR"

export GPU_IDS="${GPU_IDS:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}}"
# Generation uses every GPU; training uses the first, since several visible GPUs turn on DataParallel.
TRAIN_GPU="${GPU_IDS%%,*}"

METHOD="${METHOD:?set METHOD to sppo, sspppo or rspo}"
SFT_MODEL="${SFT_MODEL:?set SFT_MODEL}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
ITER1_DATA_TAG="${ITER1_DATA_TAG:-$RUN_TAG}"
PROMPT_FILE="${PROMPT_FILE:-${PROJECT_ROOT}/data/Ultrachat50k/train.jsonl}"
ITER_NUM="${ITER_NUM:-3}"
LR_PEAK="${LR_PEAK:-5.0e-7}"
LR_DECAY="${LR_DECAY:-1.0e-7}"
LR_DECAY_FROM_ITER="${LR_DECAY_FROM_ITER:-3}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
LAM="${LAM:-1.0}"
SIM_MODEL="${SIM_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
RSPO_LOSS_TYPE="${RSPO_LOSS_TYPE:-sppo_reversekl}"
RSPO_REG_COEF="${RSPO_REG_COEF:-0.5}"

GEN_ARGS=()
METHOD_ARGS=()
case "$METHOD" in
sppo)
    WORK_DIR="${PROJECT_ROOT}/SPPO"; ENTRY=sppo/run_sppo.py; CONFIG="${PROJECT_ROOT}/configs/sppo/config.yaml"
    METHOD_ARGS=(--loss_type=sppo)
    ;;
sspppo)
    WORK_DIR="${PROJECT_ROOT}/SPPO"; ENTRY=sppo/run_sppo.py; CONFIG="${PROJECT_ROOT}/configs/sppo/config.yaml"
    METHOD_ARGS=(--loss_type=sspppo --lam="$LAM")
    GEN_ARGS=(--sim_model "$SIM_MODEL")
    ;;
rspo)
    WORK_DIR="${PROJECT_ROOT}/RSPO"; ENTRY=sppo/run_sppo_reg.py; CONFIG="${PROJECT_ROOT}/configs/rspo/config.yaml"
    METHOD_ARGS=(--loss_type="$RSPO_LOSS_TYPE" --reg_coef="$RSPO_REG_COEF" --ref_model_name_or_path="$SFT_MODEL")
    ;;
*)
    echo "Unknown METHOD: $METHOD (expected sppo, sspppo or rspo)" >&2
    exit 1
    ;;
esac
[[ -n "$ATTN_IMPLEMENTATION" ]] && METHOD_ARGS+=(--attn_implementation="$ATTN_IMPLEMENTATION")

echo "[${METHOD}] RUN_TAG=${RUN_TAG} SFT_MODEL=${SFT_MODEL}"
echo "[${METHOD}] PROMPT_FILE=${PROMPT_FILE} ITER_NUM=${ITER_NUM} GPU_IDS=${GPU_IDS} (training on ${TRAIN_GPU})"
echo "[${METHOD}] lr=${LR_PEAK} (iter < ${LR_DECAY_FROM_ITER}), ${LR_DECAY} after; extra args: ${METHOD_ARGS[*]}"

model="$SFT_MODEL"
for i in $(seq 1 "$ITER_NUM"); do
    if (( i == 1 )); then data_tag="$ITER1_DATA_TAG"; else data_tag="$RUN_TAG"; fi
    out="data-${data_tag}-iter${i}"
    dataset="${PROJECT_ROOT}/SPPO/synthetic_data_${out#*-}_score"
    output_dir="checkpoints/${RUN_TAG}-iter${i}"
    if (( i < LR_DECAY_FROM_ITER )); then lr="$LR_PEAK"; else lr="$LR_DECAY"; fi

    echo "[${METHOD}] iter ${i}: data ${out} from ${model}"
    bash "$SCRIPT_DIR/generate.sh" --model "$model" --prompt "$PROMPT_FILE" --out_path "$out" \
        ${GEN_ARGS[@]+"${GEN_ARGS[@]}"}

    echo "[${METHOD}] iter ${i}: training ${output_dir} (lr=${lr})"
    (
        cd "$WORK_DIR"
        CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python3 "$ENTRY" "$CONFIG" \
            --model_name_or_path="$model" \
            --dataset_mixer="$dataset" \
            --output_dir="$output_dir" \
            --run_name="${RUN_TAG}-iter${i}" \
            --learning_rate="$lr" \
            "${METHOD_ARGS[@]}"
    )
    # Absolute: a relative path parses as a hub repo id and hits the network under HF_HUB_OFFLINE=1.
    model="${WORK_DIR}/${output_dir}"
done

echo "[${METHOD}] done: ${model}"
