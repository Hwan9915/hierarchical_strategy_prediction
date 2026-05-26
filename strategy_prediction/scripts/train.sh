#!/usr/bin/env bash
# Train all 5 modules (2 identifiers + 3 predictors) for one or more encoders.
# Usage: bash scripts/train.sh [--model distilbert|roberta-base|all] [--demo] [--no-log] [--gpus 3]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PKG_DIR}/.." && pwd)"

MODEL="all"
DEMO=""
NO_LOG=0
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --demo)  DEMO="--demo"; shift ;;
        --no-log) NO_LOG=1; shift ;;
        --gpus) GPUS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "${MODEL}" == "all" ]]; then
    MODELS=(distilbert roberta-base)
else
    MODELS=("${MODEL}")
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY="${PYTHON_BIN}"
else
    PY=python3
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

mkdir -p "${PKG_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"

MODEL_DIR="${PKG_DIR}/model"

REQUIRED_CKPTS=(
    "identifier/%s/exploration_support_identifier.ckpt"
    "identifier/%s/comforting_action_identifier.ckpt"
    "predictor/%s/exploration_predictor.ckpt"
    "predictor/%s/comforting_predictor.ckpt"
    "predictor/%s/action_predictor.ckpt"
)

has_all_ckpts() {
    local enc="$1"
    local rel
    for rel in "${REQUIRED_CKPTS[@]}"; do
        # shellcheck disable=SC2059
        [[ -f "${MODEL_DIR}/$(printf "${rel}" "${enc}")" ]] || return 1
    done
    return 0
}

run_one() {
    local enc="$1"
    echo "[train.sh] encoder=${enc} demo='${DEMO}'"
    "${PY}" -m strategy_prediction.stage.train \
        --encoder_key "${enc}" \
        --model_dir "${MODEL_DIR}" ${DEMO}
    "${PY}" -m strategy_prediction.strategy.train \
        --encoder_key "${enc}" \
        --model_dir "${MODEL_DIR}" ${DEMO}
}

for enc in "${MODELS[@]}"; do
    if has_all_ckpts "${enc}"; then
        echo "[train.sh] encoder=${enc} already has all 5 checkpoints under ${MODEL_DIR} → skip"
        continue
    fi
    if [[ "${NO_LOG}" -eq 1 ]]; then
        run_one "${enc}"
    else
        LOG_FILE="${PKG_DIR}/logs/train_${enc}_${TS}.log"
        echo "[train.sh] logging to ${LOG_FILE}"
        run_one "${enc}" 2>&1 | tee "${LOG_FILE}"
    fi
done

echo "[train.sh] done."
