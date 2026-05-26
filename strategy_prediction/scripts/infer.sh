#!/usr/bin/env bash
# Run cascade inference for one or more encoders.
# Usage: bash scripts/infer.sh [--model distilbert|roberta-base|all] [--no-log] [--gpus 3]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PKG_DIR}/.." && pwd)"

MODEL="all"
NO_LOG=0
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --no-log) NO_LOG=1; shift ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --demo) shift ;;  # accepted for compatibility with run_all.sh; infer uses full test set
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
OUT_BASE="${PKG_DIR}/output"

run_one() {
    local enc="$1"
    echo "[infer.sh] encoder=${enc}"
    "${PY}" -m strategy_prediction.strategy.infer \
        --encoder_key "${enc}" \
        --model_dir "${MODEL_DIR}" \
        --out_dir "${OUT_BASE}"
}

for enc in "${MODELS[@]}"; do
    if [[ "${NO_LOG}" -eq 1 ]]; then
        run_one "${enc}"
    else
        LOG_FILE="${PKG_DIR}/logs/infer_${enc}_${TS}.log"
        echo "[infer.sh] logging to ${LOG_FILE}"
        run_one "${enc}" 2>&1 | tee "${LOG_FILE}"
    fi
done

echo "[infer.sh] done."
