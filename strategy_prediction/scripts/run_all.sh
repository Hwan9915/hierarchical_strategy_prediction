#!/usr/bin/env bash
# Train then run cascade inference for one or more encoders.
# Usage: bash scripts/run_all.sh [--model distilbert|roberta-base|all] [--demo] [--no-log] [--gpus 3]
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

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

mkdir -p "${PKG_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"

train_args=(--model "${MODEL}" --gpus "${GPUS}")
infer_args=(--model "${MODEL}" --gpus "${GPUS}")
[[ -n "${DEMO}" ]] && train_args+=("${DEMO}")
if [[ "${NO_LOG}" -eq 1 ]]; then
    train_args+=(--no-log)
    infer_args+=(--no-log)
fi

run_pipeline() {
    bash "${SCRIPT_DIR}/train.sh" "${train_args[@]}"
    bash "${SCRIPT_DIR}/infer.sh" "${infer_args[@]}"
}

if [[ "${NO_LOG}" -eq 1 ]]; then
    run_pipeline
else
    LOG_FILE="${PKG_DIR}/logs/run_all_${MODEL}_${TS}.log"
    echo "[run_all.sh] logging to ${LOG_FILE}"
    run_pipeline 2>&1 | tee "${LOG_FILE}"
fi

echo "[run_all.sh] done."
