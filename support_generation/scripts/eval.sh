#!/usr/bin/env bash
#
# Eval wrapper: proposed summary CSV.
# Usage:
#   bash support_generation/scripts/eval.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PKG_DIR}/results}"

cd "$PKG_DIR" || exit 1

echo "[eval] proposed → $OUTPUT_ROOT"
"$PYTHON" eval_llm_proposed.py \
    --root "$OUTPUT_ROOT" \
    --master-csv "$OUTPUT_ROOT/master_summary_proposed.csv"

echo ""
echo "[eval] done: $(date)"
