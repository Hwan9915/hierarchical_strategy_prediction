#!/usr/bin/env bash
#
# End-to-end: vLLM + infer (proposed), then eval summary.
# Usage:
#   bash support_generation/scripts/run_all.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_vllm.sh"
bash "${SCRIPT_DIR}/eval.sh"
