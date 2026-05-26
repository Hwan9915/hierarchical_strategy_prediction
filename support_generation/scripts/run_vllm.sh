#!/usr/bin/env bash
#
# vLLM server + support-generation inference (proposed), per model.
#
# Usage:
#   bash support_generation/scripts/run_vllm.sh
#
# Common env vars:
#   PORT=8000
#   PYTHON=python3
#   DATA_DIR=<repo>/strategy_prediction/output
#   OUTPUT_ROOT=<pkg>/results
#   ALL_LLM=1                  # llama-8b/70b + qwen3-8b/32b
#   ALL_CLASSIFIER=1           # proposed: roberta + distilbert
#   MODEL_ALIAS=llama-8b-instruct
#   CLASSIFIER=roberta
#   GPU_MEMORY_UTILIZATION=0.95
#   MAX_MODEL_LEN=4096
#   RUN_INFER=1
#   RUN_CHAT_SAMPLE=0
#   INFER_MAX_WORKERS=25
#   INFER_TEMPERATURE=0.5
#   INFER_MAX_TOKENS=128
#   INFER_START=0
#   INFER_END=-1
#   INFER_EXTRA='--use_esconv_input_dialog 0'

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PORT="${PORT:-8000}"
HOST="http://localhost:${PORT}"
VLLM_BASE="${HOST}/v1"

GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_LEN="${MAX_MODEL_LEN:-4096}"

PYTHON="${PYTHON:-python3}"
RUN_INFER="${RUN_INFER:-1}"
RUN_CHAT_SAMPLE="${RUN_CHAT_SAMPLE:-0}"

INFER_MAX_WORKERS="${INFER_MAX_WORKERS:-25}"
INFER_TEMPERATURE="${INFER_TEMPERATURE:-0.5}"
INFER_MAX_TOKENS="${INFER_MAX_TOKENS:-128}"
INFER_START="${INFER_START:-0}"
INFER_END="${INFER_END:--1}"

DATA_DIR="${DATA_DIR:-$(dirname "$PKG_DIR")/strategy_prediction/output}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PKG_DIR}/results}"
LOG_DIR="${PKG_DIR}/logs"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${SUMMARY_LOG:-${LOG_DIR}/run_vllm_$(date +%Y%m%d_%H%M%S).log}"

ALL_LLM="${ALL_LLM:-0}"
ALL_CLASSIFIER="${ALL_CLASSIFIER:-0}"
MODEL_ALIAS="${MODEL_ALIAS:-llama-8b-instruct}"
CLASSIFIER="${CLASSIFIER:-roberta}"

declare -A ALIAS_TO_MODEL=(
    ["llama-8b-instruct"]="meta-llama/Meta-Llama-3-8B-Instruct"
    ["qwen3-8b"]="Qwen/Qwen3-8B"
    ["llama-70b-instruct"]="meta-llama/Meta-Llama-3-70B-Instruct"
    ["qwen3-32b"]="Qwen/Qwen3-32B"
)

declare -A ALIAS_TO_GPU=(
    ["llama-8b-instruct"]="0"
    ["qwen3-8b"]="0"
    ["llama-70b-instruct"]="0,1"
    ["qwen3-32b"]="0,1"
)

declare -A ALIAS_TO_TP=(
    ["llama-8b-instruct"]="1"
    ["qwen3-8b"]="1"
    ["llama-70b-instruct"]="2"
    ["qwen3-32b"]="2"
)

# Per-model GPU memory utilization. Tune per your hardware.
declare -A ALIAS_TO_GPU_MEM=(
    ["llama-8b-instruct"]="0.90"
    ["qwen3-8b"]="0.90"
    ["qwen3-32b"]="0.90"
)

TARGET_ALIASES=()
if [ "$ALL_LLM" = "1" ]; then
    TARGET_ALIASES=("llama-8b-instruct" "qwen3-8b" "qwen3-32b")
else
    TARGET_ALIASES=("$MODEL_ALIAS")
fi

exec > >(tee -a "$SUMMARY_LOG") 2>&1

echo "=============================================="
echo " support_generation (paper) — vLLM + infer"
echo " start : $(date)"
echo " pkg   : $PKG_DIR"
echo " api   : $VLLM_BASE"
echo " data  : $DATA_DIR"
echo " out   : $OUTPUT_ROOT"
echo " ALL_LLM=$ALL_LLM  ALL_CLASSIFIER=$ALL_CLASSIFIER"
echo " MODEL_ALIAS=$MODEL_ALIAS  CLASSIFIER=$CLASSIFIER"
echo " log   : $SUMMARY_LOG"
echo "=============================================="

print_chat_content() {
    python3 - <<'PY'
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except json.JSONDecodeError:
    print(raw)
    raise SystemExit(0)
if d.get("error"):
    print("API error:", json.dumps(d["error"], ensure_ascii=False))
    raise SystemExit(0)
choices = d.get("choices") or []
if not choices:
    print(json.dumps(d, ensure_ascii=False, indent=2))
    raise SystemExit(0)
msg = (choices[0].get("message") or {})
c = msg.get("content")
print(c if c is not None else json.dumps(d, ensure_ascii=False, indent=2))
PY
}

cleanup_vllm_group() {
    local leader_pid="${1:-}"
    if [ -n "$leader_pid" ] && ps -p "$leader_pid" >/dev/null 2>&1; then
        kill -TERM -- "-$leader_pid" 2>/dev/null || true
        sleep 3
        kill -KILL -- "-$leader_pid" 2>/dev/null || true
        wait "$leader_pid" 2>/dev/null || true
    fi
}

run_proposed_infer() {
    local MODEL="$1"
    (
        cd "$PKG_DIR" || exit 1
        if [ "$ALL_CLASSIFIER" = "1" ]; then
            # shellcheck disable=SC2086
            "$PYTHON" infer_llm_proposed.py \
                --llm_id "$MODEL" \
                --all_classifier \
                --data_dir "$DATA_DIR" \
                --output_root "$OUTPUT_ROOT" \
                --base_url "$VLLM_BASE" \
                --temperature "$INFER_TEMPERATURE" \
                --max_tokens "$INFER_MAX_TOKENS" \
                --max_workers "$INFER_MAX_WORKERS" \
                --start "$INFER_START" \
                --end "$INFER_END" \
                ${INFER_EXTRA:-}
        else
            # shellcheck disable=SC2086
            "$PYTHON" infer_llm_proposed.py \
                --llm_id "$MODEL" \
                --classifier "$CLASSIFIER" \
                --data_dir "$DATA_DIR" \
                --output_root "$OUTPUT_ROOT" \
                --base_url "$VLLM_BASE" \
                --temperature "$INFER_TEMPERATURE" \
                --max_tokens "$INFER_MAX_TOKENS" \
                --max_workers "$INFER_MAX_WORKERS" \
                --start "$INFER_START" \
                --end "$INFER_END" \
                ${INFER_EXTRA:-}
        fi
    )
}

for ALIAS in "${TARGET_ALIASES[@]}"; do
    MODEL="${ALIAS_TO_MODEL[$ALIAS]:-}"
    CUDA_DEVICES="${ALIAS_TO_GPU[$ALIAS]:-0}"
    TP_SIZE="${ALIAS_TO_TP[$ALIAS]:-1}"
    GPU_MEM_FOR_MODEL="${ALIAS_TO_GPU_MEM[$ALIAS]:-$GPU_MEM}"

    if [ -z "$MODEL" ]; then
        echo "[error] unsupported MODEL_ALIAS: $ALIAS"
        continue
    fi

    SAFE_MODEL_NAME="${MODEL//\//_}"
    SERVER_LOG="${LOG_DIR}/vllm_server_${SAFE_MODEL_NAME}.log"

    echo ""
    echo "======== alias: $ALIAS ========"
    echo "  model     : $MODEL"
    echo "  GPU / TP  : $CUDA_DEVICES / $TP_SIZE  mem_util: $GPU_MEM_FOR_MODEL"
    echo "  server log: $SERVER_LOG"

    if command -v lsof >/dev/null 2>&1 && lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "  port $PORT already in use — killing"
        command -v fuser >/dev/null 2>&1 && fuser -k -9 "${PORT}/tcp" 2>/dev/null || true
        sleep 3
    fi

    echo "[vLLM] serve…"
    setsid bash -c "CUDA_VISIBLE_DEVICES=\"$CUDA_DEVICES\" vllm serve \"$MODEL\" --port \"$PORT\" --tensor-parallel-size \"$TP_SIZE\" --gpu-memory-utilization \"$GPU_MEM_FOR_MODEL\" --max-model-len \"$MAX_LEN\" --enforce-eager > \"$SERVER_LOG\" 2>&1" &
    SERVER_PID=$!

    echo "[vLLM] /health (max 500s)…"
    MAX_WAIT=500
    WAIT_COUNT=0
    SERVER_READY=false
    while [ "$WAIT_COUNT" -lt "$MAX_WAIT" ]; do
        code=$(curl -s -o /dev/null -w "%{http_code}" "$HOST/health" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            SERVER_READY=true
            echo "  ready (${WAIT_COUNT}s)"
            break
        fi
        if ! ps -p "$SERVER_PID" >/dev/null 2>&1; then
            echo "  server died — see $SERVER_LOG"
            break
        fi
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 2))
        echo -ne "  … ${WAIT_COUNT}s\r"
    done
    echo ""

    if [ "$SERVER_READY" != true ]; then
        cleanup_vllm_group "$SERVER_PID"
        sleep 5
        continue
    fi

    if [ "$RUN_CHAT_SAMPLE" = "1" ]; then
        echo "[sample] POST /v1/chat/completions"
        BODY="$(MODEL="$MODEL" python3 -c '
import json, os
print(json.dumps({"model": os.environ["MODEL"], "messages": [{"role": "user", "content": "ping"}], "max_tokens": 32}))
')"
        curl -s "$HOST/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY" | print_chat_content
    fi

    if [ "$RUN_INFER" = "1" ]; then
        echo "[infer] proposed (roberta/distilbert)"
        if run_proposed_infer "$MODEL"; then
            echo "[infer] proposed done"
        else
            echo "[infer] proposed FAILED (exit=$?)"
        fi
    fi

    echo "[stop] vLLM PID $SERVER_PID"
    cleanup_vllm_group "$SERVER_PID"
    sleep 5
done

echo ""
echo "end: $(date)"
