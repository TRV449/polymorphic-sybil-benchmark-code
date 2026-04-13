#!/usr/bin/env bash
# =============================================================================
# Qwen 72B: Verifier → Reader mode transition
# =============================================================================
# Stops the QC verifier instance (port 8003) and restarts as a reader
# with a larger context window for 10-passage RAG prompts.
#
# Changes:
#   - Context: 8192 → 16384
#   - Alias remains the same (qwen25_72b_instruct_gguf)
#   - max_tokens is set per-request by the runner (default 64)
#
# Usage:
#   bash pilot_tools/qwen72b_to_reader_mode.sh
# =============================================================================
ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
set -euo pipefail

PORT=8003
MODEL="$ROOT/member_runtime/models/Qwen2.5-72B-Instruct-GGUF/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
LLAMA_SERVER="$ROOT/member_runtime/llama.cpp-build-cu118/bin/llama-server"
ALIAS="qwen25_72b_instruct_gguf"
CTX=16384
LOG="$ROOT/member_runtime/qwen72b_reader_$(date +%Y%m%d_%H%M%S).log"

echo "=== Qwen 72B: Verifier → Reader transition ==="
echo ""

# ── Step 1: Find and stop the existing process ──────────────────────────────
OLD_PID=$(lsof -ti :${PORT} 2>/dev/null | head -1 || true)

if [ -n "$OLD_PID" ]; then
    echo "[1/4] Stopping verifier (PID $OLD_PID) on port $PORT..."
    kill "$OLD_PID"
    # Wait for clean shutdown (up to 30s)
    for i in $(seq 1 30); do
        if ! kill -0 "$OLD_PID" 2>/dev/null; then
            echo "  Process stopped after ${i}s."
            break
        fi
        sleep 1
    done
    # Force kill if still running
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "  Forcing kill..."
        kill -9 "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
else
    echo "[1/4] No process found on port $PORT."
fi

# ── Step 2: Verify port is free ─────────────────────────────────────────────
echo "[2/4] Verifying port $PORT is free..."
if lsof -ti :${PORT} >/dev/null 2>&1; then
    echo "  ERROR: Port $PORT still in use!"
    lsof -i :${PORT}
    exit 1
fi
echo "  Port $PORT is free."

# ── Step 3: Start reader instance ───────────────────────────────────────────
echo "[3/4] Starting reader instance (ctx=$CTX)..."
echo "  Model: $MODEL"
echo "  Log: $LOG"

nohup "$LLAMA_SERVER" \
    -m "$MODEL" \
    --alias "$ALIAS" \
    -c "$CTX" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --timeout 600 \
    -ngl 99 \
    > "$LOG" 2>&1 &

NEW_PID=$!
echo "  Started PID $NEW_PID"

# Wait for server to become ready (up to 120s for 72B model load)
echo "  Waiting for server health..."
for i in $(seq 1 120); do
    HEALTH=$(curl -s --connect-timeout 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo "")
    if echo "$HEALTH" | grep -q '"ok"'; then
        echo "  Server ready after ${i}s."
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "  ERROR: Server did not become healthy within 120s."
        echo "  Check log: $LOG"
        exit 1
    fi
    sleep 1
done

# ── Step 4: Smoke test ──────────────────────────────────────────────────────
echo "[4/4] Running smoke test (5 reader-style queries)..."

SMOKE_PASS=0
SMOKE_TOTAL=5

# Simple QA prompts to verify reader capability
QUESTIONS=(
    "What is the capital of France?"
    "Who wrote Romeo and Juliet?"
    "What year did World War II end?"
    "What is the chemical symbol for gold?"
    "Who painted the Mona Lisa?"
)
EXPECTED=(
    "Paris"
    "Shakespeare"
    "1945"
    "Au"
    "da Vinci|Leonardo"
)

for idx in "${!QUESTIONS[@]}"; do
    Q="${QUESTIONS[$idx]}"
    E="${EXPECTED[$idx]}"

    RESP=$(curl -s --connect-timeout 10 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$ALIAS\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Answer in one or two words only. $Q\"}],
            \"max_tokens\": 32,
            \"temperature\": 0.0
        }" 2>/dev/null || echo "")

    ANSWER=$(echo "$RESP" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d['choices'][0]['message']['content'].strip())
except:
    print('[ERROR]')
" 2>/dev/null)

    if echo "$ANSWER" | grep -iqE "$E"; then
        STATUS="PASS"
        SMOKE_PASS=$((SMOKE_PASS + 1))
    else
        STATUS="FAIL"
    fi
    echo "  Q${idx}: [$STATUS] \"$Q\" → \"$ANSWER\""
done

echo ""
echo "=== Smoke test: $SMOKE_PASS/$SMOKE_TOTAL passed ==="
echo ""

if [ "$SMOKE_PASS" -lt 3 ]; then
    echo "WARNING: Fewer than 3/5 smoke tests passed. Check server logs."
    echo "  Log: $LOG"
fi

echo "=== Qwen 72B Reader mode ready (port $PORT, ctx=$CTX) ==="
echo "  PID: $NEW_PID"
echo "  Log: $LOG"
