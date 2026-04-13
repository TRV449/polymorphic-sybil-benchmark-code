#!/usr/bin/env bash
# =============================================================================
# day4_overnight.sh — Overnight parallel dispatcher for Day 4 experiments.
#
# Launches all remaining reader×retriever combinations in parallel via tmux.
# Qwen 7B should already be running or complete before starting this.
#
# GPU allocation:
#   GPU #0: Qwen 7B (port 8001) + GPT-OSS 120B (port 8005) + ColBERT
#   GPU #1: Qwen 72B (port 8003)
#
# Parallelism strategy:
#   - GPT-OSS and Qwen 72B use different GPUs → can run BM25 in parallel
#   - GPT-4o-mini is API → no GPU constraint, runs alongside
#   - E5+CE tracks run after BM25 for each reader (sequential per reader)
#   - ColBERT reranker is on GPU #0, so GPU #0 readers share it
#
# Usage:
#   bash day4_overnight.sh [mode]
#
#   Modes:
#     status    — check what's running and what's done
#     launch    — start all remaining runs in tmux
#     bm25_only — start BM25 tracks only (faster first pass)
#     e5ce_only — start E5+CE tracks only
# =============================================================================
set -euo pipefail

ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PT="$ROOT/pilot_tools"
RT="$ROOT/member_runtime"
SCRIPT="$PT/day4_main_runs.sh"

export DAY4_MANIFEST="$ROOT/manifests/final_manifest_qc_2982_seed42.json"
export DAY4_POISON="$RT/frozen_release_full.jsonl"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(grep OPENAI_API_KEY ~/.bashrc 2>/dev/null | cut -d'"' -f2)}"

LOGDIR="$RT/logs_day4"
mkdir -p "$LOGDIR"

SESSION="day4"

# ── Status check ─────────────────────────────────────────────────────────────
do_status() {
    echo "============================================================"
    echo "  Day 4 Overnight Status"
    echo "============================================================"

    echo ""
    echo "--- Servers ---"
    for port in 8001 8003 8005; do
        resp=$(curl -s --connect-timeout 2 "http://127.0.0.1:${port}/health" 2>/dev/null || echo '{"status":"OFFLINE"}')
        echo "  Port $port: $resp"
    done

    echo ""
    echo "--- Running experiments ---"
    ps aux | grep -E "(colbert_qwen|faiss_qwen)" | grep -v grep | \
        sed 's/.*python3/python3/' | head -10 || echo "  (none)"

    echo ""
    echo "--- Completed CSVs ---"
    for reader in qwen7b gptoss120b qwen72b gpt4omini; do
        for ret in colbert e5ce; do
            for track in clean attack forced; do
                csv=$(ls "$RT"/day4_${reader}_${ret}_${track}_*.csv 2>/dev/null | tail -1)
                if [ -n "$csv" ] && [ -s "$csv" ]; then
                    n=$(($(wc -l < "$csv") - 1))
                    echo "  [DONE] ${reader}_${ret}_${track}: $n rows"
                fi
            done
        done
    done

    echo ""
    echo "--- Tmux sessions ---"
    tmux ls 2>/dev/null || echo "  (no tmux sessions)"
}

# ── Launch helpers ───────────────────────────────────────────────────────────
launch_in_tmux() {
    local window_name=$1
    local cmd=$2

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux new-window -t "$SESSION" -n "$window_name"
    else
        tmux new-session -d -s "$SESSION" -n "$window_name"
    fi

    tmux send-keys -t "$SESSION:$window_name" \
        "export DAY4_MANIFEST='$DAY4_MANIFEST'; export DAY4_POISON='$DAY4_POISON'; export OPENAI_API_KEY='$OPENAI_API_KEY'; $cmd" C-m
}

# ── BM25 tracks (all readers in parallel) ─────────────────────────────────────
do_bm25() {
    echo "=== Launching BM25 tracks ==="

    # GPT-OSS 120B × BM25 (GPU #0, port 8005)
    echo "  [1/3] GPT-OSS 120B × BM25 (3 tracks)"
    launch_in_tmux "gptoss_bm25" \
        "bash $SCRIPT gptoss_bm25 2>&1 | tee $LOGDIR/gptoss_bm25.log"

    # Qwen 72B × BM25 (GPU #1, port 8003)
    echo "  [2/3] Qwen 72B × BM25 (3 tracks)"
    launch_in_tmux "qwen72_bm25" \
        "bash $SCRIPT qwen72_bm25 2>&1 | tee $LOGDIR/qwen72_bm25.log"

    # GPT-4o-mini × BM25 (API, no GPU)
    echo "  [3/3] GPT-4o-mini × BM25 (3 tracks)"
    launch_in_tmux "gpt4omini_bm25" \
        "bash $SCRIPT gpt4omini_bm25 2>&1 | tee $LOGDIR/gpt4omini_bm25.log"

    echo ""
    echo "All BM25 tracks launched. Monitor with: tmux attach -t $SESSION"
}

# ── E5+CE tracks (all readers in parallel) ────────────────────────────────────
do_e5ce() {
    echo "=== Launching E5+CE tracks ==="

    # GPT-OSS 120B × E5+CE
    echo "  [1/3] GPT-OSS 120B × E5+CE (3 tracks)"
    launch_in_tmux "gptoss_e5ce" \
        "bash $SCRIPT gptoss_e5ce 2>&1 | tee $LOGDIR/gptoss_e5ce.log"

    # Qwen 72B × E5+CE
    echo "  [2/3] Qwen 72B × E5+CE (3 tracks)"
    launch_in_tmux "qwen72_e5ce" \
        "bash $SCRIPT qwen72_e5ce 2>&1 | tee $LOGDIR/qwen72_e5ce.log"

    # GPT-4o-mini × E5+CE
    echo "  [3/3] GPT-4o-mini × E5+CE (3 tracks)"
    launch_in_tmux "gpt4omini_e5ce" \
        "bash $SCRIPT gpt4omini_e5ce 2>&1 | tee $LOGDIR/gpt4omini_e5ce.log"

    echo ""
    echo "All E5+CE tracks launched. Monitor with: tmux attach -t $SESSION"
}

# ── Full launch ──────────────────────────────────────────────────────────────
do_launch() {
    echo "============================================================"
    echo "  Day 4 Overnight Launch"
    echo "  Manifest: $DAY4_MANIFEST"
    echo "  Poison:   $DAY4_POISON"
    echo "============================================================"

    # Verify servers
    for port in 8001 8003 8005; do
        resp=$(curl -s --connect-timeout 2 "http://127.0.0.1:${port}/health" 2>/dev/null || echo "OFFLINE")
        if echo "$resp" | grep -q "ok"; then
            echo "  Port $port: OK"
        else
            echo "  WARNING: Port $port not healthy: $resp"
        fi
    done

    # Verify API key
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "  WARNING: OPENAI_API_KEY not set. GPT-4o-mini runs will fail."
    else
        echo "  OPENAI_API_KEY: set (${#OPENAI_API_KEY} chars)"
    fi

    echo ""
    echo "Phase 1: BM25 tracks (3 readers × 3 tracks = 9 runs)"
    do_bm25

    echo ""
    echo "Phase 2: E5+CE tracks (3 readers × 3 tracks = 9 runs)"
    do_e5ce

    echo ""
    echo "============================================================"
    echo "  18 runs launched across 6 tmux windows"
    echo "  Total: 3 readers × 2 retrievers × 3 tracks"
    echo ""
    echo "  Monitor:  tmux attach -t $SESSION"
    echo "  Status:   bash $0 status"
    echo "  Analysis: bash $PT/day4_run_all_analysis.sh"
    echo "============================================================"
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
MODE="${1:-status}"

case "$MODE" in
    status)    do_status    ;;
    launch)    do_launch    ;;
    bm25_only) do_bm25      ;;
    e5ce_only) do_e5ce      ;;
    *)
        echo "Usage: $0 {status|launch|bm25_only|e5ce_only}"
        exit 1
        ;;
esac
