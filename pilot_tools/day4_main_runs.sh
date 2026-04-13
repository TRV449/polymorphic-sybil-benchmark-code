#!/usr/bin/env bash
# =============================================================================
# Day 4 Main Experiment Runs — BM25+ColBERT substrate
# =============================================================================
# Day 4 overnight: 3 local readers × BM25+ColBERT × 3 tracks = 9 runs
#                + GPT-4o-mini (API) × BM25+ColBERT × 3 tracks = 3 runs
#                                                          Total = 12 runs
#
# Readers:
#   - GPT-OSS 120B (port 8005, GPU #0, llama_cpp_chat, max_tokens=256)
#   - Qwen 72B     (port 8003, GPU #1, llama_cpp_http, max_tokens=64)
#   - GPT-4o-mini   (OpenAI API, max_tokens=64)
#
# Note: Qwen 7B already ran BM25+E5CE in Day 3 (full 2,982).
# E5+CE cross-retriever runs are Day 7.
#
# PREREQUISITE: QC complete, final manifest generated, Qwen 72B in reader mode.
#
# Usage:
#   bash day4_main_runs.sh <section>
#
#   Sections:
#     status        — check server health & GPU
#     gptoss_bm25   — GPT-OSS 120B × BM25 × 3 tracks (GPU #0)
#     qwen72_bm25   — Qwen 72B × BM25 × 3 tracks (GPU #1)
#     gpt4omini_bm25 — GPT-4o-mini × BM25 × 3 tracks (API)
#     all_parallel  — tmux: GPU0 + GPU1 + API in parallel
#     all           — all 12 runs sequentially
# =============================================================================
set -euo pipefail

ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PT="$ROOT/pilot_tools"
RT="$ROOT/member_runtime"

# ── Paths (QC complete — using final manifest + frozen release) ──────────────
MANIFEST="${DAY4_MANIFEST:-$ROOT/manifests/final_manifest_qc_2982_seed42.json}"
POISON_JSONL="${DAY4_POISON:-$RT/frozen_release_full.jsonl}"
BASE_INDEX="$ROOT/wiki_indexes/wikipedia-dpr-100w"
FAISS_DIR="$ROOT/faiss_index_wikipedia_dpr_100w"
E5_MODEL="intfloat/multilingual-e5-large"
CE_MODEL="cross-encoder/ms-marco-MiniLM-L6-v2"
COLBERT_MODEL="${COLBERT_MODEL:-colbert-ir/colbertv2.0}"
HOTPOT="$ROOT/datasets/hotpotqa_fixed.jsonl"
NQ="$ROOT/datasets/nq_fixed.jsonl"

TS=$(date +%Y%m%d_%H%M%S)

# ── Server health check ─────────────────────────────────────────────────────
check_server() {
    local name=$1 port=$2
    local resp
    resp=$(curl -s --connect-timeout 3 "http://127.0.0.1:${port}/health" 2>/dev/null || echo '{"status":"OFFLINE"}')
    echo "  $name (port $port): $resp"
}

do_status() {
    echo "=== Server Status ==="
    check_server "Qwen-7B"    8001
    check_server "Qwen-72B"   8003
    check_server "GPT-OSS-120B" 8005
    echo ""
    echo "=== GPU Memory ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader 2>/dev/null
    echo ""
    echo "=== Manifest ==="
    if [ -f "$MANIFEST" ]; then
        echo "  $MANIFEST ($(python3 -c "import json; d=json.load(open('$MANIFEST')); print(len(d.get('questions',d.get('selected',[]))))" 2>/dev/null || echo '?') questions)"
    else
        echo "  MANIFEST NOT FOUND: $MANIFEST"
    fi
    echo "=== Frozen Poison ==="
    if [ -f "$POISON_JSONL" ]; then
        echo "  $POISON_JSONL ($(wc -l < "$POISON_JSONL") lines)"
    else
        echo "  POISON ARTIFACT NOT FOUND: $POISON_JSONL"
    fi
}

# ── BM25+ColBERT runner ─────────────────────────────────────────────────────
run_colbert() {
    local reader_name=$1 llm_backend=$2 llm_model=$3 llm_url=$4 track=$5
    local extra_env="${6:-}"
    local tag="${reader_name}_colbert_${track}"
    local outcsv="$RT/day4_${tag}_${TS}.csv"
    local logfile="$RT/day4_${tag}_${TS}.log"

    echo ""
    echo "================================================================="
    echo "  [BM25+ColBERT] reader=$reader_name track=$track"
    echo "  output: $outcsv"
    echo "================================================================="

    cd "$PT/baselines/modern_rag"
    env $extra_env \
    python3 colbert_qwen_runner.py \
        --manifest            "$MANIFEST" \
        --frozen_poison_jsonl "$POISON_JSONL" \
        --track               "$track" \
        --output_csv          "$outcsv" \
        --base_index          "$BASE_INDEX" \
        --colbert_model       "$COLBERT_MODEL" \
        --hotpot              "$HOTPOT" \
        --nq                  "$NQ" \
        --llm_backend         "$llm_backend" \
        --llm_model_id        "$llm_model" \
        --llm_base_url        "$llm_url" \
        --llm_temperature     0.0 \
        --bm25_k              1000 \
        --final_k             10 \
        --colbert_batch_size  32 \
        --poison_per_query    6 \
        --overwrite_output \
        2>&1 | tee "$logfile"

    echo "[done] $outcsv"
}

# ── E5+CE runner ─────────────────────────────────────────────────────────────
run_e5ce() {
    local reader_name=$1 llm_backend=$2 llm_model=$3 llm_url=$4 track=$5
    local extra_env="${6:-}"
    local tag="${reader_name}_e5ce_${track}"
    local outcsv="$RT/day4_${tag}_${TS}.csv"
    local logfile="$RT/day4_${tag}_${TS}.log"

    echo ""
    echo "================================================================="
    echo "  [E5+CE] reader=$reader_name track=$track"
    echo "  output: $outcsv"
    echo "================================================================="

    cd "$PT/baselines/modern_rag"
    env $extra_env \
    python3 faiss_qwen_runner.py \
        --manifest            "$MANIFEST" \
        --frozen_poison_jsonl "$POISON_JSONL" \
        --track               "$track" \
        --output_csv          "$outcsv" \
        --faiss_index         "$FAISS_DIR" \
        --base_index          "$BASE_INDEX" \
        --e5_model            "$E5_MODEL" \
        --llm_backend         "$llm_backend" \
        --llm_model_id        "$llm_model" \
        --llm_base_url        "$llm_url" \
        --llm_temperature     0.0 \
        --hotpot              "$HOTPOT" \
        --nq                  "$NQ" \
        --max_questions       0 \
        --final_k             10 \
        --faiss_k             200 \
        --doc_chars           1024 \
        --cross_encoder_rerank \
        --ce_model            "$CE_MODEL" \
        --overwrite_output \
        2>&1 | tee "$logfile"

    echo "[done] $outcsv"
}

# ── Reader configurations ────────────────────────────────────────────────────
# Qwen 7B: default max_tokens=64, llama_cpp_http (text completion)
run_qwen7_bm25() {
    for track in clean attack forced; do
        run_colbert "qwen7b" "llama_cpp_http" "qwen25_7b_instruct_gguf" \
                    "http://127.0.0.1:8001" "$track"
    done
}

run_qwen7_e5ce() {
    for track in clean attack forced; do
        run_e5ce "qwen7b" "llama_cpp_http" "qwen25_7b_instruct_gguf" \
                 "http://127.0.0.1:8001" "$track"
    done
}

# GPT-OSS 120B: max_tokens=256 (Harmony format needs room), llama_cpp_chat
run_gptoss_bm25() {
    for track in clean attack forced; do
        run_colbert "gptoss120b" "llama_cpp_chat" "gpt_oss_120b_gguf" \
                    "http://127.0.0.1:8005" "$track" \
                    "LLM_MAX_TOKENS=256"
    done
}

run_gptoss_e5ce() {
    for track in clean attack forced; do
        run_e5ce "gptoss120b" "llama_cpp_chat" "gpt_oss_120b_gguf" \
                 "http://127.0.0.1:8005" "$track" \
                 "LLM_MAX_TOKENS=256"
    done
}

# Qwen 72B: default max_tokens=64, llama_cpp_http
# QC complete — port 8003 free for reader use.
run_qwen72_bm25() {
    for track in clean attack forced; do
        run_colbert "qwen72b" "llama_cpp_http" "qwen25_72b_instruct_gguf" \
                    "http://127.0.0.1:8003" "$track"
    done
}

run_qwen72_e5ce() {
    for track in clean attack forced; do
        run_e5ce "qwen72b" "llama_cpp_http" "qwen25_72b_instruct_gguf" \
                 "http://127.0.0.1:8003" "$track"
    done
}

# GPT-4o-mini: OpenAI API, max_tokens=64 (standard short-answer)
# Rate limit: check account tier. Default ~30K TPM for tier 1.
run_gpt4omini_bm25() {
    for track in clean attack forced; do
        run_colbert "gpt4omini" "openai" "gpt-4o-mini" \
                    "" "$track"
    done
}

run_gpt4omini_e5ce() {
    for track in clean attack forced; do
        run_e5ce "gpt4omini" "openai" "gpt-4o-mini" \
                 "" "$track"
    done
}

# FiD (Fusion-in-Decoder): separate runner, BM25 retrieval only
# Requires: fid_model_path with pretrained checkpoint
FID_MODEL="${FID_MODEL:-$ROOT/member_runtime/fid_nq_large}"
FID_RUNNER="$PT/baselines/fid/fid_runner.py"

run_fid() {
    local tag="fid"
    local outcsv="$RT/day4_${tag}_${TS}.csv"
    local logfile="$RT/day4_${tag}_${TS}.log"

    if [ ! -d "$FID_MODEL" ]; then
        echo "[SKIP] FiD model not found: $FID_MODEL"
        echo "  Download with: python -c \"from transformers import AutoModel; AutoModel.from_pretrained('facebook/fid-nq-large')\""
        return 1
    fi

    echo ""
    echo "================================================================="
    echo "  [FiD] tracks=base,attack,forced"
    echo "  output: $outcsv"
    echo "================================================================="

    cd "$PT/baselines/fid"
    python3 fid_runner.py \
        --manifest            "$MANIFEST" \
        --frozen_poison_jsonl "$POISON_JSONL" \
        --base_index          "$BASE_INDEX" \
        --hotpot              "$HOTPOT" \
        --nq                  "$NQ" \
        --fid_model_path      "$FID_MODEL" \
        --output_csv          "$outcsv" \
        --tracks              base attack forced \
        --overwrite_output \
        2>&1 | tee "$logfile"

    echo "[done] $outcsv"
}

# ── Section dispatch ─────────────────────────────────────────────────────────
SECTION="${1:-status}"

case "$SECTION" in
    status)
        do_status
        ;;
    qwen7_bm25)      run_qwen7_bm25     ;;
    qwen7_e5ce)      run_qwen7_e5ce     ;;
    gptoss_bm25)     run_gptoss_bm25    ;;
    gptoss_e5ce)     run_gptoss_e5ce    ;;
    qwen72_bm25)     run_qwen72_bm25    ;;
    qwen72_e5ce)     run_qwen72_e5ce    ;;
    gpt4omini_bm25)  run_gpt4omini_bm25 ;;
    gpt4omini_e5ce)  run_gpt4omini_e5ce ;;
    fid)             run_fid            ;;
    all_local)
        echo "=== Day 4: Local models (24 runs) ==="
        echo "=== Phase 1: Qwen 7B ==="
        run_qwen7_bm25
        run_qwen7_e5ce
        echo "=== Phase 2: GPT-OSS 120B ==="
        run_gptoss_bm25
        run_gptoss_e5ce
        echo "=== Phase 3: Qwen 72B ==="
        run_qwen72_bm25
        run_qwen72_e5ce
        echo "=== Day 4 local COMPLETE ==="
        ;;
    all_api)
        echo "=== Day 4: API models ==="
        run_gpt4omini_bm25
        run_gpt4omini_e5ce
        echo "=== Day 4 API COMPLETE ==="
        ;;
    all_parallel)
        # Day 4 overnight: 3 tmux panes running in parallel
        #   - gpu0: GPT-OSS 120B × BM25 × 3 tracks
        #   - gpu1: Qwen 72B × BM25 × 3 tracks
        #   - api:  GPT-4o-mini × BM25 × 3 tracks
        SESSION="day4_overnight"
        SCRIPT_PATH="$(readlink -f "$0")"

        # Kill existing session if any
        tmux kill-session -t "$SESSION" 2>/dev/null || true

        echo "=== Launching Day 4 overnight in tmux session: $SESSION ==="
        echo "  gpu0: GPT-OSS 120B × BM25 × {clean,attack,forced}"
        echo "  gpu1: Qwen 72B     × BM25 × {clean,attack,forced}"
        echo "  api:  GPT-4o-mini  × BM25 × {clean,attack,forced}"
        echo ""

        tmux new-session -d -s "$SESSION" -n "gpu0" \
            "bash '$SCRIPT_PATH' gptoss_bm25; echo '=== GPU0 DONE ==='; read"

        tmux new-window -t "$SESSION" -n "gpu1" \
            "bash '$SCRIPT_PATH' qwen72_bm25; echo '=== GPU1 DONE ==='; read"

        tmux new-window -t "$SESSION" -n "api" \
            "bash '$SCRIPT_PATH' gpt4omini_bm25; echo '=== API DONE ==='; read"

        echo "tmux session '$SESSION' started with 3 windows."
        echo "  Attach: tmux attach -t $SESSION"
        echo "  List:   tmux list-windows -t $SESSION"
        echo ""
        tmux list-windows -t "$SESSION"
        ;;
    all)
        echo "=== Day 4: Full sequential execution (12 runs) ==="
        echo "=== Phase 1: GPT-OSS 120B × BM25 ==="
        run_gptoss_bm25
        echo "=== Phase 2: Qwen 72B × BM25 ==="
        run_qwen72_bm25
        echo "=== Phase 3: GPT-4o-mini × BM25 (API) ==="
        run_gpt4omini_bm25
        echo "=== Day 4 COMPLETE ==="
        ;;
    *)
        echo "Usage: $0 {status|gptoss_bm25|qwen72_bm25|gpt4omini_bm25|all_parallel|all}"
        echo ""
        echo "Day 4 sections (BM25+ColBERT only):"
        echo "  gptoss_bm25     GPT-OSS 120B × 3 tracks (GPU #0)"
        echo "  qwen72_bm25     Qwen 72B × 3 tracks (GPU #1)"
        echo "  gpt4omini_bm25  GPT-4o-mini × 3 tracks (API)"
        echo "  all_parallel    tmux: all 3 in parallel (overnight)"
        echo "  all             all 12 runs sequentially"
        echo ""
        echo "Legacy / Day 7 sections:"
        echo "  qwen7_bm25|qwen7_e5ce|gptoss_e5ce|qwen72_e5ce|gpt4omini_e5ce|fid|all_local|all_api"
        exit 1
        ;;
esac
