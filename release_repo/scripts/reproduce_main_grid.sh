#!/usr/bin/env bash
# =============================================================================
# validation_reader_runs.sh — M7 (TriviaQA) + M8 (2Wiki) reader grid.
#
# Usage:
#   bash validation_reader_runs.sh --reader qwen72b --port 8004 --backend llama_cpp_http
#   bash validation_reader_runs.sh --reader llama70b --port 8007 --backend llama_cpp_http
#   bash validation_reader_runs.sh --reader gpt4omini --port "" --backend openai
#   bash validation_reader_runs.sh --reader qwen7b --port 8006 --backend llama_cpp_http
#
# Runs: 2 datasets × 2 retrievers × 3 tracks = 12 cells per reader invocation.
# =============================================================================
set -euo pipefail
ROOT=/mnt/data/2020112002
PT=$ROOT/pilot_tools
RT=$ROOT/member_runtime
TS=$(date +%Y%m%d_%H%M%S)

# ── Parse args ───────────────────────────────────────────────────────────────
READER=""
PORT=""
BACKEND=""
MODEL_ID=""
EXTRA_ENV=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --reader)  READER="$2"; shift 2;;
        --port)    PORT="$2"; shift 2;;
        --backend) BACKEND="$2"; shift 2;;
        --model_id) MODEL_ID="$2"; shift 2;;
        --extra_env) EXTRA_ENV="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# ── Reader defaults ──────────────────────────────────────────────────────────
case "$READER" in
    qwen72b)
        MODEL_ID="${MODEL_ID:-qwen25_72b_instruct_gguf}"
        BACKEND="${BACKEND:-llama_cpp_http}"
        PORT="${PORT:-8004}"
        ;;
    llama70b)
        MODEL_ID="${MODEL_ID:-llama-3.1-70b}"
        BACKEND="${BACKEND:-llama_cpp_http}"
        PORT="${PORT:-8007}"
        ;;
    qwen7b)
        MODEL_ID="${MODEL_ID:-qwen25_7b_instruct_gguf}"
        BACKEND="${BACKEND:-llama_cpp_http}"
        PORT="${PORT:-8006}"
        ;;
    gpt4omini)
        MODEL_ID="${MODEL_ID:-gpt-4o-mini}"
        BACKEND="${BACKEND:-openai}"
        PORT=""
        ;;
    gptoss120b)
        MODEL_ID="${MODEL_ID:-gpt_oss_120b_gguf}"
        BACKEND="${BACKEND:-llama_cpp_chat}"
        PORT="${PORT:-8002}"
        ;;
    *) echo "Unknown reader: $READER"; exit 1;;
esac

if [ "$BACKEND" = "openai" ]; then
    LLM_URL=""
    LLM_URL_ARG=""  # argparse default now "" → openai client uses api.openai.com
else
    LLM_URL="http://127.0.0.1:${PORT}"
    LLM_URL_ARG="--llm_base_url $LLM_URL"
fi

# ── Dataset configs ──────────────────────────────────────────────────────────
# TriviaQA (M7)
TRIVIA_MANIFEST=$RT/trivia_2k_accepted_manifest.json
TRIVIA_POISON=$RT/triviaqa_2k_poison.frozen_qc.jsonl
TRIVIA_GOLD=$RT/triviaqa_2k_target_qs.jsonl

# 2Wiki (M8)
WIKI2_MANIFEST=$RT/wiki2_3k_accepted_manifest.json
WIKI2_POISON=$RT/poison_docs_2wiki_v2_3k.frozen_qc.jsonl
WIKI2_GOLD=$RT/2wiki_validation_3000.jsonl

# Shared infra
BASE_INDEX=$ROOT/wiki_indexes/wikipedia-dpr-100w
FAISS_DIR=$ROOT/faiss_index_wikipedia_dpr_100w
E5_MODEL=intfloat/multilingual-e5-large
CE_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
COLBERT_MODEL=colbert-ir/colbertv2.0

LOG=$RT/logs; mkdir -p "$LOG"

# ── Create 2Wiki manifest if missing ────────────────────────────────────────
if [ ! -f "$WIKI2_MANIFEST" ]; then
    python3 -c "
import json
qs = [{'ds':'wiki2','id':json.loads(l)['id']} for l in open('$WIKI2_GOLD')]
json.dump({'name':'wiki2_3k','seed':42,'n_questions':len(qs),
           'source':'2wiki_validation_3000.jsonl','note':'2Wiki 3k validation, tau_lex=0.8',
           'questions':qs}, open('$WIKI2_MANIFEST','w'))
print(f'[*] wrote {len(qs)} qids to $WIKI2_MANIFEST')
"
fi

# ── Health check (skip for openai) ───────────────────────────────────────────
if [ -n "$PORT" ]; then
    curl -s -m 5 "http://127.0.0.1:${PORT}/health" | grep -q '"ok"' || {
        echo "[!] Reader $READER on port $PORT unhealthy"; exit 1
    }
fi

# ── Run function ─────────────────────────────────────────────────────────────
run_e5ce() {
    local dataset=$1 manifest=$2 poison=$3 gold_flag=$4 gold_path=$5 track=$6
    local tag="val_${READER}_e5ce_${dataset}_${track}"
    local outcsv="$RT/${tag}_${TS}.csv"
    local logf="$LOG/${tag}_${TS}.log"
    echo ""; echo "===== [E5+CE] $READER × $dataset × $track → $outcsv ====="
    cd "$PT/baselines/modern_rag"
    env $EXTRA_ENV \
    python3 faiss_qwen_runner.py \
        --manifest            "$manifest" \
        --frozen_poison_jsonl "$poison" \
        --track               "$track" \
        --output_csv          "$outcsv" \
        --faiss_index         "$FAISS_DIR" \
        --base_index          "$BASE_INDEX" \
        --e5_model            "$E5_MODEL" \
        --llm_backend         "$BACKEND" \
        --llm_model_id        "$MODEL_ID" \
        $LLM_URL_ARG \
        --llm_temperature     0.0 \
        $gold_flag "$gold_path" \
        --max_questions       0 \
        --final_k             10 \
        --faiss_k             200 \
        --doc_chars           1024 \
        --cross_encoder_rerank \
        --ce_model            "$CE_MODEL" \
        --overwrite_output \
        2>&1 | tee "$logf"
    echo "[done] $outcsv ($(wc -l < "$outcsv") rows)"
}

run_colbert() {
    local dataset=$1 manifest=$2 poison=$3 gold_flag=$4 gold_path=$5 track=$6
    local tag="val_${READER}_colbert_${dataset}_${track}"
    local outcsv="$RT/${tag}_${TS}.csv"
    local logf="$LOG/${tag}_${TS}.log"
    echo ""; echo "===== [ColBERT] $READER × $dataset × $track → $outcsv ====="
    cd "$PT/baselines/modern_rag"
    env $EXTRA_ENV \
    python3 colbert_qwen_runner.py \
        --manifest            "$manifest" \
        --frozen_poison_jsonl "$poison" \
        --track               "$track" \
        --output_csv          "$outcsv" \
        --base_index          "$BASE_INDEX" \
        --colbert_model       "$COLBERT_MODEL" \
        --llm_backend         "$BACKEND" \
        --llm_model_id        "$MODEL_ID" \
        $LLM_URL_ARG \
        --llm_temperature     0.0 \
        $gold_flag "$gold_path" \
        --bm25_k              1000 \
        --final_k             10 \
        --colbert_batch_size  32 \
        --poison_per_query    6 \
        --overwrite_output \
        2>&1 | tee "$logf"
    echo "[done] $outcsv ($(wc -l < "$outcsv") rows)"
}

# ── Execute grid ─────────────────────────────────────────────────────────────
echo "======================================================================"
echo "  Reader: $READER ($BACKEND @ $PORT)  TS=$TS"
echo "  Datasets: TriviaQA (1398Q) + 2Wiki (2696Q)"
echo "  Retrievers: E5+CE + ColBERT"
echo "  Tracks: clean, attack, forced"
echo "  Total: 12 cells"
echo "======================================================================"

for track in clean attack forced; do
    # TriviaQA × E5+CE
    run_e5ce  "trivia" "$TRIVIA_MANIFEST" "$TRIVIA_POISON" "--trivia" "$TRIVIA_GOLD" "$track"
    # TriviaQA × ColBERT
    run_colbert "trivia" "$TRIVIA_MANIFEST" "$TRIVIA_POISON" "--trivia" "$TRIVIA_GOLD" "$track"
    # 2Wiki × E5+CE
    run_e5ce  "wiki2" "$WIKI2_MANIFEST" "$WIKI2_POISON" "--wiki2" "$WIKI2_GOLD" "$track"
    # 2Wiki × ColBERT
    run_colbert "wiki2" "$WIKI2_MANIFEST" "$WIKI2_POISON" "--wiki2" "$WIKI2_GOLD" "$track"
done

echo "======================================================================"
echo "  ALL 12 cells complete for $READER at $(date)"
echo "======================================================================"
