#!/bin/bash
# FAISS dense pilot: 3 tracks × 200Q
# Usage: bash run_faiss_pilot.sh [clean|attack|forced|all]
set -e

ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PT=$ROOT/pilot_tools
RT=$ROOT/member_runtime
TRACK=${1:-all}

MANIFEST=$RT/pilot_balanced_200_seed42.json   # 100 hotpot + 100 NQ
POISON=$PT/_not_used/poison_docs_llm.jsonl
FAISS_DIR=$ROOT/faiss_index_wikipedia_dpr_100w
BM25_DIR=$ROOT/wiki_indexes/wikipedia-dpr-100w
E5_MODEL=intfloat/multilingual-e5-large
LLM_URL=http://127.0.0.1:8001
LLM_MODEL=qwen25_7b_instruct_gguf
MAX_Q=0   # manifest already has 200Q (100 hotpot + 100 NQ)

run_track() {
    local track=$1
    local outcsv=$RT/faiss_dense_${track}_200_$(date +%Y%m%d_%H%M%S).csv
    echo "=== [dense/$track] START ==="
    python3 $PT/baselines/modern_rag/faiss_qwen_runner.py \
        --manifest            $MANIFEST \
        --frozen_poison_jsonl $POISON \
        --track               $track \
        --output_csv          $outcsv \
        --faiss_index         $FAISS_DIR \
        --base_index          $BM25_DIR \
        --e5_model            $E5_MODEL \
        --llm_backend         llama_cpp_http \
        --llm_model_id        $LLM_MODEL \
        --llm_base_url        $LLM_URL \
        --hotpot              $ROOT/datasets/hotpotqa_fixed.jsonl \
        --nq                  $ROOT/datasets/nq_fixed.jsonl \
        --max_questions       $MAX_Q \
        --final_k             10 \
        --faiss_k             1000 \
        --doc_chars           1024 \
        2>&1 | tee ${outcsv%.csv}.log
    echo "=== [dense/$track] DONE: $outcsv ==="
}

run_track_rerank() {
    local track=$1
    local outcsv=$RT/e5_colbert_${track}_200_$(date +%Y%m%d_%H%M%S).csv
    echo "=== [e5+colbert/$track] START ==="
    python3 $PT/baselines/modern_rag/faiss_qwen_runner.py \
        --manifest            $MANIFEST \
        --frozen_poison_jsonl $POISON \
        --track               $track \
        --output_csv          $outcsv \
        --faiss_index         $FAISS_DIR \
        --base_index          $BM25_DIR \
        --e5_model            $E5_MODEL \
        --llm_backend         llama_cpp_http \
        --llm_model_id        $LLM_MODEL \
        --llm_base_url        $LLM_URL \
        --hotpot              $ROOT/datasets/hotpotqa_fixed.jsonl \
        --nq                  $ROOT/datasets/nq_fixed.jsonl \
        --max_questions       $MAX_Q \
        --final_k             10 \
        --faiss_k             1000 \
        --doc_chars           1024 \
        --colbert_rerank \
        --colbert_model       colbert-ir/colbertv2.0 \
        2>&1 | tee ${outcsv%.csv}.log
    echo "=== [e5+colbert/$track] DONE: $outcsv ==="
}

run_track_ce() {
    local track=$1
    local outcsv=$RT/e5_ce_${track}_200_$(date +%Y%m%d_%H%M%S).csv
    echo "=== [e5+ce/$track] START ==="
    python3 $PT/baselines/modern_rag/faiss_qwen_runner.py \
        --manifest            $MANIFEST \
        --frozen_poison_jsonl $POISON \
        --track               $track \
        --output_csv          $outcsv \
        --faiss_index         $FAISS_DIR \
        --base_index          $BM25_DIR \
        --e5_model            $E5_MODEL \
        --llm_backend         llama_cpp_http \
        --llm_model_id        $LLM_MODEL \
        --llm_base_url        $LLM_URL \
        --hotpot              $ROOT/datasets/hotpotqa_fixed.jsonl \
        --nq                  $ROOT/datasets/nq_fixed.jsonl \
        --max_questions       $MAX_Q \
        --final_k             10 \
        --faiss_k             200 \
        --doc_chars           1024 \
        --cross_encoder_rerank \
        --ce_model            cross-encoder/ms-marco-MiniLM-L6-v2 \
        2>&1 | tee ${outcsv%.csv}.log
    echo "=== [e5+ce/$track] DONE: $outcsv ==="
}

if [ "$TRACK" = "all" ]    || [ "$TRACK" = "clean" ];   then run_track clean;   fi
if [ "$TRACK" = "all" ]    || [ "$TRACK" = "attack" ];  then run_track attack;  fi
if [ "$TRACK" = "all" ]    || [ "$TRACK" = "forced" ];  then run_track forced;  fi
if [ "$TRACK" = "rerank" ] || [ "$TRACK" = "rerank_clean" ];  then run_track_rerank clean;  fi
if [ "$TRACK" = "rerank" ] || [ "$TRACK" = "rerank_attack" ]; then run_track_rerank attack; fi
if [ "$TRACK" = "rerank" ] || [ "$TRACK" = "rerank_forced" ]; then run_track_rerank forced; fi
if [ "$TRACK" = "ce" ]     || [ "$TRACK" = "ce_clean" ];  then run_track_ce clean;  fi
if [ "$TRACK" = "ce" ]     || [ "$TRACK" = "ce_attack" ]; then run_track_ce attack; fi
if [ "$TRACK" = "ce" ]     || [ "$TRACK" = "ce_forced" ]; then run_track_ce forced; fi
