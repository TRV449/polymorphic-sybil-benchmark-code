#!/usr/bin/env bash
# =============================================================================
# CANONICAL: nq_hotpot_public500_v1 ColBERTv2+Qwen2.5 baseline
# Strategy: BM25 top-1000 → ColBERTv2 MaxSim rerank → top-10 → Qwen2.5
# Reads ONLY frozen QC artifact (poison_docs_llm.frozen_qc.jsonl).
# Tracks: clean | attack | forced   (set TRACK=clean|attack|forced)
# =============================================================================
set -euo pipefail

ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PT="$ROOT/pilot_tools"
RT="$ROOT/member_runtime"

MANIFEST="$PT/benchmark_package/fixed_splits/public_all_500_balanced_seed42.json"
POISON_JSONL="$RT/poison_docs_llm.frozen_qc.jsonl"
BASE_INDEX="$ROOT/wiki_indexes/wikipedia-dpr-100w"
HOTPOT="$ROOT/datasets/hotpotqa_fixed.jsonl"
NQ="$ROOT/datasets/nq_fixed.jsonl"

# ColBERTv2 checkpoint (HF hub ID or local path)
COLBERT_MODEL="${COLBERT_MODEL:-colbert-ir/colbertv2.0}"

TRACK="${TRACK:-attack}"
TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="${OUTPUT_CSV:-$RT/release_colbert_${TRACK}_500_${TS}.csv}"

echo "[colbert] track:          $TRACK"
echo "[colbert] manifest:       $MANIFEST"
echo "[colbert] frozen artifact: $POISON_JSONL"
echo "[colbert] colbert_model:  $COLBERT_MODEL"
echo "[colbert] base_index:     $BASE_INDEX"
echo "[colbert] output:         $OUTPUT_CSV"

cd "$PT/baselines/modern_rag"
python3 colbert_qwen_runner.py \
  --manifest            "$MANIFEST" \
  --frozen_poison_jsonl "$POISON_JSONL" \
  --track               "$TRACK" \
  --output_csv          "$OUTPUT_CSV" \
  --base_index          "$BASE_INDEX" \
  --colbert_model       "$COLBERT_MODEL" \
  --hotpot              "$HOTPOT" \
  --nq                  "$NQ" \
  --llm_backend         "${LLM_BACKEND:-llama_cpp_http}" \
  --llm_model_id        "${LLM_MODEL_ID:-qwen25_7b_instruct_gguf}" \
  --llm_base_url        "${LLM_BASE_URL:-http://127.0.0.1:8001}" \
  --llm_temperature     "${LLM_TEMPERATURE:-0.0}" \
  --bm25_k              1000 \
  --final_k             10 \
  --colbert_batch_size  32 \
  --poison_per_query    6 \
  --overwrite_output \
  "$@"

echo "[colbert] done → $OUTPUT_CSV"
