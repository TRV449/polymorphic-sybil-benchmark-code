#!/usr/bin/env bash
# =============================================================================
# CANONICAL: nq_hotpot_public500_v1 forced-exposure oracle re-run
# Reads ONLY frozen QC artifact (poison_docs_llm.frozen_qc.jsonl).
# Track: forced_exposure_gold_fixed
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

TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="${OUTPUT_CSV:-$RT/release_oracle_7b_500_${TS}.csv}"

echo "[release_oracle] manifest: $MANIFEST"
echo "[release_oracle] frozen artifact: $POISON_JSONL"
echo "[release_oracle] output: $OUTPUT_CSV"

cd "$PT"
python3 fullwiki_eval_oracle_controls.py \
  --base_index "$BASE_INDEX" \
  --hotpot "$HOTPOT" \
  --nq "$NQ" \
  --poison_jsonl "$POISON_JSONL" \
  --output_csv "$OUTPUT_CSV" \
  --track forced_exposure_gold_fixed \
  --question_manifest "$MANIFEST" \
  --llm_backend "${LLM_BACKEND:-llama_cpp_http}" \
  --llm_model_id "${LLM_MODEL_ID:-qwen25_7b_instruct_gguf}" \
  --llm_base_url "${LLM_BASE_URL:-http://127.0.0.1:8001}" \
  --llm_temperature "${LLM_TEMPERATURE:-0.0}" \
  "$@"

echo "[release_oracle] done → $OUTPUT_CSV"
