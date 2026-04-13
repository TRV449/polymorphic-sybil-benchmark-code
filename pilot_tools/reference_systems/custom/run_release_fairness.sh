#!/usr/bin/env bash
# =============================================================================
# CANONICAL: nq_hotpot_public500_v1 fairness-track re-run
# Reads ONLY frozen QC artifact (poison_docs_llm.frozen_qc.jsonl).
# Tracks: base / attack / v25-defense
# =============================================================================
set -euo pipefail

ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PT="$ROOT/pilot_tools"
RT="$ROOT/member_runtime"

# --- Release-locked paths (DO NOT CHANGE without updating release_spec) ---
MANIFEST="$PT/benchmark_package/fixed_splits/public_all_500_balanced_seed42.json"
POISON_JSONL="$RT/poison_docs_llm.frozen_qc.jsonl"
BASE_INDEX="$ROOT/wiki_indexes/wikipedia-dpr-100w"
ATTACK_INDEX="$ROOT/wiki_indexes/wikipedia-dpr-100w_attack"
HOTPOT="$ROOT/datasets/hotpotqa_fixed.jsonl"
NQ="$ROOT/datasets/nq_fixed.jsonl"

TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="${OUTPUT_CSV:-$RT/release_fairness_7b_500_${TS}.csv}"

echo "[release_fairness] manifest: $MANIFEST"
echo "[release_fairness] frozen artifact: $POISON_JSONL"
echo "[release_fairness] output: $OUTPUT_CSV"

cd "$PT"
python3 fullwiki_eval_pipeline_v24.py \
  --base_index "$BASE_INDEX" \
  --attack_index "$ATTACK_INDEX" \
  --hotpot "$HOTPOT" \
  --nq "$NQ" \
  --output_csv "$OUTPUT_CSV" \
  --llm_backend "${LLM_BACKEND:-llama_cpp_http}" \
  --llm_model_id "${LLM_MODEL_ID:-qwen25_7b_instruct_gguf}" \
  --llm_base_url "${LLM_BASE_URL:-http://127.0.0.1:8001}" \
  --llm_temperature "${LLM_TEMPERATURE:-0.0}" \
  --closed_set_rerank \
  --poison_jsonl "$POISON_JSONL" \
  --question_manifest "$MANIFEST" \
  --candidate_k 50 \
  --final_k 10 \
  --recall_k 40 \
  --base_final_k 6 \
  --poison_per_query 6 \
  --min_support_clusters 2 \
  --min_support_pairs 2 \
  --min_union_clusters 3 \
  --min_support_margin 0.15 \
  --max_hotpot_units 20 \
  --overwrite_output \
  "$@"

echo "[release_fairness] done → $OUTPUT_CSV"
