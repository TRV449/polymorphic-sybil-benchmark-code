#!/usr/bin/env bash
# =============================================================================
# CANONICAL: nq_hotpot_public500_v1 FiD external baseline re-run
# Reads ONLY frozen QC artifact (poison_docs_llm.frozen_qc.jsonl).
# Tracks: base / attack / forced
# =============================================================================
set -euo pipefail

ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PT="$ROOT/pilot_tools"
RT="$ROOT/member_runtime"

MANIFEST="$PT/benchmark_package/fixed_splits/public_all_500_balanced_seed42.json"
POISON_JSONL="$RT/poison_docs_llm.frozen_qc.jsonl"
BASE_INDEX="$ROOT/wiki_indexes/wikipedia-dpr-100w"
ATTACK_INDEX="$ROOT/wiki_indexes/wikipedia-dpr-100w_attack"
HOTPOT="$ROOT/datasets/hotpotqa_fixed.jsonl"
NQ="$ROOT/datasets/nq_fixed.jsonl"
FID_MODEL="$RT/nq_reader_large"

TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="${OUTPUT_CSV:-$RT/release_fid_500_${TS}.csv}"

echo "[release_fid] manifest: $MANIFEST"
echo "[release_fid] frozen artifact: $POISON_JSONL"
echo "[release_fid] fid model: $FID_MODEL"
echo "[release_fid] output: $OUTPUT_CSV"

cd "$PT/baselines/fid"
python3 fid_runner.py \
  --base_index "$BASE_INDEX" \
  --attack_index "$ATTACK_INDEX" \
  --hotpot "$HOTPOT" \
  --nq "$NQ" \
  --fid_model_path "$FID_MODEL" \
  --output_csv "$OUTPUT_CSV" \
  --frozen_poison_jsonl "$POISON_JSONL" \
  --manifest "$MANIFEST" \
  --tracks base attack forced \
  --recall_k 100 \
  --n_passages 100 \
  --forced_k 4 \
  --poison_per_query 6 \
  --overwrite_output \
  "$@"

echo "[release_fid] done → $OUTPUT_CSV"
