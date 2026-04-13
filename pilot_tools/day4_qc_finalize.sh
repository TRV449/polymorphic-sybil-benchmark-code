#!/usr/bin/env bash
# =============================================================================
# day4_qc_finalize.sh — Run after Main QC completes.
#
# Steps:
#   1. Verify QC output integrity
#   2. Report kept/dropped counts and DS balance
#   3. Generate final manifest from QC-passed questions
#   4. Update day4_main_runs.sh MANIFEST variable
#   5. Print launch-ready command
#
# Usage:
#   bash day4_qc_finalize.sh
# =============================================================================
set -euo pipefail

ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PT="$ROOT/pilot_tools"
RT="$ROOT/member_runtime"

QC_JSONL="$RT/frozen_release_full.jsonl"
QC_REPORT="$RT/qc_report_full.json"
QC_AUDIT="$RT/qc_audit_full.csv"
INPUT_MANIFEST="$RT/near_balanced_3145_seed42.json"

echo "============================================================"
echo "  Day 4 QC Finalization"
echo "============================================================"

# ── Step 1: Verify QC completion ────────────────────────────────────────────
echo ""
echo "--- Step 1: QC Output Verification ---"

if [ ! -s "$QC_JSONL" ]; then
    echo "ERROR: $QC_JSONL is empty or missing. QC may not be complete."
    echo "Check: ps -p $(pgrep -f poison_qc 2>/dev/null || echo '?') -o pid,etime"
    exit 1
fi

KEPT=$(wc -l < "$QC_JSONL")
echo "  frozen_release_full.jsonl:  $KEPT lines (kept groups)"
echo "  qc_report_full.json:       $(stat -c%s "$QC_REPORT" 2>/dev/null || echo '?') bytes"
echo "  qc_audit_full.csv:         $(wc -l < "$QC_AUDIT" 2>/dev/null || echo '?') lines"

# ── Step 2: Report from QC JSON ─────────────────────────────────────────────
echo ""
echo "--- Step 2: QC Report Summary ---"
python3 -c "
import json
report = json.load(open('$QC_REPORT'))
print(f'  Input poison groups:  {report[\"input_poison_groups\"]}')
print(f'  Kept poison groups:   {report[\"kept_poison_groups\"]}')
print(f'  Dropped:              {report[\"dropped_poison_groups\"]}')
print(f'  Keep rate:            {report[\"kept_poison_groups\"]/max(report[\"input_poison_groups\"],1)*100:.1f}%')
print(f'  tau_lex:              {report[\"tau_lex\"]}')
print(f'  Mean pairwise Jaccard:{report[\"mean_pairwise_jaccard\"]:.4f}')
print(f'  Max pairwise Jaccard: {report[\"max_pairwise_jaccard\"]:.4f}')
print(f'  Verifier model:       {report[\"verifier\"][\"model_id\"]}')
"

# ── Step 3: DS balance of kept questions ────────────────────────────────────
echo ""
echo "--- Step 3: DS Balance ---"
python3 -c "
import json
from collections import Counter

kept = []
with open('$QC_JSONL') as f:
    for line in f:
        kept.append(json.loads(line))

ds_counter = Counter()
for row in kept:
    ds = row.get('ds', 'unknown')
    ds_counter[ds] += 1

total = len(kept)
print(f'  Total kept: {total}')
for ds, count in sorted(ds_counter.items()):
    print(f'  {ds:10} {count:5d}  ({count/total*100:.1f}%)')

# Balance ratio
if len(ds_counter) >= 2:
    vals = list(ds_counter.values())
    ratio = min(vals) / max(vals)
    print(f'  Balance ratio (min/max): {ratio:.3f}')
"

# ── Step 4: Generate final manifest ─────────────────────────────────────────
echo ""
echo "--- Step 4: Generate Final Manifest ---"
FINAL_MANIFEST="$RT/final_manifest_qc_${KEPT}_seed42.json"

python3 -c "
import json, hashlib

# Read kept question keys from QC output
kept_keys = set()
with open('$QC_JSONL') as f:
    for line in f:
        row = json.loads(line)
        kept_keys.add((row['ds'], str(row['id'])))

# Read original manifest
with open('$INPUT_MANIFEST') as f:
    orig = json.load(f)

# Filter to only kept questions
orig_questions = orig.get('questions', orig.get('selected', []))
kept_questions = [q for q in orig_questions if (q['ds'], str(q['id'])) in kept_keys]

# Verify all kept questions found
found_keys = {(q['ds'], str(q['id'])) for q in kept_questions}
missing = kept_keys - found_keys
if missing:
    print(f'  WARNING: {len(missing)} kept keys not found in original manifest!')

# Build final manifest
manifest = {
    'name': f'final_qc_{len(kept_questions)}_seed42',
    'seed': 42,
    'n_questions': len(kept_questions),
    'source_manifest': '$INPUT_MANIFEST',
    'qc_artifact': '$QC_JSONL',
    'questions': kept_questions,
}

# Compute integrity hash
canon = json.dumps(kept_questions, separators=(',', ':'), sort_keys=True)
sha = hashlib.sha256(canon.encode()).hexdigest()
manifest['sha256'] = sha

with open('$FINAL_MANIFEST', 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'  Final manifest: $FINAL_MANIFEST')
print(f'  Questions:      {len(kept_questions)}')
print(f'  SHA-256:        {sha[:16]}...')
"

# ── Step 5: Print launch command ────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  QC FINALIZATION COMPLETE"
echo "============================================================"
echo ""
echo "  Final manifest: $FINAL_MANIFEST"
echo ""
echo "  Launch Day 4 runs:"
echo ""
echo "    # Option A: All 18 runs sequentially"
echo "    DAY4_MANIFEST=$FINAL_MANIFEST DAY4_POISON=$QC_JSONL \\"
echo "      bash $PT/day4_main_runs.sh all"
echo ""
echo "    # Option B: Individual sections"
echo "    export DAY4_MANIFEST=$FINAL_MANIFEST"
echo "    export DAY4_POISON=$QC_JSONL"
echo "    bash $PT/day4_main_runs.sh qwen7_bm25      # Phase 1"
echo "    bash $PT/day4_main_runs.sh gptoss_bm25     # Phase 2"
echo "    bash $PT/day4_main_runs.sh qwen72_bm25     # Phase 3 (after QC server freed)"
echo ""
echo "    # Run GPT-OSS and Qwen7 E5+CE in parallel (different GPUs):"
echo "    bash $PT/day4_main_runs.sh gptoss_e5ce &"
echo "    bash $PT/day4_main_runs.sh qwen7_e5ce"
echo ""
