#!/usr/bin/env bash
# =============================================================================
# day4_run_all_analysis.sh — Run auto-analysis for all completed Day 4 readers.
#
# Checks which reader×retriever combinations have all 3 track CSVs,
# then runs day4_auto_analysis.py on each.
#
# Usage:
#   bash day4_run_all_analysis.sh [csv_dir]
# =============================================================================
set -euo pipefail

ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PT="$ROOT/pilot_tools"
CSV_DIR="${1:-$ROOT/member_runtime}"

READERS=(qwen7b gptoss120b qwen72b gpt4omini)
RETRIEVERS=(colbert e5ce)

echo "============================================================"
echo "  Day 4: Auto-Analysis for All Completed Runs"
echo "  CSV dir: $CSV_DIR"
echo "============================================================"

analyzed=0
skipped=0

for reader in "${READERS[@]}"; do
    for retriever in "${RETRIEVERS[@]}"; do
        # Check if at least clean track exists
        clean_csv=$(ls "$CSV_DIR"/day4_${reader}_${retriever}_clean_*.csv 2>/dev/null | tail -1)
        if [ -z "$clean_csv" ]; then
            echo "  [SKIP] $reader × $retriever — no clean CSV"
            skipped=$((skipped + 1))
            continue
        fi

        # Count available tracks
        available_tracks=()
        for track in clean attack forced; do
            csv=$(ls "$CSV_DIR"/day4_${reader}_${retriever}_${track}_*.csv 2>/dev/null | tail -1)
            if [ -n "$csv" ]; then
                available_tracks+=("$track")
            fi
        done

        echo ""
        echo "  [RUN] $reader × $retriever — tracks: ${available_tracks[*]}"

        python3 "$PT/day4_auto_analysis.py" \
            --reader_tag "$reader" \
            --retriever "$retriever" \
            --csv_dir "$CSV_DIR" \
            --tracks "${available_tracks[@]}" \
            --output_json "$CSV_DIR/analysis_${reader}_${retriever}.json"

        analyzed=$((analyzed + 1))
    done
done

echo ""
echo "============================================================"
echo "  Done: $analyzed analyzed, $skipped skipped"
echo "============================================================"

# Summary table if any analysis JSONs exist
if [ $analyzed -gt 0 ]; then
    echo ""
    echo "  Compact Summary (ACC / ASR / Abstain, all scope):"
    echo "  ────────────────────────────────────────────────────"
    python3 -c "
import json, glob, os
files = sorted(glob.glob('$CSV_DIR/analysis_*.json'))
for f in files:
    data = json.load(open(f))
    tag = f'{data[\"reader_tag\"]:>12} × {data[\"retriever\"]:<8}'
    tracks = data.get('tracks', {})
    parts = []
    for t in ['clean', 'attack', 'forced']:
        if t in tracks and 'all' in tracks[t]:
            m = tracks[t]['all']
            acc = m['ACC']['pt']*100
            asr = m['ASR']['pt']*100
            abst = m['Abstain']['pt']*100
            parts.append(f'{t}: ACC={acc:.1f} ASR={asr:.1f} Abs={abst:.1f}')
    print(f'  {tag}  ' + '  |  '.join(parts))
"
fi
