#!/usr/bin/env python3
"""
Merge separate per-track ColBERT CSVs into one combined CSV,
then run paired transition analysis.

Usage:
  python3 merge_and_transition.py \
    --clean_csv  $RT/colbert_clean.csv  \
    --attack_csv $RT/colbert_attack.csv \
    --forced_csv $RT/colbert_forced.csv \
    --src_prefix colbert \
    --output_merged $RT/colbert_merged.csv \
    --output_json   $RT/colbert_transitions.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from official_eval import load_csv_rows
from transition_matrix import build_report, render_summary

SHARED_FIELDS = ["ds", "id", "question", "gold_answer", "poison_target"]
ANSWER_SUFFIXES = ["answer_raw", "answer_eval", "answer_final", "abstain", "conflict_flag"]


def merge_csvs(
    clean_path: str,
    attack_path: str,
    forced_path: str,
    src_prefix: str,
    output_path: str,
) -> list:
    clean_rows = {(r["ds"], r["id"]): r for r in load_csv_rows(clean_path)}
    attack_rows = {(r["ds"], r["id"]): r for r in load_csv_rows(attack_path)}
    forced_rows = {(r["ds"], r["id"]): r for r in load_csv_rows(forced_path)} if forced_path else {}

    tracks = ["clean", "attack"]
    if forced_path:
        tracks.append("forced")

    fieldnames = list(SHARED_FIELDS)
    for track in tracks:
        for sfx in ANSWER_SUFFIXES:
            fieldnames.append(f"{track}_{sfx}")

    merged = []
    for key in clean_rows:
        cr = clean_rows[key]
        ar = attack_rows.get(key)
        fr = forced_rows.get(key)
        if not ar:
            continue
        row = {f: cr.get(f, "") for f in SHARED_FIELDS}
        for sfx in ANSWER_SUFFIXES:
            row[f"clean_{sfx}"] = cr.get(f"{src_prefix}_{sfx}", "")
            row[f"attack_{sfx}"] = ar.get(f"{src_prefix}_{sfx}", "")
            if fr:
                row[f"forced_{sfx}"] = fr.get(f"{src_prefix}_{sfx}", "")
        merged.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_csv", required=True)
    parser.add_argument("--attack_csv", required=True)
    parser.add_argument("--forced_csv", default="")
    parser.add_argument("--src_prefix", default="colbert")
    parser.add_argument("--output_merged", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--bootstrap_n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merged_path = args.output_merged or args.clean_csv.replace("clean", "merged")
    merged = merge_csvs(
        args.clean_csv, args.attack_csv, args.forced_csv,
        args.src_prefix, merged_path,
    )
    print(f"[merge] {len(merged)} rows → {merged_path}")

    rows = load_csv_rows(merged_path)

    # clean→attack
    print("\n" + "=" * 70)
    print("[transition] clean → attack")
    report_atk = build_report(rows, "clean", "attack", None, args.bootstrap_n, args.seed)
    print(render_summary(report_atk, "clean", "attack", None))

    # clean→forced
    report_frc = None
    if args.forced_csv:
        print("\n" + "=" * 70)
        print("[transition] clean → forced")
        report_frc = build_report(rows, "clean", "forced", None, args.bootstrap_n, args.seed)
        print(render_summary(report_frc, "clean", "forced", None))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        result = {"clean_to_attack": report_atk}
        if report_frc:
            result["clean_to_forced"] = report_frc
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[*] saved → {out}")


if __name__ == "__main__":
    main()
