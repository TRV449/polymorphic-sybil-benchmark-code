#!/usr/bin/env python3
"""
run_drift_audit.py — Drift origin audit pipeline (Appendix C.1).

Samples drift instances from Day 4 experiment CSVs, classifies them
with an LLM into 3 categories (GENUINE / EXTRACTION_ARTIFACT / DATASET_ISSUE),
and produces an audit JSON for author validation.

Sampling strategy (per reader x retriever):
  - 100 clean-condition drift instances
  - 50 paired clean-gold -> attack-drift transitions

Usage:
  python3 run_drift_audit.py \\
      --reader_tag qwen7b \\
      --retriever colbert \\
      --csv_dir ./member_runtime \\
      --classifier_backend llama_cpp_http \\
      --classifier_url http://127.0.0.1:8003 \\
      --classifier_model qwen25_72b_instruct_gguf \\
      --output audit_qwen7b_colbert.json

  # Sample only (no LLM classification, for review):
  python3 run_drift_audit.py \\
      --reader_tag qwen7b \\
      --retriever colbert \\
      --csv_dir ./member_runtime \\
      --sample_only \\
      --output audit_qwen7b_colbert_samples.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, ".."))
sys.path.insert(0, os.path.join(_ROOT, "..", "benchmark_core"))

from inference_backends import LLMBackendConfig
from schema_validator import validate_csv, backfill_missing_fields
from drift_audit_classifier import (
    classify_drift_batch,
    CATEGORY_DESCRIPTIONS,
    DRIFT_AUDIT_PROMPT_HASH,
)
from sample_drift_for_audit import (
    label_df,
    sample_clean_drift,
    sample_paired_drift,
)

RETRIEVER_PREFIX = {
    "colbert": "colbert",
    "e5ce": "dense",
}


def find_csv(csv_dir: str, reader_tag: str, retriever: str, track: str) -> str | None:
    pattern = os.path.join(csv_dir, f"day4_{reader_tag}_{retriever}_{track}_*.csv")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Drift origin audit (Appendix C.1)")
    parser.add_argument("--reader_tag", required=True)
    parser.add_argument("--retriever", required=True, choices=["colbert", "e5ce"])
    parser.add_argument("--csv_dir", default="./member_runtime")
    parser.add_argument("--n_clean", type=int, default=100)
    parser.add_argument("--n_paired", type=int, default=50)
    parser.add_argument("--classifier_backend", default="llama_cpp_http")
    parser.add_argument("--classifier_url", default="http://127.0.0.1:8003")
    parser.add_argument("--classifier_model", default="qwen25_72b_instruct_gguf")
    parser.add_argument("--classifier_temperature", type=float, default=0.0)
    parser.add_argument("--classifier_max_tokens", type=int, default=16)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample_only", action="store_true",
                        help="Only sample drift instances, skip LLM classification")
    args = parser.parse_args()

    prefix = RETRIEVER_PREFIX[args.retriever]

    # Find CSVs
    clean_path = find_csv(args.csv_dir, args.reader_tag, args.retriever, "clean")
    attack_path = find_csv(args.csv_dir, args.reader_tag, args.retriever, "attack")

    if not clean_path:
        print(f"ERROR: clean CSV not found for {args.reader_tag} x {args.retriever}")
        sys.exit(1)

    print(f"=== Drift Audit: {args.reader_tag} x {args.retriever} ===")
    print(f"  Clean:  {os.path.basename(clean_path)}")

    clean_df = pd.read_csv(clean_path)
    is_valid, missing = validate_csv(clean_df, prefix, strict=False)
    if not is_valid:
        print(f"  [SCHEMA] clean: backfilling {len(missing)} missing fields: {missing}")
        clean_df = backfill_missing_fields(clean_df, prefix)
    clean_df = label_df(clean_df, prefix)

    attack_df = None
    if attack_path:
        print(f"  Attack: {os.path.basename(attack_path)}")
        attack_df = pd.read_csv(attack_path)
        is_valid_a, missing_a = validate_csv(attack_df, prefix, strict=False)
        if not is_valid_a:
            print(f"  [SCHEMA] attack: backfilling {len(missing_a)} missing fields: {missing_a}")
            attack_df = backfill_missing_fields(attack_df, prefix)
        attack_df = label_df(attack_df, prefix)

    # Sample
    clean_samples = sample_clean_drift(clean_df, prefix, n=args.n_clean)
    paired_samples = (
        sample_paired_drift(clean_df, attack_df, prefix, n=args.n_paired)
        if attack_df is not None else []
    )
    total = len(clean_samples) + len(paired_samples)
    print(f"\n  Sampled: {len(clean_samples)} clean drift + {len(paired_samples)} paired drift")

    if args.sample_only or total == 0:
        output = {
            "reader_tag": args.reader_tag,
            "retriever": args.retriever,
            "classifier_prompt_hash": DRIFT_AUDIT_PROMPT_HASH,
            "n_clean_sampled": len(clean_samples),
            "n_paired_sampled": len(paired_samples),
            "clean_drift_samples": clean_samples,
            "paired_drift_samples": paired_samples,
        }
    else:
        # Classify with LLM
        config = LLMBackendConfig(
            backend=args.classifier_backend,
            base_url=args.classifier_url,
            model_id=args.classifier_model,
            api_key_env="OPENAI_API_KEY",
            temperature=args.classifier_temperature,
            max_tokens=args.classifier_max_tokens,
            timeout=60,
        )

        print(f"\n  Classifying {total} instances with {args.classifier_model}...")
        clean_classified = classify_drift_batch(clean_samples, config, verbose=True)
        paired_classified = classify_drift_batch(paired_samples, config, verbose=True)

        # Summary
        all_labels = [r["drift_label"] for r in clean_classified + paired_classified]
        counts = Counter(all_labels)
        print(f"\n  === Drift Audit Summary ===")
        for cat in sorted(CATEGORY_DESCRIPTIONS.keys()):
            n = counts.get(cat, 0)
            pct = n / total * 100 if total > 0 else 0
            print(f"    {cat:25s}: {n:4d} ({pct:5.1f}%) — {CATEGORY_DESCRIPTIONS[cat]}")

        genuine_count = counts.get("GENUINE", 0)
        artifact_count = counts.get("EXTRACTION_ARTIFACT", 0)
        dataset_count = counts.get("DATASET_ISSUE", 0)
        print(f"\n  Total audited: {total}")
        print(f"  Genuine drift:         {genuine_count} ({genuine_count/total*100:.1f}%)")
        print(f"  Extraction artifacts:  {artifact_count} ({artifact_count/total*100:.1f}%)")
        print(f"  Dataset issues:        {dataset_count} ({dataset_count/total*100:.1f}%)")

        output = {
            "reader_tag": args.reader_tag,
            "retriever": args.retriever,
            "classifier_model": args.classifier_model,
            "classifier_prompt_hash": DRIFT_AUDIT_PROMPT_HASH,
            "n_clean_sampled": len(clean_classified),
            "n_paired_sampled": len(paired_classified),
            "summary": dict(counts),
            "genuine_fraction": genuine_count / total if total > 0 else 0,
            "clean_drift_audit": clean_classified,
            "paired_drift_audit": paired_classified,
        }

    out_path = args.output or os.path.join(
        args.csv_dir, f"drift_audit_{args.reader_tag}_{args.retriever}.json")
    Path(out_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
