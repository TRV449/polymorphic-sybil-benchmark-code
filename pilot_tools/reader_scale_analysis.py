#!/usr/bin/env python3
"""
reader_scale_analysis.py — Reader Scale Trend & Gold-given Abstain Rate Framework.

Given per-reader CSV results, computes:
  1. Per-reader, per-track metrics with bootstrap CI
  2. Paired transition rates (clean→attack) per reader
  3. Gold-given abstain/drift/correct rates (reader bottleneck analysis)
  4. Scale trend summary table

Usage:
  python3 reader_scale_analysis.py --config reader_config.json

Config JSON format:
  {
    "readers": [
      {
        "name": "Qwen-7B",
        "scale": 7,
        "clean_csv": "path/to/clean.csv",
        "attack_csv": "path/to/attack.csv",
        "forced_csv": "path/to/forced.csv",
        "prefix": "colbert"
      },
      ...
    ]
  }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "benchmark_core"))
sys.path.insert(0, _ROOT)

from benchmark_eval_utils import classify_answer


# ── Bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point, lo, hi)."""
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    pt = np.mean(values)
    alpha = (1 - ci) / 2
    boots = np.array([np.mean(values[rng.randint(0, n, n)]) for _ in range(n_boot)])
    lo = np.percentile(boots, alpha * 100)
    hi = np.percentile(boots, (1 - alpha) * 100)
    return pt, lo, hi


def fmt_ci(pt, lo, hi, pct=True):
    """Format point estimate with CI."""
    m = 100 if pct else 1
    return f"{pt*m:5.1f}% [{lo*m:.1f}-{hi*m:.1f}]"


# ── Label computation ────────────────────────────────────────────────────────

def compute_labels(df: pd.DataFrame, prefix: str) -> pd.Series:
    labels = []
    for _, r in df.iterrows():
        result = classify_answer(
            eval_answer=str(r[f"{prefix}_answer_final"]),
            extracted_text_for_conflict=str(r[f"{prefix}_answer_raw"]),
            gold_aliases=[g.strip() for g in str(r["gold_answer"]).split("|")],
            target_aliases=[t.strip() for t in str(r.get("poison_target", "")).split("|") if t.strip()],
            explicit_abstain=bool(r[f"{prefix}_abstain"]),
        )
        labels.append(result["label"])
    return pd.Series(labels, index=df.index)


# ── Analysis modules ─────────────────────────────────────────────────────────

def analyze_track(df: pd.DataFrame, n_boot: int = 10000, seed: int = 42) -> dict:
    """Compute metrics for a single track."""
    results = {}
    for scope in ["all", "hotpot", "nq"]:
        sub = df if scope == "all" else df[df["ds"] == scope]
        n = len(sub)
        if n == 0:
            continue

        metrics = {}
        for metric, label in [("ACC", "gold"), ("ASR", "target"),
                               ("Abstain", "abstain"), ("Drift", "drift")]:
            vals = (sub["label"] == label).astype(float).values
            pt, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
            metrics[metric] = {"pt": pt, "lo": lo, "hi": hi}

        # CACC, CASR
        abst_rate = metrics["Abstain"]["pt"]
        non_abst = 1.0 - abst_rate if abst_rate < 1.0 else 0.001
        metrics["CACC"] = {"pt": metrics["ACC"]["pt"] / non_abst,
                           "lo": 0.0, "hi": 0.0}  # CI for ratio is complex, skip
        metrics["CASR"] = {"pt": metrics["ASR"]["pt"] / non_abst,
                           "lo": 0.0, "hi": 0.0}

        # Gold@10 if available
        if "gold_in_top10" in sub.columns:
            vals = sub["gold_in_top10"].astype(float).values
            pt, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
            metrics["Gold@10"] = {"pt": pt, "lo": lo, "hi": hi}

        # Sybil@10 if available
        if "sybil_in_top10" in sub.columns:
            vals = sub["sybil_in_top10"].astype(float).values
            pt, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
            metrics["Sybil@10"] = {"pt": pt, "lo": lo, "hi": hi}

        metrics["n"] = n
        results[scope] = metrics

    return results


def analyze_paired_transition(
    df_clean: pd.DataFrame,
    df_attack: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired transition rates from clean-gold subset."""
    merged = df_clean[["ds", "id", "label"]].merge(
        df_attack[["ds", "id", "label"]],
        on=["ds", "id"], suffixes=("_clean", "_attack"),
    )
    results = {}
    for scope in ["all", "hotpot", "nq"]:
        sub = merged if scope == "all" else merged[merged["ds"] == scope]
        clean_gold = sub[sub["label_clean"] == "gold"]
        n_cg = len(clean_gold)
        if n_cg == 0:
            results[scope] = {"n_clean_gold": 0}
            continue

        transitions = {}
        for name, label in [("survive", "gold"), ("hijack", "target"),
                             ("drift", "drift"), ("abstain", "abstain")]:
            vals = (clean_gold["label_attack"] == label).astype(float).values
            pt, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
            transitions[name] = {"pt": pt, "lo": lo, "hi": hi}

        transitions["n_clean_gold"] = n_cg
        transitions["n_total"] = len(sub)
        results[scope] = transitions

    return results


def analyze_gold_given(
    df: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """Gold-given abstain/drift/correct rates (reader bottleneck analysis)."""
    if "gold_in_top10" not in df.columns:
        return {}

    results = {}
    for scope in ["all", "hotpot", "nq"]:
        sub = df if scope == "all" else df[df["ds"] == scope]
        gold_present = sub[sub["gold_in_top10"] == 1]
        gold_absent = sub[sub["gold_in_top10"] == 0]

        metrics = {}
        for name, subset, label_desc in [
            ("gold_present", gold_present, "Gold in top-10"),
            ("gold_absent", gold_absent, "Gold NOT in top-10"),
        ]:
            n = len(subset)
            if n == 0:
                continue
            rates = {}
            for metric, label in [("correct", "gold"), ("abstain", "abstain"),
                                   ("drift", "drift"), ("target", "target")]:
                vals = (subset["label"] == label).astype(float).values
                pt, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
                rates[metric] = {"pt": pt, "lo": lo, "hi": hi}
            rates["n"] = n

            # Utilization rate = correct / gold_present
            if name == "gold_present" and n > 0:
                correct_rate = rates["correct"]["pt"]
                rates["utilization"] = correct_rate  # how well reader uses gold docs

            metrics[name] = rates
        results[scope] = metrics

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reader scale trend analysis")
    parser.add_argument("--config", required=True, help="JSON config file")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default="", help="Save results to JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    all_results = {}

    for reader in config["readers"]:
        name = reader["name"]
        prefix = reader["prefix"]
        scale = reader.get("scale", 0)

        print(f"\n{'='*70}")
        print(f"  {name} (scale={scale}B)")
        print(f"{'='*70}")

        reader_results = {"name": name, "scale": scale, "tracks": {}}

        # Load and label each track
        tracks_data = {}
        for track in ["clean", "attack", "forced"]:
            csv_key = f"{track}_csv"
            if csv_key not in reader or not reader[csv_key]:
                continue
            csv_path = reader[csv_key]
            if not os.path.exists(csv_path):
                print(f"  [SKIP] {track}: {csv_path} not found")
                continue
            df = pd.read_csv(csv_path)
            # Support per-track prefix overrides (e.g. merged CSV with clean_/attack_ columns)
            track_prefix = reader.get(f"{track}_prefix", prefix)
            df["label"] = compute_labels(df, track_prefix)
            tracks_data[track] = df

            # Per-track metrics
            print(f"\n  --- {track.upper()} ---")
            track_metrics = analyze_track(df, n_boot=args.n_boot, seed=args.seed)
            reader_results["tracks"][track] = track_metrics
            for scope, metrics in track_metrics.items():
                n = metrics["n"]
                parts = []
                for m in ["ACC", "ASR", "Abstain", "Drift"]:
                    if m in metrics:
                        parts.append(f"{m}={fmt_ci(metrics[m]['pt'], metrics[m]['lo'], metrics[m]['hi'])}")
                print(f"    {scope:6} n={n:4d}  " + "  ".join(parts))

        # Paired transition (clean→attack)
        if "clean" in tracks_data and "attack" in tracks_data:
            print(f"\n  --- PAIRED TRANSITION (clean→attack) ---")
            trans = analyze_paired_transition(
                tracks_data["clean"], tracks_data["attack"],
                n_boot=args.n_boot, seed=args.seed,
            )
            reader_results["transitions"] = trans
            for scope, t in trans.items():
                n_cg = t.get("n_clean_gold", 0)
                if n_cg == 0:
                    continue
                parts = []
                for m in ["survive", "hijack", "drift", "abstain"]:
                    if m in t:
                        parts.append(f"{m}={fmt_ci(t[m]['pt'], t[m]['lo'], t[m]['hi'])}")
                print(f"    {scope:6} clean-gold n={n_cg:3d}  " + "  ".join(parts))

        # Gold-given analysis (clean track)
        if "clean" in tracks_data:
            print(f"\n  --- GOLD-GIVEN RATES (clean, reader bottleneck) ---")
            gold_given = analyze_gold_given(
                tracks_data["clean"], n_boot=args.n_boot, seed=args.seed,
            )
            reader_results["gold_given"] = gold_given
            for scope, sg in gold_given.items():
                for cond_name, cond_data in sg.items():
                    n = cond_data.get("n", 0)
                    if n == 0:
                        continue
                    parts = []
                    for m in ["correct", "abstain", "drift"]:
                        if m in cond_data:
                            parts.append(f"{m}={fmt_ci(cond_data[m]['pt'], cond_data[m]['lo'], cond_data[m]['hi'])}")
                    util = cond_data.get("utilization", None)
                    util_str = f"  util={util*100:.1f}%" if util is not None else ""
                    print(f"    {scope:6} {cond_name:14} n={n:3d}  " + "  ".join(parts) + util_str)

        all_results[name] = reader_results

    # ── Scale trend summary ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SCALE TREND SUMMARY")
    print(f"{'='*70}")

    readers_sorted = sorted(all_results.values(), key=lambda r: r["scale"])

    # Table header
    print(f"\n  {'Reader':<20} {'Scale':>5} {'Clean ACC':>10} {'Attack ASR':>10} "
          f"{'Hijack%':>10} {'Gold-Util':>10}")
    print(f"  {'-'*20} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in readers_sorted:
        name = r["name"]
        scale = r["scale"]

        clean_acc = "--"
        attack_asr = "--"
        hijack = "--"
        gold_util = "--"

        if "clean" in r["tracks"] and "all" in r["tracks"]["clean"]:
            acc = r["tracks"]["clean"]["all"]["ACC"]["pt"]
            clean_acc = f"{acc*100:.1f}%"

        if "attack" in r["tracks"] and "all" in r["tracks"]["attack"]:
            asr = r["tracks"]["attack"]["all"]["ASR"]["pt"]
            attack_asr = f"{asr*100:.1f}%"

        if "transitions" in r and "all" in r["transitions"]:
            t = r["transitions"]["all"]
            if "hijack" in t:
                hijack = f"{t['hijack']['pt']*100:.1f}%"

        if "gold_given" in r and "all" in r["gold_given"]:
            gg = r["gold_given"]["all"]
            if "gold_present" in gg and "utilization" in gg["gold_present"]:
                gold_util = f"{gg['gold_present']['utilization']*100:.1f}%"

        print(f"  {name:<20} {scale:>4}B {clean_acc:>10} {attack_asr:>10} "
              f"{hijack:>10} {gold_util:>10}")

    # Save results
    if args.output_json:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2, default=convert)
        print(f"\n  Results saved to {args.output_json}")

    print("\n  [reader_scale_analysis] done")


if __name__ == "__main__":
    main()
