#!/usr/bin/env python3
"""
bootstrap_ci.py — Bootstrap confidence intervals for benchmark metrics.

Computes 95% CI for ACC, ASR, Abstain, Drift rates via BCa bootstrap.
Supports paired comparison between two systems.

Usage:
  python3 bootstrap_ci.py --csvs file1.csv file2.csv --prefixes colbert dense \
    --names "BM25+ColBERT" "E5+CE" --n_boot 10000
"""
from __future__ import annotations

import argparse
import sys
import os
import numpy as np
import pandas as pd
from typing import List, Tuple

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "benchmark_core"))
sys.path.insert(0, _ROOT)

from benchmark_eval_utils import classify_answer


def get_labels(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Compute 4-way label column from CSV."""
    labels = []
    for _, row in df.iterrows():
        result = classify_answer(
            eval_answer=str(row[f"{prefix}_answer_final"]),
            extracted_text_for_conflict=str(row[f"{prefix}_answer_raw"]),
            gold_aliases=[g.strip() for g in str(row["gold_answer"]).split("|")],
            target_aliases=[t.strip() for t in str(row["poison_target"]).split("|") if t.strip()],
            explicit_abstain=bool(row[f"{prefix}_abstain"]),
        )
        labels.append(result["label"])
    return pd.Series(labels, index=df.index)


def bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """BCa bootstrap CI. Returns (point_estimate, ci_low, ci_high)."""
    rng = np.random.RandomState(seed)
    n = len(values)
    point = stat_fn(values)

    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.randint(0, n, size=n)]
        boot_stats[b] = stat_fn(sample)

    alpha = (1 - ci) / 2

    # Bias correction
    z0 = _norm_ppf(np.mean(boot_stats < point))

    # Acceleration (jackknife)
    jack_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(values, i)
        jack_stats[i] = stat_fn(jack_sample)
    jack_mean = jack_stats.mean()
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a_hat = num / den if den != 0 else 0.0

    # Adjusted percentiles
    z_alpha = _norm_ppf(alpha)
    z_1alpha = _norm_ppf(1 - alpha)
    p_low = _norm_cdf(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha)))
    p_high = _norm_cdf(z0 + (z0 + z_1alpha) / (1 - a_hat * (z0 + z_1alpha)))

    p_low = np.clip(p_low, 0.001, 0.999)
    p_high = np.clip(p_high, 0.001, 0.999)

    ci_low = np.percentile(boot_stats, p_low * 100)
    ci_high = np.percentile(boot_stats, p_high * 100)

    return point, ci_low, ci_high


def _norm_ppf(p):
    """Normal inverse CDF (approximation)."""
    from scipy.stats import norm
    return norm.ppf(np.clip(p, 1e-10, 1 - 1e-10))


def _norm_cdf(z):
    """Normal CDF."""
    from scipy.stats import norm
    return norm.cdf(z)


def bootstrap_paired_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    stat_fn=np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the difference stat(A) - stat(B), paired."""
    rng = np.random.RandomState(seed)
    n = len(values_a)
    assert len(values_b) == n, "Paired comparison requires same length"
    point = stat_fn(values_a) - stat_fn(values_b)

    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_diffs[b] = stat_fn(values_a[idx]) - stat_fn(values_b[idx])

    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_diffs, alpha * 100)
    ci_high = np.percentile(boot_diffs, (1 - alpha) * 100)
    return point, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI for benchmark metrics")
    parser.add_argument("--csvs", nargs="+", required=True, help="CSV files to analyze")
    parser.add_argument("--prefixes", nargs="+", required=True, help="Column prefixes per CSV")
    parser.add_argument("--names", nargs="+", required=True, help="Display names per CSV")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paired", action="store_true",
                        help="Compute paired difference CI (requires exactly 2 CSVs)")
    args = parser.parse_args()

    assert len(args.csvs) == len(args.prefixes) == len(args.names)

    dfs = []
    label_cols = []
    for csv_path, prefix, name in zip(args.csvs, args.prefixes, args.names):
        df = pd.read_csv(csv_path)
        df["label"] = get_labels(df, prefix)
        dfs.append(df)
        label_cols.append(df["label"])
        print(f"Loaded {name}: {len(df)} rows from {os.path.basename(csv_path)}")

    metrics = ["gold", "target", "abstain", "drift"]
    metric_display = {"gold": "ACC", "target": "ASR", "abstain": "Abstain", "drift": "Drift"}

    print("\n" + "=" * 80)
    print("  Bootstrap 95% CI (BCa, n_boot={})".format(args.n_boot))
    print("=" * 80)

    for di, (df, name) in enumerate(zip(dfs, args.names)):
        print(f"\n  [{name}]")
        for scope in ["all", "hotpot", "nq"]:
            sub = df if scope == "all" else df[df["ds"] == scope]
            n = len(sub)
            parts = []
            for metric in metrics:
                vals = (sub["label"] == metric).astype(float).values
                pt, lo, hi = bootstrap_ci(vals, n_boot=args.n_boot, seed=args.seed)
                parts.append(f"{metric_display[metric]}={pt*100:5.1f}% [{lo*100:.1f}-{hi*100:.1f}]")
            print(f"    {scope:6} n={n:4d}  " + "  ".join(parts))

    # Paired comparison
    if args.paired and len(dfs) == 2:
        print(f"\n{'='*80}")
        print(f"  Paired Difference: {args.names[0]} - {args.names[1]}")
        print(f"  (positive = {args.names[0]} higher)")
        print(f"{'='*80}")

        df_a, df_b = dfs
        # Align on (ds, id)
        merged = df_a.merge(df_b, on=["ds", "id"], suffixes=("_A", "_B"), how="inner")
        print(f"  Paired on {len(merged)} shared questions")

        for scope in ["all", "hotpot", "nq"]:
            sub = merged if scope == "all" else merged[merged["ds"] == scope]
            n = len(sub)
            parts = []
            for metric in metrics:
                vals_a = (sub["label_A"] == metric).astype(float).values
                vals_b = (sub["label_B"] == metric).astype(float).values
                pt, lo, hi = bootstrap_paired_diff(vals_a, vals_b,
                                                    n_boot=args.n_boot, seed=args.seed)
                sig = "*" if (lo > 0 or hi < 0) else " "
                parts.append(f"Δ{metric_display[metric]}={pt*100:+5.1f}pp [{lo*100:+.1f},{hi*100:+.1f}]{sig}")
            print(f"    {scope:6} n={n:4d}  " + "  ".join(parts))

    print()


if __name__ == "__main__":
    main()
