#!/usr/bin/env python3
"""
day4_auto_analysis.py — Unified analysis pipeline for Day 4 experiment CSVs.

Given a reader tag and retriever, finds the output CSVs and computes:
  - Per-track metrics (ACC, ASR, Abstain, Drift, CACC, CASR) with bootstrap CI
  - Paired transition rates (clean→attack, clean→forced)
  - Paired diffs with significance
  - Gold-given utilization (if gold_in_top10 column present)
  - Sybil dose-response (if sybil_count_in_top10 column present)
  - Per-dataset (hotpot/nq) breakdowns

Usage:
  python3 day4_auto_analysis.py \\
      --reader_tag qwen7b \\
      --retriever colbert \\
      --csv_dir ./member_runtime \\
      [--tracks clean attack forced] \\
      [--output_json auto_analysis_qwen7b_colbert.json]

CSV naming convention:
  day4_{reader_tag}_{retriever}_{track}_{timestamp}.csv
  e.g. day4_qwen7b_colbert_clean_20260413_083634.csv
"""
from __future__ import annotations
import argparse, glob, json, os, sys, re
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "benchmark_core"))
sys.path.insert(0, _ROOT)
from benchmark_eval_utils import classify_answer

N_BOOT = 10_000
SEED = 42

# Map retriever name to column prefix used by runners
RETRIEVER_PREFIX = {
    "colbert": "colbert",
    "e5ce":    "dense",
}


# ── Bootstrap utilities ─────────────────────────────────────────────────────
def bootstrap_ci(values: np.ndarray, n_boot=N_BOOT, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    pt = float(np.mean(values))
    alpha = (1 - ci) / 2
    boots = np.array([np.mean(values[rng.randint(0, n, n)]) for _ in range(n_boot)])
    return pt, float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))


def bootstrap_paired_diff(va, vb, n_boot=N_BOOT, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(va)
    pt = float(np.mean(va) - np.mean(vb))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boots[b] = np.mean(va[idx]) - np.mean(vb[idx])
    alpha = (1 - ci) / 2
    return pt, float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))


def fmt(pt, lo, hi):
    return f"{pt*100:5.1f}% [{lo*100:.1f}-{hi*100:.1f}]"

def fmt_diff(pt, lo, hi):
    sig = "*" if (lo > 0 or hi < 0) else " "
    return f"{pt*100:+5.1f}pp [{lo*100:+.1f},{hi*100:+.1f}]{sig}"


# ── Label computation (with gold alias splitting fix) ────────────────────────
def compute_labels(df: pd.DataFrame, prefix: str) -> pd.Series:
    labels = []
    for _, r in df.iterrows():
        golds = [g.strip() for g in str(r["gold_answer"]).split("|")]
        targets = [t.strip() for t in str(r.get("poison_target", "")).split("|") if t.strip()]
        result = classify_answer(
            eval_answer=str(r[f"{prefix}_answer_final"]),
            extracted_text_for_conflict=str(r[f"{prefix}_answer_raw"]),
            gold_aliases=golds,
            target_aliases=targets,
            explicit_abstain=bool(r[f"{prefix}_abstain"]),
        )
        labels.append(result["label"])
    return pd.Series(labels, index=df.index)


# ── Per-track metrics ────────────────────────────────────────────────────────
def track_metrics(df, label_col="label"):
    results = {}
    for scope in ["all", "hotpot", "nq"]:
        sub = df if scope == "all" else df[df["ds"] == scope]
        n = len(sub)
        if n == 0:
            continue
        m = {}
        for metric, lbl in [("ACC", "gold"), ("ASR", "target"), ("Abstain", "abstain"), ("Drift", "drift")]:
            vals = (sub[label_col] == lbl).astype(float).values
            pt, lo, hi = bootstrap_ci(vals)
            m[metric] = {"pt": pt, "lo": lo, "hi": hi}
        abst = m["Abstain"]["pt"]
        denom = max(1.0 - abst, 0.001)
        m["CACC"] = {"pt": m["ACC"]["pt"] / denom, "lo": 0.0, "hi": 0.0}
        m["CASR"] = {"pt": m["ASR"]["pt"] / denom, "lo": 0.0, "hi": 0.0}
        for rc in ["gold_in_top10", "sybil_in_top10"]:
            if rc in sub.columns:
                vals = sub[rc].astype(float).values
                pt, lo, hi = bootstrap_ci(vals)
                m[rc] = {"pt": pt, "lo": lo, "hi": hi}
        m["n"] = n
        results[scope] = m
    return results


# ── Paired transitions ───────────────────────────────────────────────────────
def paired_transitions(merged_df, clean_col, attack_col):
    results = {}
    for scope in ["all", "hotpot", "nq"]:
        sub = merged_df if scope == "all" else merged_df[merged_df["ds"] == scope]
        cg = sub[sub[clean_col] == "gold"]
        n_cg = len(cg)
        if n_cg == 0:
            results[scope] = {"n_clean_gold": 0}
            continue
        t = {}
        for name, lbl in [("survive", "gold"), ("hijack", "target"), ("drift", "drift"), ("abstain", "abstain")]:
            vals = (cg[attack_col] == lbl).astype(float).values
            pt, lo, hi = bootstrap_ci(vals)
            t[name] = {"pt": pt, "lo": lo, "hi": hi}
        t["n_clean_gold"] = n_cg
        t["n_total"] = len(sub)
        results[scope] = t
    return results


# ── Gold-given analysis ──────────────────────────────────────────────────────
def gold_given_analysis(df, label_col="label"):
    if "gold_in_top10" not in df.columns:
        return {}
    results = {}
    for scope in ["all", "hotpot", "nq"]:
        sub = df if scope == "all" else df[df["ds"] == scope]
        metrics = {}
        for cond_name, cond_mask in [("gold_present", sub["gold_in_top10"] == 1),
                                      ("gold_absent", sub["gold_in_top10"] == 0)]:
            subset = sub[cond_mask]
            n = len(subset)
            if n == 0:
                continue
            rates = {}
            for metric, lbl in [("correct", "gold"), ("abstain", "abstain"), ("drift", "drift"), ("target", "target")]:
                vals = (subset[label_col] == lbl).astype(float).values
                pt, lo, hi = bootstrap_ci(vals)
                rates[metric] = {"pt": pt, "lo": lo, "hi": hi}
            rates["n"] = n
            if cond_name == "gold_present":
                rates["utilization"] = rates["correct"]["pt"]
            metrics[cond_name] = rates
        results[scope] = metrics
    return results


# ── Sybil dose-response ─────────────────────────────────────────────────────
def sybil_dose_response(df, label_col="label"):
    if "sybil_count_in_top10" not in df.columns:
        return {}
    results = {}
    for count in sorted(df["sybil_count_in_top10"].dropna().unique()):
        count = int(count)
        sub = df[df["sybil_count_in_top10"] == count]
        n = len(sub)
        if n < 5:
            continue
        m = {}
        for metric, lbl in [("ACC", "gold"), ("ASR", "target"), ("Abstain", "abstain"), ("Drift", "drift")]:
            vals = (sub[label_col] == lbl).astype(float).values
            pt, lo, hi = bootstrap_ci(vals)
            m[metric] = {"pt": pt, "lo": lo, "hi": hi}
        m["n"] = n
        results[str(count)] = m
    return results


# ── CSV discovery ────────────────────────────────────────────────────────────
def find_csv(csv_dir: str, reader_tag: str, retriever: str, track: str) -> str | None:
    """Find the most recent CSV matching the pattern."""
    pattern = os.path.join(csv_dir, f"day4_{reader_tag}_{retriever}_{track}_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]  # most recent by timestamp suffix


# ── Print helpers ────────────────────────────────────────────────────────────
def print_track(name, m):
    print(f"\n  --- {name} ---")
    for scope, sm in m.items():
        if not isinstance(sm, dict) or "n" not in sm:
            continue
        parts = [f"{k}={fmt(sm[k]['pt'], sm[k]['lo'], sm[k]['hi'])}"
                 for k in ["ACC", "ASR", "Abstain", "Drift"] if k in sm]
        cacc = f"CACC={sm['CACC']['pt']*100:.1f}%" if "CACC" in sm else ""
        extras = []
        if "gold_in_top10" in sm:
            extras.append(f"Gold@10={fmt(sm['gold_in_top10']['pt'], sm['gold_in_top10']['lo'], sm['gold_in_top10']['hi'])}")
        if "sybil_in_top10" in sm:
            extras.append(f"Sybil@10={fmt(sm['sybil_in_top10']['pt'], sm['sybil_in_top10']['lo'], sm['sybil_in_top10']['hi'])}")
        line = f"    {scope:6} n={sm['n']:5d}  " + "  ".join(parts) + f"  {cacc}"
        if extras:
            line += "  " + "  ".join(extras)
        print(line)


def print_transitions(name, trans):
    print(f"\n  --- PAIRED TRANSITION ({name}) ---")
    for scope, t in trans.items():
        n_cg = t.get("n_clean_gold", 0)
        if n_cg == 0:
            continue
        parts = [f"{k}={fmt(t[k]['pt'], t[k]['lo'], t[k]['hi'])}"
                 for k in ["survive", "hijack", "drift", "abstain"] if k in t]
        print(f"    {scope:6} clean-gold n={n_cg:5d}  " + "  ".join(parts))


def print_diffs(name, diffs):
    print(f"\n  --- PAIRED DIFF ({name}) ---")
    for scope, d in diffs.items():
        parts = [f"Δ{k[1:]}={fmt_diff(d[k]['pt'], d[k]['lo'], d[k]['hi'])}" for k in d]
        print(f"    {scope:6}  " + "  ".join(parts))


# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Day 4 auto-analysis pipeline")
    parser.add_argument("--reader_tag", required=True, help="Reader tag (e.g. qwen7b, gptoss120b, qwen72b, gpt4omini)")
    parser.add_argument("--retriever", required=True, choices=["colbert", "e5ce"], help="Retriever type")
    parser.add_argument("--csv_dir", default="./member_runtime", help="Directory with output CSVs")
    parser.add_argument("--tracks", nargs="+", default=["clean", "attack", "forced"], help="Tracks to analyze")
    parser.add_argument("--output_json", default=None, help="Output JSON path (default: auto)")
    parser.add_argument("--csv_clean", default=None, help="Override: path to clean CSV")
    parser.add_argument("--csv_attack", default=None, help="Override: path to attack CSV")
    parser.add_argument("--csv_forced", default=None, help="Override: path to forced CSV")
    args = parser.parse_args()

    prefix = RETRIEVER_PREFIX[args.retriever]

    # Discover CSVs
    track_csvs = {}
    for track in args.tracks:
        override = getattr(args, f"csv_{track}", None)
        if override:
            csv_path = override
        else:
            csv_path = find_csv(args.csv_dir, args.reader_tag, args.retriever, track)
        if csv_path and os.path.isfile(csv_path):
            track_csvs[track] = csv_path
        else:
            print(f"  [WARN] CSV not found for track={track} "
                  f"(pattern: day4_{args.reader_tag}_{args.retriever}_{track}_*.csv)")

    if not track_csvs:
        print("ERROR: No CSVs found. Nothing to analyze.")
        sys.exit(1)

    print("=" * 72)
    print(f"  Day 4 Auto-Analysis: {args.reader_tag} × {args.retriever}")
    print(f"  Column prefix: {prefix}")
    print("=" * 72)
    for track, path in track_csvs.items():
        n = sum(1 for _ in open(path)) - 1
        print(f"  {track:8} → {os.path.basename(path)} ({n} rows)")

    # Load and label
    track_dfs = {}
    all_results = {
        "reader_tag": args.reader_tag,
        "retriever": args.retriever,
        "prefix": prefix,
        "tracks": {},
    }

    for track, csv_path in track_csvs.items():
        df = pd.read_csv(csv_path)
        df["label"] = compute_labels(df, prefix)
        track_dfs[track] = df

        m = track_metrics(df)
        all_results["tracks"][track] = m
        print_track(f"{track.upper()} (n={len(df)})", m)

    # Paired transitions: clean→attack
    if "clean" in track_dfs and "attack" in track_dfs:
        merged = track_dfs["clean"][["ds", "id", "label"]].merge(
            track_dfs["attack"][["ds", "id", "label"]],
            on=["ds", "id"], suffixes=("_clean", "_attack"))

        trans = paired_transitions(merged, "label_clean", "label_attack")
        all_results["transitions_clean_attack"] = trans
        print_transitions("clean→attack", trans)

        # Paired diffs
        diffs = {}
        for scope in ["all", "hotpot", "nq"]:
            sub = merged if scope == "all" else merged[merged["ds"] == scope]
            if len(sub) == 0:
                continue
            d = {}
            for metric, lbl in [("ACC", "gold"), ("ASR", "target"), ("Abstain", "abstain"), ("Drift", "drift")]:
                va = (sub["label_clean"] == lbl).astype(float).values
                vb = (sub["label_attack"] == lbl).astype(float).values
                pt, lo, hi = bootstrap_paired_diff(va, vb)
                d[f"d{metric}"] = {"pt": pt, "lo": lo, "hi": hi}
            diffs[scope] = d
        all_results["paired_diff_clean_attack"] = diffs
        print_diffs("clean - attack", diffs)

    # Paired transitions: clean→forced
    if "clean" in track_dfs and "forced" in track_dfs:
        merged_f = track_dfs["clean"][["ds", "id", "label"]].merge(
            track_dfs["forced"][["ds", "id", "label"]],
            on=["ds", "id"], suffixes=("_clean", "_forced"))

        trans_f = paired_transitions(merged_f, "label_clean", "label_forced")
        all_results["transitions_clean_forced"] = trans_f
        print_transitions("clean→forced", trans_f)

    # Gold-given analysis (per track)
    for track in ["clean", "attack"]:
        if track in track_dfs:
            gg = gold_given_analysis(track_dfs[track])
            if gg:
                all_results[f"gold_given_{track}"] = gg
                print(f"\n  --- GOLD-GIVEN ({track.upper()}) ---")
                for scope, sg in gg.items():
                    for cond_name, cond_data in sg.items():
                        n = cond_data.get("n", 0)
                        if n == 0:
                            continue
                        parts = [f"{k}={fmt(cond_data[k]['pt'], cond_data[k]['lo'], cond_data[k]['hi'])}"
                                 for k in ["correct", "abstain", "drift", "target"] if k in cond_data]
                        util = cond_data.get("utilization")
                        util_str = f"  util={util*100:.1f}%" if util is not None else ""
                        print(f"    {scope:6} {cond_name:14} n={n:5d}  " + "  ".join(parts) + util_str)

    # Sybil dose-response (attack track)
    if "attack" in track_dfs:
        dose = sybil_dose_response(track_dfs["attack"])
        if dose:
            all_results["sybil_dose_response"] = dose
            print(f"\n  --- SYBIL DOSE-RESPONSE (attack) ---")
            print(f"  {'Sybil#':>6} {'n':>5}  {'ACC':>18}  {'ASR':>18}  {'Abstain':>18}")
            for sc, m in sorted(dose.items(), key=lambda x: int(x[0])):
                print(f"  {sc:>6} {m['n']:>5}  {fmt(m['ACC']['pt'], m['ACC']['lo'], m['ACC']['hi']):>18}  "
                      f"{fmt(m['ASR']['pt'], m['ASR']['lo'], m['ASR']['hi']):>18}  "
                      f"{fmt(m['Abstain']['pt'], m['Abstain']['lo'], m['Abstain']['hi']):>18}")

    # Save JSON
    out_path = args.output_json or os.path.join(
        args.csv_dir, f"analysis_{args.reader_tag}_{args.retriever}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else o)

    print(f"\n{'=' * 72}")
    print(f"  RESULTS SAVED → {out_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
