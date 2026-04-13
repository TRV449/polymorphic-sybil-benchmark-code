#!/usr/bin/env python3
"""
full_recompute_post_bugfix.py — Post gold-alias-bug comprehensive re-computation.

Covers Tasks 1-4, 7-8:
  1. BM25+ColBERT full metrics (clean/attack/forced, bootstrap CI)
  2. E5+CE full metrics (clean/attack/forced, bootstrap CI)
  3. Paired transition rates (both retrievers)
  4. Forced 3-way comparison (gold_only / distractor / sybil+gold)
  7. Gold-given utilization
  8. Sybil@10, sybil count dose-response

All with corrected gold alias splitting.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
from typing import List, Tuple

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "benchmark_core"))
sys.path.insert(0, _ROOT)
from benchmark_eval_utils import classify_answer, split_answers

N_BOOT = 10000
SEED = 42

# ── Paths ────────────────────────────────────────────────────────────────────
RT = os.path.join(os.environ.get("WORKSPACE_ROOT", os.path.join(_ROOT, "..")), "member_runtime")
COLBERT_MERGED = f"{RT}/colbert_locked_merged.csv"
E5CE_CLEAN     = f"{RT}/e5_ce_clean_200_20260412_233533.csv"
E5CE_ATTACK    = f"{RT}/e5_ce_attack_200_20260412_234753.csv"
E5CE_FORCED    = f"{RT}/e5_ce_forced_200_20260412_235628.csv"
FORCED_REF     = f"{RT}/forced_ref_200.csv"

# ── Bootstrap ────────────────────────────────────────────────────────────────
def bootstrap_ci(values: np.ndarray, n_boot=N_BOOT, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    pt = np.mean(values)
    alpha = (1 - ci) / 2
    boots = np.array([np.mean(values[rng.randint(0, n, n)]) for _ in range(n_boot)])
    lo = np.percentile(boots, alpha * 100)
    hi = np.percentile(boots, (1 - alpha) * 100)
    return float(pt), float(lo), float(hi)

def bootstrap_paired_diff(va, vb, n_boot=N_BOOT, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(va)
    pt = float(np.mean(va) - np.mean(vb))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boots[b] = np.mean(va[idx]) - np.mean(vb[idx])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boots, alpha * 100))
    hi = float(np.percentile(boots, (1 - alpha) * 100))
    return pt, lo, hi

def fmt(pt, lo, hi, pct=True):
    m = 100 if pct else 1
    return f"{pt*m:5.1f}% [{lo*m:.1f}-{hi*m:.1f}]"

def fmt_diff(pt, lo, hi):
    sig = "*" if (lo > 0 or hi < 0) else " "
    return f"{pt*100:+5.1f}pp [{lo*100:+.1f},{hi*100:+.1f}]{sig}"

# ── Label computation (FIXED) ───────────────────────────────────────────────
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
        for metric, lbl in [("ACC","gold"),("ASR","target"),("Abstain","abstain"),("Drift","drift")]:
            vals = (sub[label_col] == lbl).astype(float).values
            pt, lo, hi = bootstrap_ci(vals)
            m[metric] = {"pt": pt, "lo": lo, "hi": hi}
        # CACC, CASR
        abst = m["Abstain"]["pt"]
        denom = max(1.0 - abst, 0.001)
        m["CACC"] = {"pt": m["ACC"]["pt"] / denom, "lo": 0.0, "hi": 0.0}
        m["CASR"] = {"pt": m["ASR"]["pt"] / denom, "lo": 0.0, "hi": 0.0}
        # Retrieval metrics
        for rc in ["gold_in_top10", "sybil_in_top10"]:
            if rc in sub.columns:
                vals = sub[rc].astype(float).values
                pt, lo, hi = bootstrap_ci(vals)
                m[rc] = {"pt": pt, "lo": lo, "hi": hi}
        m["n"] = n
        results[scope] = m
    return results

# ── Paired transitions ───────────────────────────────────────────────────────
def paired_transitions(df, clean_label_col, attack_label_col):
    results = {}
    for scope in ["all", "hotpot", "nq"]:
        sub = df if scope == "all" else df[df["ds"] == scope]
        clean_gold = sub[sub[clean_label_col] == "gold"]
        n_cg = len(clean_gold)
        if n_cg == 0:
            results[scope] = {"n_clean_gold": 0}
            continue
        t = {}
        for name, lbl in [("survive","gold"),("hijack","target"),("drift","drift"),("abstain","abstain")]:
            vals = (clean_gold[attack_label_col] == lbl).astype(float).values
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
            for metric, lbl in [("correct","gold"),("abstain","abstain"),("drift","drift"),("target","target")]:
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
        for metric, lbl in [("ACC","gold"),("ASR","target"),("Abstain","abstain"),("Drift","drift")]:
            vals = (sub[label_col] == lbl).astype(float).values
            pt, lo, hi = bootstrap_ci(vals)
            m[metric] = {"pt": pt, "lo": lo, "hi": hi}
        m["n"] = n
        results[str(count)] = m
    return results

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    all_results = {}

    # ── 1. BM25+ColBERT ─────────────────────────────────────────────────────
    print("=" * 70)
    print("  TASK 1: BM25+ColBERT Full Metrics (500Q)")
    print("=" * 70)
    df_cb = pd.read_csv(COLBERT_MERGED)
    bm25 = {"name": "Qwen-7B (BM25+ColBERT)", "scale": 7, "n": len(df_cb)}

    for track, prefix in [("clean", "clean"), ("attack", "attack"), ("forced", "forced")]:
        df_cb[f"label_{track}"] = compute_labels(df_cb, prefix)
        m = track_metrics(df_cb, label_col=f"label_{track}")
        bm25[track] = m
        print(f"\n  --- {track.upper()} ---")
        for scope, sm in m.items():
            if isinstance(sm, dict) and "n" in sm:
                parts = [f"{k}={fmt(sm[k]['pt'],sm[k]['lo'],sm[k]['hi'])}" for k in ["ACC","ASR","Abstain","Drift"] if k in sm]
                cacc = f"CACC={sm['CACC']['pt']*100:.1f}%" if "CACC" in sm else ""
                print(f"    {scope:6} n={sm['n']:4d}  " + "  ".join(parts) + f"  {cacc}")

    # Paired transition
    trans_bm25 = paired_transitions(df_cb, "label_clean", "label_attack")
    bm25["transitions_clean_attack"] = trans_bm25
    print(f"\n  --- PAIRED TRANSITION (clean→attack) ---")
    for scope, t in trans_bm25.items():
        n_cg = t.get("n_clean_gold", 0)
        if n_cg == 0:
            continue
        parts = [f"{k}={fmt(t[k]['pt'],t[k]['lo'],t[k]['hi'])}" for k in ["survive","hijack","drift","abstain"] if k in t]
        print(f"    {scope:6} clean-gold n={n_cg:3d}  " + "  ".join(parts))

    # Paired transition (clean→forced)
    trans_bm25_f = paired_transitions(df_cb, "label_clean", "label_forced")
    bm25["transitions_clean_forced"] = trans_bm25_f
    print(f"\n  --- PAIRED TRANSITION (clean→forced) ---")
    for scope, t in trans_bm25_f.items():
        n_cg = t.get("n_clean_gold", 0)
        if n_cg == 0:
            continue
        parts = [f"{k}={fmt(t[k]['pt'],t[k]['lo'],t[k]['hi'])}" for k in ["survive","hijack","drift","abstain"] if k in t]
        print(f"    {scope:6} clean-gold n={n_cg:3d}  " + "  ".join(parts))

    # Paired diff clean-attack
    print(f"\n  --- PAIRED DIFF (clean - attack) ---")
    bm25_paired = {}
    for scope in ["all", "hotpot", "nq"]:
        sub = df_cb if scope == "all" else df_cb[df_cb["ds"] == scope]
        diffs = {}
        for metric, lbl in [("ACC","gold"),("ASR","target"),("Abstain","abstain"),("Drift","drift")]:
            va = (sub["label_clean"] == lbl).astype(float).values
            vb = (sub["label_attack"] == lbl).astype(float).values
            pt, lo, hi = bootstrap_paired_diff(va, vb)
            diffs[f"d{metric}"] = {"pt": pt, "lo": lo, "hi": hi}
        bm25_paired[scope] = diffs
        parts = [f"Δ{k[1:]}={fmt_diff(diffs[k]['pt'],diffs[k]['lo'],diffs[k]['hi'])}" for k in diffs]
        print(f"    {scope:6} n={len(sub):4d}  " + "  ".join(parts))
    bm25["paired_diff_clean_attack"] = bm25_paired
    all_results["bm25_colbert"] = bm25

    # ── 2. E5+CE ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TASK 2: E5+CE Full Metrics (200Q)")
    print("=" * 70)
    e5ce = {"name": "Qwen-7B (E5+CE)", "scale": 7}
    e5ce_dfs = {}
    for track, csv_path in [("clean", E5CE_CLEAN), ("attack", E5CE_ATTACK), ("forced", E5CE_FORCED)]:
        df = pd.read_csv(csv_path)
        df["label"] = compute_labels(df, "dense")
        e5ce_dfs[track] = df
        m = track_metrics(df)
        e5ce[track] = m
        print(f"\n  --- {track.upper()} ---")
        for scope, sm in m.items():
            if isinstance(sm, dict) and "n" in sm:
                parts = [f"{k}={fmt(sm[k]['pt'],sm[k]['lo'],sm[k]['hi'])}" for k in ["ACC","ASR","Abstain","Drift"] if k in sm]
                cacc = f"CACC={sm['CACC']['pt']*100:.1f}%" if "CACC" in sm else ""
                extras = []
                if "gold_in_top10" in sm:
                    extras.append(f"Gold@10={fmt(sm['gold_in_top10']['pt'],sm['gold_in_top10']['lo'],sm['gold_in_top10']['hi'])}")
                if "sybil_in_top10" in sm:
                    extras.append(f"Sybil@10={fmt(sm['sybil_in_top10']['pt'],sm['sybil_in_top10']['lo'],sm['sybil_in_top10']['hi'])}")
                print(f"    {scope:6} n={sm['n']:4d}  " + "  ".join(parts) + f"  {cacc}" + ("  " + "  ".join(extras) if extras else ""))

    # ── 3. E5+CE Paired transitions ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TASK 3: Paired Transition Rates")
    print("=" * 70)

    merged_e5 = e5ce_dfs["clean"][["ds","id","label"]].merge(
        e5ce_dfs["attack"][["ds","id","label"]], on=["ds","id"], suffixes=("_clean","_attack"))
    trans_e5 = {}
    print(f"\n  --- E5+CE clean→attack ---")
    for scope in ["all", "hotpot", "nq"]:
        sub = merged_e5 if scope == "all" else merged_e5[merged_e5["ds"] == scope]
        cg = sub[sub["label_clean"] == "gold"]
        n_cg = len(cg)
        if n_cg == 0:
            continue
        t = {}
        for name, lbl in [("survive","gold"),("hijack","target"),("drift","drift"),("abstain","abstain")]:
            vals = (cg["label_attack"] == lbl).astype(float).values
            pt, lo, hi = bootstrap_ci(vals)
            t[name] = {"pt": pt, "lo": lo, "hi": hi}
        t["n_clean_gold"] = n_cg
        trans_e5[scope] = t
        parts = [f"{k}={fmt(t[k]['pt'],t[k]['lo'],t[k]['hi'])}" for k in ["survive","hijack","drift","abstain"] if k in t]
        print(f"    {scope:6} clean-gold n={n_cg:3d}  " + "  ".join(parts))
    e5ce["transitions_clean_attack"] = trans_e5

    # Paired diff
    print(f"\n  --- E5+CE PAIRED DIFF (clean - attack) ---")
    e5_paired = {}
    for scope in ["all", "hotpot", "nq"]:
        sub = merged_e5 if scope == "all" else merged_e5[merged_e5["ds"] == scope]
        diffs = {}
        for metric, lbl in [("ACC","gold"),("ASR","target"),("Abstain","abstain"),("Drift","drift")]:
            va = (sub["label_clean"] == lbl).astype(float).values
            vb = (sub["label_attack"] == lbl).astype(float).values
            pt, lo, hi = bootstrap_paired_diff(va, vb)
            diffs[f"d{metric}"] = {"pt": pt, "lo": lo, "hi": hi}
        e5_paired[scope] = diffs
        parts = [f"Δ{k[1:]}={fmt_diff(diffs[k]['pt'],diffs[k]['lo'],diffs[k]['hi'])}" for k in diffs]
        print(f"    {scope:6} n={len(sub):4d}  " + "  ".join(parts))
    e5ce["paired_diff_clean_attack"] = e5_paired
    all_results["e5_ce"] = e5ce

    # ── 4. Forced 3-way comparison (gold_only / distractor) ─────────────────
    print(f"\n{'='*70}")
    print("  TASK 4: Forced 3-way Comparison")
    print("=" * 70)
    df_fr = pd.read_csv(FORCED_REF)
    # Re-label with fixed splitting
    labels_fixed = []
    for _, r in df_fr.iterrows():
        golds = [g.strip() for g in str(r["gold_answer"]).split("|")]
        targets = [t.strip() for t in str(r["poison_target"]).split("|") if t.strip()]
        result = classify_answer(
            eval_answer=str(r["answer_final"]),
            extracted_text_for_conflict=str(r["answer_raw"]),
            gold_aliases=golds,
            target_aliases=targets,
            explicit_abstain=bool(r["abstain"]),
        )
        labels_fixed.append(result["label"])
    df_fr["label_fixed"] = labels_fixed

    forced_ref = {}
    for cond in ["gold_only", "distractor"]:
        cdf = df_fr[df_fr["condition"] == cond]
        cdf_m = {}
        print(f"\n  [{cond}]")
        for scope in ["all", "hotpot", "nq"]:
            sub = cdf if scope == "all" else cdf[cdf["ds"] == scope]
            n = len(sub)
            m = {}
            for metric, lbl in [("ACC","gold"),("ASR","target"),("Abstain","abstain"),("Drift","drift")]:
                vals = (sub["label_fixed"] == lbl).astype(float).values
                pt, lo, hi = bootstrap_ci(vals)
                m[metric] = {"pt": pt, "lo": lo, "hi": hi}
            abst = m["Abstain"]["pt"]
            denom = max(1.0 - abst, 0.001)
            m["CACC"] = {"pt": m["ACC"]["pt"] / denom, "lo": 0.0, "hi": 0.0}
            m["n"] = n
            cdf_m[scope] = m
            parts = [f"{k}={fmt(m[k]['pt'],m[k]['lo'],m[k]['hi'])}" for k in ["ACC","Abstain","Drift"]]
            print(f"    {scope:6} n={n:3d}  " + "  ".join(parts) + f"  CACC={m['CACC']['pt']*100:.1f}%")
        forced_ref[cond] = cdf_m

    # Add sybil+gold from E5+CE forced track
    if "forced" in e5ce:
        forced_ref["sybil_gold_e5ce"] = e5ce["forced"]

    # Add sybil+gold from BM25+ColBERT forced track
    if "forced" in bm25:
        forced_ref["sybil_gold_bm25"] = bm25["forced"]

    print(f"\n  --- Cross-Retriever CACC Summary ---")
    print(f"  {'Condition':<20} {'BM25+ColBERT':>15} {'E5+CE':>15}")
    for cond_key, cond_label in [("gold_only", "Gold-only"),
                                  ("distractor", "Distractor"),
                                  ("sybil_gold_bm25", "Sybil+Gold (BM25)"),
                                  ("sybil_gold_e5ce", "Sybil+Gold (E5)")]:
        if cond_key in forced_ref and "all" in forced_ref[cond_key]:
            cacc = forced_ref[cond_key]["all"]["CACC"]["pt"] * 100
            print(f"  {cond_label:<20} {cacc:>14.1f}%")

    all_results["forced_reference"] = forced_ref

    # ── 7. Gold-given utilization ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TASK 7: Gold-given Utilization (E5+CE)")
    print("=" * 70)
    gg_clean = gold_given_analysis(e5ce_dfs["clean"])
    gg_attack = gold_given_analysis(e5ce_dfs["attack"])
    e5ce_gg = {"clean": gg_clean, "attack": gg_attack}
    for track_name, gg in [("CLEAN", gg_clean), ("ATTACK", gg_attack)]:
        print(f"\n  --- {track_name} ---")
        for scope, sg in gg.items():
            for cond_name, cond_data in sg.items():
                n = cond_data.get("n", 0)
                if n == 0:
                    continue
                parts = [f"{k}={fmt(cond_data[k]['pt'],cond_data[k]['lo'],cond_data[k]['hi'])}" for k in ["correct","abstain","drift","target"] if k in cond_data]
                util = cond_data.get("utilization")
                util_str = f"  util={util*100:.1f}%" if util is not None else ""
                print(f"    {scope:6} {cond_name:14} n={n:3d}  " + "  ".join(parts) + util_str)
    all_results["gold_given"] = e5ce_gg

    # ── 8. Sybil@10 dose-response ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TASK 8: Sybil@10 & Dose-Response (E5+CE Attack)")
    print("=" * 70)
    dose = sybil_dose_response(e5ce_dfs["attack"])
    if dose:
        print(f"\n  {'Sybil#':>6} {'n':>4}  {'ACC':>18}  {'ASR':>18}  {'Abstain':>18}")
        for sc, m in sorted(dose.items(), key=lambda x: int(x[0])):
            print(f"  {sc:>6} {m['n']:>4}  {fmt(m['ACC']['pt'],m['ACC']['lo'],m['ACC']['hi']):>18}  "
                  f"{fmt(m['ASR']['pt'],m['ASR']['lo'],m['ASR']['hi']):>18}  "
                  f"{fmt(m['Abstain']['pt'],m['Abstain']['lo'],m['Abstain']['hi']):>18}")
    else:
        print("  No sybil_count_in_top10 column found")

    # Overall Sybil@10 verification
    if "attack" in e5ce and "all" in e5ce["attack"] and "sybil_in_top10" in e5ce["attack"]["all"]:
        s10 = e5ce["attack"]["all"]["sybil_in_top10"]
        print(f"\n  Sybil@10 overall: {fmt(s10['pt'], s10['lo'], s10['hi'])}")
    if "attack" in e5ce and "all" in e5ce["attack"] and "gold_in_top10" in e5ce["attack"]["all"]:
        g10 = e5ce["attack"]["all"]["gold_in_top10"]
        print(f"  Gold@10 (attack): {fmt(g10['pt'], g10['lo'], g10['hi'])}")

    all_results["sybil_dose_response"] = dose

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path = f"{RT}/full_metrics_post_bugfix.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else o)
    print(f"\n{'='*70}")
    print(f"  ALL RESULTS SAVED → {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
