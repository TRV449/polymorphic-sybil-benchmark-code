#!/usr/bin/env python3
"""
Answer Transition Matrix + Paired Analysis

Computes:
  - 4x4 transition matrix  (clean→attack, attack→v25)
  - Paired Drift Rate            P(attack=drift   | clean=gold)
  - Poison-induced Abstention    P(attack=abstain | clean=gold)
  - Defense Recovery Rate        P(v25=gold       | attack∈{target,drift})
  - Defense Abstention Rate      P(v25=abstain    | attack∈{target,drift})
  - 95% bootstrap CI for all paired rates
  - Dataset-wise (hotpot / nq) breakdown

Usage:
  python3 transition_matrix.py \\
      --input_csv results/fairness.csv \\
      --src_system base \\
      --dst_system attack \\
      --def_system v25 \\
      --output_json results/transitions.json \\
      --bootstrap_n 2000 \\
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmark_eval_utils import classify_answer, resolve_eval_answer, resolve_raw_answer, split_answers
from official_eval import discover_system_prefixes, load_csv_rows

LABELS = ("gold", "target", "drift", "abstain")


# ---------------------------------------------------------------------------
# label helper
# ---------------------------------------------------------------------------

def row_label(row: dict, prefix: str) -> str:
    result = classify_answer(
        eval_answer=resolve_eval_answer(row, prefix),
        extracted_text_for_conflict=resolve_raw_answer(row, prefix),
        gold_aliases=split_answers(row.get("gold_answer", "")),
        target_aliases=split_answers(row.get("poison_target", "")),
        explicit_abstain=row.get(f"{prefix}_abstain", ""),
    )
    return result["label"]


# ---------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------

def empty_matrix() -> Dict[str, Dict[str, int]]:
    return {src: {dst: 0 for dst in LABELS} for src in LABELS}


def build_matrix(rows: List[dict], src_prefix: str, dst_prefix: str) -> Dict[str, Dict[str, int]]:
    matrix = empty_matrix()
    for row in rows:
        src = row_label(row, src_prefix)
        dst = row_label(row, dst_prefix)
        matrix[src][dst] += 1
    return matrix


# ---------------------------------------------------------------------------
# paired rates
# ---------------------------------------------------------------------------

def paired_rates(
    rows: List[dict],
    clean_prefix: str,
    attack_prefix: str,
    defense_prefix: Optional[str] = None,
) -> dict:
    """
    Returns paired transition rates for one dataset slice.

    clean=gold 행을 분모로:
      - Paired Drift Rate            P(attack=drift   | clean=gold)
      - Poison-induced Abstention    P(attack=abstain | clean=gold)
      - Gold Retention under Attack  P(attack=gold    | clean=gold)
      - Targeted Hijack Rate         P(attack=target  | clean=gold)

    attack∈{target,drift} 행을 분모로 (defense_prefix 있을 때):
      - Defense Recovery Rate     P(v25=gold    | attack∈{target,drift})
      - Defense Abstention Rate   P(v25=abstain | attack∈{target,drift})
    """
    clean_gold_rows = []
    attack_failed_rows = []  # attack∈{target, drift}

    for row in rows:
        cl = row_label(row, clean_prefix)
        at = row_label(row, attack_prefix)
        if cl == "gold":
            clean_gold_rows.append((cl, at, row))
        if at in ("target", "drift"):
            attack_failed_rows.append((cl, at, row))

    n_cg = len(clean_gold_rows)
    result: dict = {
        "clean_gold_count": n_cg,
        "paired_drift_rate": 0.0,
        "poison_induced_abstention_rate": 0.0,
        "gold_retention_under_attack": 0.0,
        "targeted_hijack_from_gold": 0.0,
    }

    if n_cg > 0:
        result["paired_drift_rate"] = sum(1 for _, at, _ in clean_gold_rows if at == "drift") / n_cg
        result["poison_induced_abstention_rate"] = sum(1 for _, at, _ in clean_gold_rows if at == "abstain") / n_cg
        result["gold_retention_under_attack"] = sum(1 for _, at, _ in clean_gold_rows if at == "gold") / n_cg
        result["targeted_hijack_from_gold"] = sum(1 for _, at, _ in clean_gold_rows if at == "target") / n_cg

    if defense_prefix and attack_failed_rows:
        n_af = len(attack_failed_rows)
        result["attack_failed_count"] = n_af
        def_labels = [row_label(row, defense_prefix) for _, _, row in attack_failed_rows]
        result["defense_recovery_rate"] = def_labels.count("gold") / n_af
        result["defense_abstention_rate"] = def_labels.count("abstain") / n_af
        result["defense_hijack_residual"] = def_labels.count("target") / n_af
    elif defense_prefix:
        result["attack_failed_count"] = 0
        result["defense_recovery_rate"] = 0.0
        result["defense_abstention_rate"] = 0.0
        result["defense_hijack_residual"] = 0.0

    return result


# ---------------------------------------------------------------------------
# bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    rows: List[dict],
    clean_prefix: str,
    attack_prefix: str,
    defense_prefix: Optional[str],
    n: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """95% CI via question-level bootstrap for all paired rates.
    Pre-computes labels once to avoid repeated classify_answer calls.
    """
    # Pre-compute labels for all rows once
    labeled = []
    for row in rows:
        cl = row_label(row, clean_prefix)
        at = row_label(row, attack_prefix)
        df = row_label(row, defense_prefix) if defense_prefix else None
        labeled.append((cl, at, df))

    def _rates_from_labels(sample: List[Tuple]) -> dict:
        clean_gold = [(cl, at, df) for cl, at, df in sample if cl == "gold"]
        attack_failed = [(cl, at, df) for cl, at, df in sample if at in ("target", "drift")]
        n_cg = len(clean_gold)
        result: dict = {
            "paired_drift_rate": 0.0,
            "poison_induced_abstention_rate": 0.0,
            "gold_retention_under_attack": 0.0,
            "targeted_hijack_from_gold": 0.0,
        }
        if n_cg > 0:
            result["paired_drift_rate"] = sum(1 for _, at, _ in clean_gold if at == "drift") / n_cg
            result["poison_induced_abstention_rate"] = sum(1 for _, at, _ in clean_gold if at == "abstain") / n_cg
            result["gold_retention_under_attack"] = sum(1 for _, at, _ in clean_gold if at == "gold") / n_cg
            result["targeted_hijack_from_gold"] = sum(1 for _, at, _ in clean_gold if at == "target") / n_cg
        if defense_prefix and attack_failed:
            n_af = len(attack_failed)
            result["defense_recovery_rate"] = sum(1 for _, _, df in attack_failed if df == "gold") / n_af
            result["defense_abstention_rate"] = sum(1 for _, _, df in attack_failed if df == "abstain") / n_af
            result["defense_hijack_residual"] = sum(1 for _, _, df in attack_failed if df == "target") / n_af
        elif defense_prefix:
            result["defense_recovery_rate"] = 0.0
            result["defense_abstention_rate"] = 0.0
            result["defense_hijack_residual"] = 0.0
        return result

    keys = [
        "paired_drift_rate",
        "poison_induced_abstention_rate",
        "gold_retention_under_attack",
        "targeted_hijack_from_gold",
    ]
    if defense_prefix:
        keys += ["defense_recovery_rate", "defense_abstention_rate", "defense_hijack_residual"]

    rng = random.Random(seed)
    samples: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n):
        boot = rng.choices(labeled, k=len(labeled))
        r = _rates_from_labels(boot)
        for k in keys:
            if k in r:
                samples[k].append(r[k])

    ci: dict = {}
    lo = int(n * alpha / 2)
    hi = int(n * (1 - alpha / 2))
    for k, vals in samples.items():
        vals.sort()
        ci[k] = {"lo": vals[lo], "hi": vals[min(hi, n - 1)]}
    return ci


# ---------------------------------------------------------------------------
# dataset split
# ---------------------------------------------------------------------------

def by_dataset(rows: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for row in rows:
        ds = str(row.get("ds", "")).strip() or "unknown"
        out.setdefault(ds, []).append(row)
    return out


# ---------------------------------------------------------------------------
# full report
# ---------------------------------------------------------------------------

def build_report(
    rows: List[dict],
    src_prefix: str,
    dst_prefix: str,
    def_prefix: Optional[str],
    bootstrap_n: int,
    seed: int,
) -> dict:
    slices = {"combined": rows}
    slices.update(by_dataset(rows))

    report: dict = {}
    for scope, scope_rows in slices.items():
        # transition matrices
        clean_to_attack = build_matrix(scope_rows, src_prefix, dst_prefix)
        matrices = {"clean_to_attack": clean_to_attack}
        if def_prefix:
            matrices["attack_to_defense"] = build_matrix(scope_rows, dst_prefix, def_prefix)

        # paired rates (point estimate)
        rates = paired_rates(scope_rows, src_prefix, dst_prefix, def_prefix)

        # bootstrap CI
        ci = bootstrap_ci(scope_rows, src_prefix, dst_prefix, def_prefix,
                          n=bootstrap_n, seed=seed)

        report[scope] = {
            "n": len(scope_rows),
            "rates": rates,
            "ci_95": ci,
            "matrices": matrices,
        }

    return report


# ---------------------------------------------------------------------------
# text summary
# ---------------------------------------------------------------------------

def render_summary(report: dict, src: str, dst: str, defn: Optional[str]) -> str:
    lines = []
    for scope, data in report.items():
        r = data["rates"]
        ci = data["ci_95"]
        lines.append(f"\n{'='*60}")
        lines.append(f"[paired_analysis] scope={scope}  n={data['n']}")
        lines.append(f"  {src}→{dst} transitions (clean_gold_count={r['clean_gold_count']})")

        def fmt(key: str) -> str:
            v = r.get(key, 0.0)
            lo = ci.get(key, {}).get("lo", v)
            hi = ci.get(key, {}).get("hi", v)
            return f"{v*100:5.1f}%  [{lo*100:.1f}%, {hi*100:.1f}%]"

        lines.append(f"    Paired Drift Rate          : {fmt('paired_drift_rate')}")
        lines.append(f"    Poison-induced Abstention  : {fmt('poison_induced_abstention_rate')}")
        lines.append(f"    Gold→Gold (retained)       : {fmt('gold_retention_under_attack')}")
        lines.append(f"    Gold→Target (hijack)       : {fmt('targeted_hijack_from_gold')}")

        if defn and "defense_recovery_rate" in r:
            lines.append(f"  {dst}→{defn} (attack_failed_count={r.get('attack_failed_count',0)})")
            lines.append(f"    Defense Recovery Rate      : {fmt('defense_recovery_rate')}")
            lines.append(f"    Defense Abstention Rate    : {fmt('defense_abstention_rate')}")
            lines.append(f"    Defense Hijack Residual    : {fmt('defense_hijack_residual')}")

        # clean→attack matrix (Gold row만 본문에)
        m = data["matrices"]["clean_to_attack"]
        lines.append(f"  4×4 clean→attack matrix (row=clean, col=attack):")
        header = "          " + "".join(f"{lb:>10}" for lb in LABELS)
        lines.append(header)
        for src_lb in LABELS:
            row_total = sum(m[src_lb].values())
            row_str = f"  {src_lb:8}" + "".join(f"{m[src_lb][dst_lb]:>10}" for dst_lb in LABELS)
            lines.append(row_str)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Answer transition matrix + paired analysis")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--src_system", default="base", help="clean reference prefix (default: base)")
    parser.add_argument("--dst_system", default="attack", help="attack prefix (default: attack)")
    parser.add_argument("--def_system", default="", help="defense prefix (default: none / v25)")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--bootstrap_n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_csv_rows(args.input_csv)
    if not rows:
        raise SystemExit("[!] No rows in input CSV")

    # auto-detect prefixes if not overridden
    detected = discover_system_prefixes(rows[0].keys())
    src = args.src_system if args.src_system in detected else (detected[0] if detected else "base")
    dst = args.dst_system if args.dst_system in detected else (detected[1] if len(detected) > 1 else "attack")
    defn = args.def_system if args.def_system and args.def_system in detected else None

    def_label = defn or "(none)"
    print(f"[*] src={src}  dst={dst}  def={def_label}")
    print(f"[*] rows={len(rows)}  bootstrap_n={args.bootstrap_n}")

    report = build_report(rows, src, dst, defn, args.bootstrap_n, args.seed)
    print(render_summary(report, src, dst, defn))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[*] saved → {out}")


if __name__ == "__main__":
    main()
