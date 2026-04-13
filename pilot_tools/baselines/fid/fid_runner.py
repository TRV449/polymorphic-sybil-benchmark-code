#!/usr/bin/env python3
"""
FiD (Fusion-in-Decoder) external baseline evaluation.
Retrieval: BM25 (Lucene, same index as reference systems)
Reader:    FiD-large (T5-large, facebook/FiD NQ pretrained)

Tracks: base | attack | forced
  base   — clean BM25 retrieval, no poison
  attack — BM25 retrieval, frozen sybil poison docs prepended to candidate pool
  forced — Forced Exposure: gold docs at fixed positions, remaining slots sybil fill
           (NQ: gold at [0], Hotpot: gold at [0, 2])

Output schema per row:
  ds, id, question, gold_answer, poison_target,
  {prefix}_answer_raw, {prefix}_answer_eval, {prefix}_answer_final,
  {prefix}_abstain, {prefix}_conflict_flag, {prefix}_em, {prefix}_asr,
  [forced only] forced_gold_positions, forced_gold_count, forced_poison_count
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from pyserini.search.lucene import LuceneSearcher
from transformers import T5Tokenizer

# --- Path setup: file lives in baselines/fid/, benchmark_core/ is at pilot_tools/benchmark_core/ ---
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "benchmark_core"))
sys.path.insert(0, os.path.join(_ROOT, "stress_protocols"))
sys.path.insert(0, _ROOT)

# FiD source
FID_SRC = os.path.join(_ROOT, "..", "member_runtime", "fid_code", "src")
if FID_SRC not in sys.path:
    sys.path.insert(0, FID_SRC)
import model as fid_model

from benchmark_eval_utils import (
    eval_answer_from_raw,
    abstain_from_eval,
    conflict_flag_from_text,
    classify_answer,
    load_poison_maps_from_jsonl,
)
from manifest_utils import apply_manifest, load_manifest, validate_selected_questions
from official_eval import print_evaluation_report
from run_metadata import build_run_metadata, write_run_metadata
from stress_protocols import (
    oracle_gold_docs,
    build_forced_exposure_context,
    forced_gold_positions_for_dataset,
)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_questions(hotpot_path: str, nq_path: str, wiki2_path: str = "") -> List[dict]:
    questions = []
    if hotpot_path:
        with open(hotpot_path) as f:
            for line in f:
                d = json.loads(line)
                questions.append({"id": str(d["id"]), "q": d["question"], "a": [d["answer"]], "ds": "hotpot"})
    if nq_path:
        with open(nq_path) as f:
            for line in f:
                d = json.loads(line)
                ans = d.get("answer", [])
                if isinstance(ans, str):
                    ans = [ans]
                questions.append({"id": str(d["id"]), "q": d["question"], "a": ans, "ds": "nq"})
    if wiki2_path:
        with open(wiki2_path) as f:
            for line in f:
                d = json.loads(line)
                ans = d.get("golden_answers", d.get("answer", []))
                if isinstance(ans, str):
                    ans = [ans]
                questions.append({"id": str(d["id"]), "q": d["question"], "a": ans, "ds": "wiki2"})
    return questions


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_docs(searcher: LuceneSearcher, question: str, k: int) -> List[str]:
    hits = searcher.search(question, k=k)
    docs = []
    for h in hits:
        try:
            raw = json.loads(searcher.doc(h.docid).raw())
            docs.append(raw.get("contents", ""))
        except Exception:
            continue
    return docs


# ---------------------------------------------------------------------------
# FiD reader
# ---------------------------------------------------------------------------

def fid_answer(
    model,
    tokenizer,
    question: str,
    passages: List[str],
    n_passages: int = 100,
    max_length: int = 50,
    device: str = "cuda",
) -> str:
    """Run FiD inference for a single question."""
    passages = [p for p in passages if p and p.strip()][:n_passages]
    if not passages:
        return "Unknown"
    inputs = [f"question: {question} title: passage context: {p}" for p in passages]
    encoding = tokenizer(
        inputs,
        max_length=250,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].unsqueeze(0).to(device)
    attention_mask = encoding["attention_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Per-track answer evaluation helpers
# ---------------------------------------------------------------------------

def eval_row(answer_raw: str, golds: List[str], poison_targets: List[str]) -> dict:
    """Compute canonical eval fields from a raw answer string."""
    answer_eval  = eval_answer_from_raw(answer_raw)
    answer_final = answer_eval
    abstain      = int(abstain_from_eval(answer_eval))
    conflict     = int(conflict_flag_from_text(answer_raw, golds, poison_targets))
    result       = classify_answer(answer_eval, answer_raw, golds, poison_targets,
                                   explicit_abstain=bool(abstain))
    em  = 1 if result["label"] == "gold"   else 0
    asr = 1 if result["label"] == "target" else 0
    return {
        "answer_raw":   answer_raw,
        "answer_eval":  answer_eval,
        "answer_final": answer_final,
        "abstain":      abstain,
        "conflict_flag": conflict,
        "em":           em,
        "asr":          asr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FiD external baseline evaluation (canonical)")
    # --- Canonical args ---
    parser.add_argument("--manifest",            default="",   help="Fixed manifest JSON")
    parser.add_argument("--frozen_poison_jsonl", default="",   help="Frozen QC JSONL artifact")
    # --- Backward-compat aliases (hidden) ---
    parser.add_argument("--question_manifest",   dest="manifest",            default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)
    parser.add_argument("--poison_jsonl",        dest="frozen_poison_jsonl", default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)
    # --- Indexes ---
    parser.add_argument("--base_index",   required=True, help="BM25 base index (clean + forced gold retrieval)")
    parser.add_argument("--attack_index", default="",    help="BM25 attack index (unused; kept for API compat)")
    # --- Datasets ---
    parser.add_argument("--hotpot", default="")
    parser.add_argument("--nq",     default="")
    parser.add_argument("--wiki2",  default="")
    # --- Model ---
    parser.add_argument("--fid_model_path", required=True, help="Path to FiD checkpoint dir")
    # --- Output ---
    parser.add_argument("--output_csv",      required=True)
    parser.add_argument("--output_json",     default="")
    parser.add_argument("--overwrite_output", action="store_true")
    # --- Track params ---
    parser.add_argument("--tracks", nargs="+", default=["base", "attack"],
                        choices=["base", "attack", "forced"])
    parser.add_argument("--recall_k",       type=int, default=100,  help="BM25 recall k for base/attack")
    parser.add_argument("--n_passages",     type=int, default=100,  help="Passages fed to FiD for base/attack")
    parser.add_argument("--forced_k",       type=int, default=4,    help="Total slots for Forced Exposure context")
    parser.add_argument("--poison_per_query", type=int, default=6)
    args = parser.parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite_output:
        raise SystemExit(f"[!] output_csv already exists: {args.output_csv}. Pass --overwrite_output.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load FiD model ---
    print(f"[fid] loading model from {args.fid_model_path}")
    from transformers import T5Config
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    config    = T5Config.from_pretrained(args.fid_model_path)
    t5        = fid_model.FiDT5(config)
    state_dict = torch.load(os.path.join(args.fid_model_path, "pytorch_model.bin"), map_location="cpu")
    t5.load_state_dict(state_dict, strict=False)
    t5 = t5.to(device).eval()
    print(f"[fid] model loaded on {device}")

    # --- Load indexes ---
    base_searcher = LuceneSearcher(args.base_index)
    print("[fid] BM25 index loaded")

    # --- Load poison + manifest ---
    poison_map: Dict[Tuple[str, str], List[str]] = {}
    poison_target_map: Dict[Tuple[str, str], List[str]] = {}
    if args.frozen_poison_jsonl and os.path.exists(args.frozen_poison_jsonl):
        poison_map, poison_target_map = load_poison_maps_from_jsonl(args.frozen_poison_jsonl)
        print(f"[fid] poison map: {len(poison_map)} keys")

    questions = load_questions(args.hotpot, args.nq, args.wiki2)
    manifest = None
    if args.manifest:
        manifest  = load_manifest(args.manifest)
        questions = apply_manifest(questions, manifest)
        validate_selected_questions(questions, manifest)
        print(f"[fid] manifest: {manifest.get('name')} ({len(questions)} questions)")
    print(f"[fid] questions: {len(questions)}, tracks: {args.tracks}")

    # --- Field names ---
    base_fields    = ["base_answer_raw", "base_answer_eval", "base_answer_final",
                      "base_abstain", "base_conflict_flag", "base_em", "base_asr"]
    attack_fields  = ["attack_answer_raw", "attack_answer_eval", "attack_answer_final",
                      "attack_abstain", "attack_conflict_flag", "attack_em", "attack_asr",
                      "attack_poison_ratio"]
    forced_fields  = ["forced_answer_raw", "forced_answer_eval", "forced_answer_final",
                      "forced_abstain", "forced_conflict_flag", "forced_em", "forced_asr",
                      "forced_gold_positions", "forced_gold_count", "forced_poison_count",
                      "forced_context_len"]

    fieldnames = ["ds", "id", "question", "gold_answer", "poison_target"]
    if "base"   in args.tracks: fieldnames += base_fields
    if "attack" in args.tracks: fieldnames += attack_fields
    if "forced" in args.tracks: fieldnames += forced_fields

    with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i, item in enumerate(questions):
            q, qid, ds, golds = item["q"], item["id"], item["ds"], item["a"]
            key = (ds, qid)
            poison_docs    = poison_map.get(key, [])[:args.poison_per_query]
            poison_targets = poison_target_map.get(key, [])
            poison_target_str = " | ".join(poison_targets)

            row: dict = {
                "ds": ds, "id": qid, "question": q,
                "gold_answer": " | ".join(golds),
                "poison_target": poison_target_str,
            }

            # --- BASE ---
            if "base" in args.tracks:
                b_docs   = retrieve_docs(base_searcher, q, args.recall_k)
                base_raw = fid_answer(t5, tokenizer, q, b_docs, args.n_passages, device=device)
                ev = eval_row(base_raw, golds, poison_targets)
                for k_ev, v in ev.items():
                    row[f"base_{k_ev}"] = v

            # --- ATTACK ---
            if "attack" in args.tracks:
                a_docs = retrieve_docs(base_searcher, q, args.recall_k)
                if poison_docs:
                    a_docs = [p.strip() for p in poison_docs if p.strip()] + a_docs
                    a_docs = a_docs[:args.recall_k]
                poison_ratio = len(poison_docs) / max(len(a_docs), 1)
                attack_raw = fid_answer(t5, tokenizer, q, a_docs, args.n_passages, device=device)
                ev = eval_row(attack_raw, golds, poison_targets)
                for k_ev, v in ev.items():
                    row[f"attack_{k_ev}"] = v
                row["attack_poison_ratio"] = round(poison_ratio, 4)

            # --- FORCED EXPOSURE (canonical: gold fixed positions, remaining = sybil fill) ---
            if "forced" in args.tracks:
                gold_docs      = oracle_gold_docs(base_searcher, golds, k=20, max_docs=2)
                gold_positions = forced_gold_positions_for_dataset(ds, total_k=args.forced_k)
                context_str    = build_forced_exposure_context(
                    gold_docs=gold_docs,
                    poison_docs=poison_docs,
                    total_slots=args.forced_k,
                    gold_positions=gold_positions,
                    per_doc_chars=900,
                )
                # FiD expects a list of passages, not a joined string
                forced_passages = [seg for seg in context_str.split("\n\n") if seg.strip()]
                forced_raw = fid_answer(t5, tokenizer, q, forced_passages,
                                        n_passages=args.forced_k, device=device)
                ev = eval_row(forced_raw, golds, poison_targets)
                for k_ev, v in ev.items():
                    row[f"forced_{k_ev}"] = v
                row["forced_gold_positions"]  = json.dumps(gold_positions)
                row["forced_gold_count"]      = len(gold_docs)
                row["forced_poison_count"]    = len(poison_docs)
                row["forced_context_len"]     = len(context_str)

            writer.writerow(row)
            csvfile.flush()

            if (i + 1) % 10 == 0 or (i + 1) == len(questions):
                print(f"[fid] {i+1}/{len(questions)}", flush=True)

    print(f"[fid] saved → {args.output_csv}")

    # --- Canonical evaluator ---
    active_prefixes = [t for t in ["base", "attack", "forced"] if t in args.tracks]
    print_evaluation_report(args.output_csv, system_prefixes=active_prefixes, strict_schema=True)

    # --- Run metadata ---
    run_meta = build_run_metadata(
        workspace_root=_ROOT,
        manifest=manifest,
        base_index=args.base_index,
        attack_index=getattr(args, "attack_index", ""),
        poison_jsonl=args.frozen_poison_jsonl,
        llm_backend="hf_local",
        llm_model_id=args.fid_model_path,
        llm_base_url="",
        llm_temperature=0.0,
        extra_fields={
            "fid_tracks": " ".join(args.tracks),
            "recall_k": args.recall_k,
            "n_passages": args.n_passages,
            "forced_k": args.forced_k,
            "poison_per_query": args.poison_per_query,
            "n_questions": len(questions),
        },
    )
    write_run_metadata(args.output_csv, run_meta)

    print("[fid] done.")


if __name__ == "__main__":
    main()
