#!/usr/bin/env python3
"""
forced_reference_pilot.py — Gold-only & Distractor forced reference experiments.

Two conditions (same 200Q):
  gold_only:   4 slots all gold docs (no sybil) — reader ceiling
  distractor:  4 slots distractor docs (topically related but no gold answer)

Usage:
  python3 forced_reference_pilot.py \
    --manifest $RT/pilot_balanced_200_seed42.json \
    --frozen_poison_jsonl $PT/_not_used/poison_docs_llm.jsonl \
    --base_index $ROOT/wiki_indexes/wikipedia-dpr-100w \
    --hotpot $ROOT/datasets/hotpotqa_fixed.jsonl \
    --nq $ROOT/datasets/nq_fixed.jsonl \
    --output_csv $RT/forced_ref_200.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import List

from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "benchmark_core"))
sys.path.insert(0, _ROOT)

from benchmark_eval_utils import (
    eval_answer_from_raw,
    abstain_from_eval,
    conflict_flag_from_text,
    load_poison_maps_from_jsonl,
    classify_answer,
)
from manifest_utils import load_manifest, apply_manifest, validate_selected_questions
from inference_backends import LLMBackendConfig, complete_text
from llm_answering import build_qa_prompt

sys.path.insert(0, os.path.join(_ROOT, "stress_protocols"))
from stress_protocols import oracle_gold_docs


def load_questions(hotpot_path: str, nq_path: str) -> List[dict]:
    questions = []
    if hotpot_path:
        with open(hotpot_path) as f:
            for line in f:
                d = json.loads(line)
                questions.append({"id": str(d["id"]), "q": d["question"],
                                   "a": [d["answer"]], "ds": "hotpot"})
    if nq_path:
        with open(nq_path) as f:
            for line in f:
                d = json.loads(line)
                ans = d.get("answer", [])
                if isinstance(ans, str):
                    ans = [ans]
                questions.append({"id": str(d["id"]), "q": d["question"],
                                   "a": ans, "ds": "nq"})
    return questions


def get_distractor_docs(searcher, question: str, golds: List[str],
                        k: int = 30, max_docs: int = 4) -> List[str]:
    """BM25 search for question, exclude passages containing gold answer."""
    hits = searcher.search(question, k=k)
    distractors = []
    for h in hits:
        try:
            raw = json.loads(searcher.doc(h.docid).raw())
            text = raw.get("contents", "")
            if not text:
                continue
            # Exclude gold-bearing passages
            text_lower = text.lower()
            if any(g.lower() in text_lower for g in golds if g):
                continue
            distractors.append(text)
            if len(distractors) >= max_docs:
                break
        except Exception:
            continue
    return distractors


def main():
    parser = argparse.ArgumentParser(description="Gold-only & Distractor forced reference")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frozen_poison_jsonl", required=True)
    parser.add_argument("--base_index", required=True)
    parser.add_argument("--hotpot", default="")
    parser.add_argument("--nq", default="")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--llm_backend", default="llama_cpp_http")
    parser.add_argument("--llm_model_id", default="qwen25_7b_instruct_gguf")
    parser.add_argument("--llm_base_url", default="http://127.0.0.1:8001")
    parser.add_argument("--llm_temperature", default="0.0")
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--total_slots", type=int, default=4)
    parser.add_argument("--doc_chars", type=int, default=1024)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    questions = load_questions(args.hotpot, args.nq)
    questions = apply_manifest(questions, manifest)
    validate_selected_questions(questions, manifest)
    if args.max_questions > 0:
        questions = questions[:args.max_questions]
    print(f"[forced_ref] {len(questions)} questions", flush=True)

    poison_map, poison_target_map = load_poison_maps_from_jsonl(args.frozen_poison_jsonl)

    from pyserini.search.lucene import LuceneSearcher
    lucene = LuceneSearcher(args.base_index)
    print(f"[forced_ref] Lucene: {lucene.num_docs:,} docs", flush=True)

    config = LLMBackendConfig(
        backend=args.llm_backend,
        base_url=args.llm_base_url.rstrip("/"),
        model_id=args.llm_model_id,
        api_key_env="OPENAI_API_KEY",
        temperature=float(args.llm_temperature),
        max_tokens=64,
        timeout=90,
    )

    conditions = ["gold_only", "distractor"]
    fieldnames = [
        "condition", "ds", "id", "question", "gold_answer", "poison_target",
        "answer_raw", "answer_eval", "answer_final", "abstain", "conflict",
        "label", "n_gold_docs", "n_distractor_docs",
    ]

    rows = []
    for cond in conditions:
        print(f"\n[forced_ref] === {cond} ===", flush=True)
        for qi, q in enumerate(tqdm(questions, desc=f"forced/{cond}")):
            ds, qid, question, golds = q["ds"], q["id"], q["q"], q["a"]
            key = (ds, qid)
            poison_targets = poison_target_map.get(key, [])
            poison_target = poison_targets[0] if poison_targets else ""

            if cond == "gold_only":
                # Fill all slots with gold docs
                gold_docs = oracle_gold_docs(lucene, golds, k=30,
                                              max_docs=args.total_slots)
                context_parts = [d[:args.doc_chars] for d in gold_docs]
                context = "\n\n".join(context_parts)
                n_gold = len(gold_docs)
                n_dist = 0
            else:  # distractor
                dist_docs = get_distractor_docs(lucene, question, golds,
                                                 k=30, max_docs=args.total_slots)
                context_parts = [d[:args.doc_chars] for d in dist_docs]
                context = "\n\n".join(context_parts)
                n_gold = 0
                n_dist = len(dist_docs)

            prompt = build_qa_prompt(context, question)
            try:
                answer_raw = complete_text(prompt, config) or "[ERROR: Empty]"
            except Exception as e:
                answer_raw = f"[ERROR: {e}]"

            answer_eval = eval_answer_from_raw(answer_raw)
            answer_final = answer_eval
            abstain = int(abstain_from_eval(answer_eval))
            conflict = int(conflict_flag_from_text(answer_raw, golds, poison_targets))

            result = classify_answer(
                eval_answer=answer_final,
                extracted_text_for_conflict=answer_raw,
                gold_aliases=golds,
                target_aliases=poison_targets,
                explicit_abstain=abstain,
            )
            label = result["label"]

            rows.append({
                "condition": cond,
                "ds": ds,
                "id": qid,
                "question": question,
                "gold_answer": " | ".join(golds),
                "poison_target": poison_target,
                "answer_raw": answer_raw,
                "answer_eval": answer_eval,
                "answer_final": answer_final,
                "abstain": abstain,
                "conflict": conflict,
                "label": label,
                "n_gold_docs": n_gold,
                "n_distractor_docs": n_dist,
            })

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("  Forced Reference Summary")
    print("=" * 70)
    for cond in conditions:
        cdf = df[df["condition"] == cond]
        print(f"\n  [{cond}]")
        for scope in ["all", "hotpot", "nq"]:
            sub = cdf if scope == "all" else cdf[cdf["ds"] == scope]
            n = len(sub)
            acc = (sub["label"] == "gold").mean() * 100
            abstain_r = (sub["label"] == "abstain").mean() * 100
            drift = (sub["label"] == "drift").mean() * 100
            print(f"    {scope:6} n={n:3d}  ACC={acc:5.1f}%  "
                  f"Abstain={abstain_r:5.1f}%  Drift={drift:5.1f}%")

    print(f"\n[forced_ref] done → {args.output_csv}")


if __name__ == "__main__":
    main()
