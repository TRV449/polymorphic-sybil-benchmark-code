#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List

from tqdm import tqdm

# Path setup: file lives in stress_protocols/; modules are in pilot_tools/ root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT, os.path.join(_ROOT, "benchmark_core"), os.path.dirname(os.path.abspath(__file__))]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pyserini.search.lucene import LuceneSearcher

from llm_answering import get_anchor_answer, get_llama_answer_common, check_em, BASE_READER_PROMPT
from benchmark_eval_utils import (
    eval_answer_from_raw,
    abstain_from_eval,
    conflict_flag_from_text,
    load_poison_maps_from_jsonl,
)
from manifest_utils import load_manifest, apply_manifest, validate_selected_questions
from official_eval import print_evaluation_report
from run_metadata import build_run_metadata, write_run_metadata
from stress_protocols import (
    oracle_gold_docs,
    build_forced_exposure_context,
    forced_gold_positions_for_dataset,
)


def load_questions(hotpot_path: str, nq_path: str, wiki2_path: str = "") -> List[dict]:
    questions = []
    if hotpot_path:
        with open(hotpot_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                questions.append({"id": str(d["id"]), "q": d["question"], "a": [d["answer"]], "ds": "hotpot"})
    if nq_path:
        with open(nq_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ans = d.get("answer", [])
                if isinstance(ans, str):
                    ans = [ans]
                questions.append({"id": str(d["id"]), "q": d["question"], "a": ans, "ds": "nq"})
    if wiki2_path:
        with open(wiki2_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ans = d.get("golden_answers", d.get("answer", []))
                if isinstance(ans, str):
                    ans = [ans]
                questions.append({"id": str(d["id"]), "q": d["question"], "a": ans, "ds": "wiki2"})
    return questions




def pseudo_oracle_answer_match_context(searcher: LuceneSearcher, golds: List[str], k: int = 20) -> str:
    for gold in golds:
        hits = searcher.search(gold, k=k)
        for h in hits:
            try:
                raw = json.loads(searcher.doc(h.docid).raw())
                text = raw.get("contents", "")
                if gold.lower() in text.lower():
                    return text[:1400]
            except Exception:
                continue
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-oracle/control benchmark tracks")
    parser.add_argument("--base_index", required=True)
    parser.add_argument("--hotpot", default="")
    parser.add_argument("--nq", default="")
    parser.add_argument("--wiki2", default="", help="2WikiMultiHopQA JSONL (id, question, golden_answers[])")
    parser.add_argument("--poison_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--track",
        required=True,
        choices=[
            "parametric_only",
            "oracle_gold_only",
            "oracle_gold_plus_poison",
            "pseudo_oracle_answer_match",
            "pseudo_oracle_answer_match_plus_poison",
            "answer_match_pseudo_oracle_gold",
            "answer_match_pseudo_oracle_gold_plus_poison",
            "forced_exposure_gold_fixed",
        ],
    )
    parser.add_argument("--question_manifest", type=str, default="")
    parser.add_argument("--llama_url", type=str, default="http://127.0.0.1:8000/completion")
    parser.add_argument("--llm_backend", type=str, default=os.environ.get("LLM_BACKEND", "llama_cpp_http"))
    parser.add_argument("--llm_model_id", type=str, default=os.environ.get("LLM_MODEL_ID", ""))
    parser.add_argument("--llm_base_url", type=str, default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--llm_temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0.0")))
    parser.add_argument("--n_predict", type=int, default=64)
    parser.add_argument("--forced_k", type=int, default=4)
    args = parser.parse_args()
    track_alias = {
        "oracle_gold_only": "answer_match_pseudo_oracle_gold",
        "oracle_gold_plus_poison": "answer_match_pseudo_oracle_gold_plus_poison",
        "pseudo_oracle_answer_match": "answer_match_pseudo_oracle_gold",
        "pseudo_oracle_answer_match_plus_poison": "answer_match_pseudo_oracle_gold_plus_poison",
    }
    canonical_track = track_alias.get(args.track, args.track)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    llm_url = args.llm_base_url or args.llama_url
    manifest = None
    questions = load_questions(args.hotpot, args.nq, wiki2_path=args.wiki2)
    if args.question_manifest:
        manifest = load_manifest(args.question_manifest)
        source_label = " + ".join(p for p in [args.hotpot, args.nq, args.wiki2] if p)
        questions = apply_manifest(questions, manifest, source_label=source_label)
        validate_selected_questions(questions, manifest, source_label=source_label)
    poison_docs_map, poison_target_map = load_poison_maps_from_jsonl(args.poison_jsonl)
    searcher = LuceneSearcher(args.base_index)
    run_meta = build_run_metadata(
        workspace_root=os.path.dirname(os.path.abspath(__file__)),
        manifest=manifest,
        base_index=args.base_index,
        poison_jsonl=args.poison_jsonl,
        prompt_text=BASE_READER_PROMPT,
        llm_backend=args.llm_backend,
        llm_model_id=args.llm_model_id,
        llm_base_url=llm_url,
        llm_temperature=args.llm_temperature,
        n_predict=args.n_predict,
        extra_fields={"oracle_control_track": canonical_track},
    )

    fieldnames = [
        "ds", "id", "question", "gold_answer", "poison_target", "benchmark_track",
        "manifest_name", "manifest_selection_sha256", "source_hotpot_sha256", "source_nq_sha256",
        "corpus_hash", "index_hash", "faiss_index_hash", "attack_index_hash", "poison_jsonl_sha256",
        "poison_generator_version", "poison_generator_prompt_hash", "git_commit", "prompt_hash",
        "generator_model_id", "generator_model_family", "generator_model_provider", "generator_model_date",
        "llm_backend", "llm_base_url", "llm_temperature", "run_n_predict",
        "oracle_answer_raw", "oracle_answer_eval", "oracle_answer_final", "oracle_best_raw",
        "oracle_em", "oracle_abstain", "oracle_conflict_flag", "oracle_context_len", "oracle_control_track",
        "gold_context_found", "poison_context_injected", "forced_exposure_used",
        "forced_track_variant", "forced_k", "forced_gold_positions", "forced_gold_count", "forced_poison_count", "forced_context_len", "forced_gold_complete",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for item in tqdm(questions, desc=f"Oracle({canonical_track})"):
            q = item["q"]
            golds = item["a"]
            key = (item["ds"], str(item["id"]))
            poison_docs = poison_docs_map.get(key, [])
            poison_targets = poison_target_map.get(key, [])
            poison_target = " | ".join(str(x) for x in poison_targets if x)
            if canonical_track == "parametric_only":
                raw = get_anchor_answer(
                    q,
                    url=llm_url,
                    n_predict=args.n_predict,
                    backend=args.llm_backend,
                    model_id=args.llm_model_id,
                    temperature=args.llm_temperature,
                ) or "Unknown"
                context = ""
                gold_context_found = False
                poison_context_injected = False
                forced_exposure_used = False
                forced_track_variant = ""
                forced_gold_positions = []
                forced_gold_count = 0
                forced_poison_count = 0
                forced_gold_complete = ""
            else:
                if canonical_track == "forced_exposure_gold_fixed":
                    gold_docs = oracle_gold_docs(searcher, golds, max_docs=2)
                    forced_gold_positions = forced_gold_positions_for_dataset(item["ds"], args.forced_k)
                    gold_context_found = bool(gold_docs)
                    context = build_forced_exposure_context(
                        gold_docs=gold_docs,
                        poison_docs=poison_docs,
                        total_slots=args.forced_k,
                        gold_positions=forced_gold_positions,
                    )
                    forced_gold_count = min(len(gold_docs), len(forced_gold_positions))
                    forced_poison_count = min(len(poison_docs), max(0, args.forced_k - forced_gold_count))
                    poison_context_injected = bool(poison_docs[: max(0, args.forced_k - forced_gold_count)])
                    forced_exposure_used = True
                    forced_track_variant = f"{item['ds']}_gold_fixed"
                    forced_gold_complete = forced_gold_count == len(forced_gold_positions)
                else:
                    gold_context = pseudo_oracle_answer_match_context(searcher, golds)
                    gold_context_found = bool(gold_context)
                    if canonical_track == "answer_match_pseudo_oracle_gold_plus_poison" and poison_docs:
                        context = f"{gold_context}\n\n{poison_docs[0][:900]}"
                        poison_context_injected = True
                    else:
                        context = gold_context
                        poison_context_injected = False
                    forced_exposure_used = False
                    forced_track_variant = ""
                    forced_gold_positions = []
                    forced_gold_count = 0
                    forced_poison_count = 0
                    forced_gold_complete = ""
                raw = get_llama_answer_common(
                    context=context,
                    question=q,
                    url=llm_url,
                    n_predict=args.n_predict,
                    backend=args.llm_backend,
                    model_id=args.llm_model_id,
                    temperature=args.llm_temperature,
                ) if context else "Unknown"
            ans_eval = eval_answer_from_raw(raw)
            oracle_abstain = abstain_from_eval(ans_eval)
            oracle_conflict_flag = conflict_flag_from_text(raw, golds, poison_targets)
            row = {
                "ds": item["ds"],
                "id": item["id"],
                "question": q,
                "gold_answer": " | ".join(golds),
                "poison_target": poison_target,
                "benchmark_track": "oracle_control",
                "manifest_name": run_meta["manifest_name"],
                "manifest_selection_sha256": run_meta["manifest_selection_sha256"],
                "source_hotpot_sha256": run_meta["source_hotpot_sha256"],
                "source_nq_sha256": run_meta["source_nq_sha256"],
                "corpus_hash": run_meta["corpus_hash"],
                "index_hash": run_meta["index_hash"],
                "faiss_index_hash": run_meta["faiss_index_hash"],
                "attack_index_hash": run_meta["attack_index_hash"],
                "poison_jsonl_sha256": run_meta["poison_jsonl_sha256"],
                "poison_generator_version": run_meta["poison_generator_version"],
                "poison_generator_prompt_hash": run_meta["poison_generator_prompt_hash"],
                "git_commit": run_meta["git_commit"],
                "prompt_hash": run_meta["prompt_hash"],
                "generator_model_id": run_meta["generator_model_id"],
                "generator_model_family": run_meta["generator_model_family"],
                "generator_model_provider": run_meta["generator_model_provider"],
                "generator_model_date": run_meta["generator_model_date"],
                "llm_backend": run_meta["llm_backend"],
                "llm_base_url": run_meta["llm_base_url"],
                "llm_temperature": run_meta["llm_temperature"],
                "run_n_predict": run_meta["n_predict"],
                "oracle_answer_raw": raw,
                "oracle_answer_eval": ans_eval,
                "oracle_answer_final": ans_eval,
                "oracle_best_raw": raw,
                "oracle_em": 1 if check_em(ans_eval, golds) else 0,
                "oracle_abstain": oracle_abstain,
                "oracle_conflict_flag": int(oracle_conflict_flag),
                "oracle_context_len": len(context),
                "oracle_control_track": canonical_track,
                "gold_context_found": gold_context_found,
                "poison_context_injected": poison_context_injected,
                "forced_exposure_used": forced_exposure_used,
                "forced_track_variant": forced_track_variant,
                "forced_k": args.forced_k if forced_exposure_used else "",
                "forced_gold_positions": json.dumps(forced_gold_positions) if forced_exposure_used else "",
                "forced_gold_count": forced_gold_count if forced_exposure_used else "",
                "forced_poison_count": forced_poison_count if forced_exposure_used else "",
                "forced_context_len": len(context) if forced_exposure_used else "",
                "forced_gold_complete": forced_gold_complete if forced_exposure_used else "",
            }
            writer.writerow(row)

    write_run_metadata(args.output_csv, run_meta)
    print_evaluation_report(
        args.output_csv,
        system_prefixes=["oracle"],
        strict_schema=True,
    )


if __name__ == "__main__":
    main()
