#!/usr/bin/env python3
"""
colbert_qwen_runner.py — ColBERTv2 reranking + Qwen2.5 reader baseline.

Strategy: BM25 top-1000 (pyserini) → ColBERTv2 MaxSim rerank → top-k → Qwen2.5 reader

No full ColBERT index required (21M docs would be 100-200GB).
ColBERTv2 checkpoint: colbert-ir/colbertv2.0 (~450MB, BERT-based late interaction).

Tracks: clean | attack | forced

Usage:
  python3 colbert_qwen_runner.py \\
    --manifest $PT/benchmark_package/fixed_splits/public_all_500_balanced_seed42.json \\
    --frozen_poison_jsonl $RT/poison_docs_llm.frozen_qc.jsonl \\
    --track attack \\
    --output_csv $RT/colbert_qwen_attack_500.csv \\
    --base_index $ROOT/wiki_indexes/wikipedia-dpr-100w \\
    --colbert_model colbert-ir/colbertv2.0 \\
    --llm_backend llama_cpp_http \\
    --llm_model_id qwen25_7b_instruct_gguf \\
    --llm_base_url http://127.0.0.1:8001

Output schema (per row):
  ds, id, question, gold_answer, poison_target,
  colbert_answer_raw, colbert_answer_eval, colbert_answer_final,
  colbert_abstain, colbert_conflict_flag
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List, Optional

import torch
from tqdm import tqdm

# --- Benchmark core imports (path adjustment for subdir) ---
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "benchmark_core"))
sys.path.insert(0, _ROOT)

from benchmark_eval_utils import (
    eval_answer_from_raw,
    abstain_from_eval,
    conflict_flag_from_text,
    load_poison_maps_from_jsonl,
)
from manifest_utils import load_manifest, apply_manifest, validate_selected_questions
from official_eval import print_evaluation_report
from run_metadata import build_run_metadata, write_run_metadata

# stress_protocols for forced track
sys.path.insert(0, os.path.join(_ROOT, "stress_protocols"))
from stress_protocols import (
    oracle_gold_docs,
    build_forced_exposure_context,
    forced_gold_positions_for_dataset,
)

PREFIX = "colbert"

# ---------------------------------------------------------------------------
# ColBERT model (singleton, loaded once)
# ---------------------------------------------------------------------------

_colbert_ckpt = None
_colbert_cfg  = None

def load_colbert(model_path: str, doc_maxlen: int = 180, query_maxlen: int = 32):
    """Load ColBERTv2 checkpoint (once per process)."""
    global _colbert_ckpt, _colbert_cfg
    if _colbert_ckpt is not None:
        return _colbert_ckpt, _colbert_cfg
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.infra import ColBERTConfig
    _colbert_cfg  = ColBERTConfig(checkpoint=model_path, doc_maxlen=doc_maxlen, query_maxlen=query_maxlen)
    _colbert_ckpt = Checkpoint(model_path, colbert_config=_colbert_cfg)
    _colbert_ckpt.eval()
    return _colbert_ckpt, _colbert_cfg


def colbert_maxsim(Q: "torch.Tensor", D: "torch.Tensor") -> "torch.Tensor":
    """ColBERT MaxSim: sum over query tokens of max doc-token similarity.

    Q: [1, q_len, dim]  float32
    D: [N, d_len, dim]  float32
    Returns scores [N] float32
    """
    sim = torch.einsum("qld,nmd->qnlm", Q, D)   # [1, N, q_len, d_len]
    return sim.max(dim=-1).values.sum(dim=-1).squeeze(0)   # [N]


def colbert_rerank(
    ckpt,
    question: str,
    passages: List[str],
    top_k: int = 10,
    batch_size: int = 32,
) -> List[str]:
    """Rerank `passages` with ColBERTv2 MaxSim and return top_k."""
    if not passages:
        return []
    with torch.no_grad():
        Q, = (ckpt.queryFromText([question], bsize=1, to_cpu=True),)  # [1, q_len, 128]
        Q = Q.float()
        all_scores = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            D, = ckpt.docFromText(batch, bsize=batch_size, keep_dims=True, to_cpu=True)
            D = D.float()   # [B, d_len, 128]
            scores = colbert_maxsim(Q, D)
            all_scores.extend(scores.tolist())
    ranked = sorted(zip(all_scores, passages), reverse=True)
    return [p for _, p in ranked[:top_k]]


# ---------------------------------------------------------------------------
# BM25 retrieve (pyserini)
# ---------------------------------------------------------------------------

def bm25_retrieve(searcher, question: str, k: int = 1000) -> List[str]:
    """BM25 top-k from pyserini Lucene index."""
    import json as _json
    hits = searcher.search(question, k=k)
    docs = []
    for h in hits:
        try:
            raw = _json.loads(searcher.doc(h.docid).raw())
            text = raw.get("contents", "")
            if text:
                docs.append(text)
        except Exception:
            continue
    return docs


def bm25_2hop_retrieve(searcher, question: str, k_per_hop: int = 500) -> List[str]:
    """BM25 2-hop retrieval for HotpotQA.

    Step 1: BM25(question) → top-k passages.
    Step 2: Use text of top passage as bridge query → BM25(bridge) → k more passages.
    Returns merged deduplicated list (hop1 first, then hop2 extras).
    """
    hop1 = bm25_retrieve(searcher, question, k=k_per_hop)
    if not hop1:
        return []
    # Bridge: first 200 chars of top passage as second query
    bridge_query = hop1[0][:200].strip()
    hop2 = bm25_retrieve(searcher, bridge_query, k=k_per_hop) if bridge_query else []
    # Merge, deduplicate by first 100 chars
    seen: set = set()
    merged: List[str] = []
    for p in hop1 + hop2:
        key = p[:100]
        if key not in seen:
            seen.add(key)
            merged.append(p)
    return merged


# ---------------------------------------------------------------------------
# Reader (Qwen2.5 via llm_answering — uses COMMON_QA_SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

def get_answer(question: str, context: str, args) -> str:
    from llm_answering import get_llama_answer_common
    return get_llama_answer_common(
        context=context,
        question=question,
        backend=args.llm_backend,
        model_id=args.llm_model_id,
        url=args.llm_base_url,
        temperature=float(args.llm_temperature),
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ColBERTv2 reranking + Qwen2.5 reader baseline")
    # Shared mandatory args (same across all runners)
    parser.add_argument("--manifest",             required=True, help="Fixed manifest JSON")
    parser.add_argument("--frozen_poison_jsonl",  required=True, help="Frozen QC JSONL artifact")
    parser.add_argument("--track",                required=True, choices=["clean", "attack", "forced"])
    parser.add_argument("--output_csv",           required=True)
    parser.add_argument("--output_json",          default="")
    # Dataset paths
    parser.add_argument("--hotpot",               default="")
    parser.add_argument("--nq",                   default="")
    parser.add_argument("--wiki2",                default="")
    # Index paths
    parser.add_argument("--base_index",           required=True, help="BM25 Lucene index for retrieval + forced gold")
    # ColBERT model
    parser.add_argument("--colbert_model",        default="colbert-ir/colbertv2.0",
                        help="HF hub ID or local path for ColBERTv2 checkpoint")
    parser.add_argument("--colbert_doc_maxlen",   type=int, default=180)
    parser.add_argument("--colbert_query_maxlen", type=int, default=32)
    parser.add_argument("--colbert_batch_size",   type=int, default=32,
                        help="Passages per batch for ColBERT encoding")
    # LLM backend
    parser.add_argument("--llm_backend",          default="llama_cpp_http")
    parser.add_argument("--llm_model_id",         default="qwen25_7b_instruct_gguf")
    parser.add_argument("--llm_base_url",         default="http://127.0.0.1:8001")
    parser.add_argument("--llm_temperature",      default="0.0")
    # Retrieval params
    parser.add_argument("--bm25_k",              type=int, default=1000, help="BM25 candidate pool size")
    parser.add_argument("--final_k",             type=int, default=10,  help="Passages fed to reader after reranking (default for all datasets)")
    parser.add_argument("--doc_chars",           type=int, default=1024, help="Chars per passage fed to reader (default for all datasets)")
    parser.add_argument("--nq_final_k",          type=int, default=0,   help="NQ override for final_k (0 = use --final_k)")
    parser.add_argument("--nq_doc_chars",        type=int, default=0,   help="NQ override for doc_chars (0 = use --doc_chars)")
    parser.add_argument("--hotpot_final_k",      type=int, default=0,   help="Hotpot override for final_k (0 = use --final_k)")
    parser.add_argument("--hotpot_doc_chars",    type=int, default=0,   help="Hotpot override for doc_chars (0 = use --doc_chars)")
    parser.add_argument("--poison_per_query",    type=int, default=6)
    parser.add_argument("--hotpot_2hop",         action="store_true",
                        help="Use BM25 2-hop retrieval for HotpotQA questions (question→top doc→bridge query)")
    parser.add_argument("--hotpot_2hop_k",       type=int, default=500,
                        help="BM25 candidates per hop when --hotpot_2hop is set")
    parser.add_argument("--overwrite_output",    action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite_output:
        print(f"[colbert] output already exists, skipping: {args.output_csv}")
        return

    # --- Load manifest & questions ---
    manifest = load_manifest(args.manifest)
    questions = load_questions(args.hotpot, args.nq, getattr(args, 'wiki2', ''))
    questions = apply_manifest(questions, manifest)
    validate_selected_questions(questions, manifest)
    print(f"[colbert] manifest: {manifest.get('name')} ({len(questions)} questions), track: {args.track}")

    # --- Load poison map ---
    poison_map, poison_target_map = load_poison_maps_from_jsonl(args.frozen_poison_jsonl)
    print(f"[colbert] poison map: {len(poison_map)} keys")

    # --- Load ColBERT model ---
    print(f"[colbert] loading ColBERTv2 from {args.colbert_model}")
    ckpt, _ = load_colbert(args.colbert_model, args.colbert_doc_maxlen, args.colbert_query_maxlen)
    print("[colbert] ColBERTv2 loaded")

    # --- Load BM25 index (used for all tracks + forced gold retrieval) ---
    from pyserini.search.lucene import LuceneSearcher
    bm25_searcher = LuceneSearcher(args.base_index)
    print(f"[colbert] BM25 index loaded: {bm25_searcher.num_docs} docs")

    # For forced track, base_searcher = bm25_searcher (same index)
    base_searcher = bm25_searcher

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    fieldnames = [
        "ds", "id", "question", "gold_answer", "poison_target",
        f"{PREFIX}_answer_raw", f"{PREFIX}_answer_eval", f"{PREFIX}_answer_final",
        f"{PREFIX}_abstain", f"{PREFIX}_conflict_flag",
    ]

    def ds_final_k(ds: str) -> int:
        if ds == "nq"     and args.nq_final_k     > 0: return args.nq_final_k
        if ds == "hotpot" and args.hotpot_final_k > 0: return args.hotpot_final_k
        return args.final_k

    def ds_doc_chars(ds: str) -> int:
        if ds == "nq"     and args.nq_doc_chars     > 0: return args.nq_doc_chars
        if ds == "hotpot" and args.hotpot_doc_chars > 0: return args.hotpot_doc_chars
        return args.doc_chars

    rows = []
    for q in tqdm(questions, desc=f"colbert/{args.track}"):
        ds, qid, question, golds = q["ds"], q["id"], q["q"], q["a"]
        key = (ds, qid)
        poison_docs    = poison_map.get(key, [])[:args.poison_per_query]
        poison_targets = poison_target_map.get(key, [])
        poison_target  = poison_targets[0] if poison_targets else ""

        final_k   = ds_final_k(ds)
        doc_chars = ds_doc_chars(ds)

        # --- Build context ---
        if args.track == "clean":
            if ds == "hotpot" and args.hotpot_2hop:
                candidates = bm25_2hop_retrieve(bm25_searcher, question, k_per_hop=args.hotpot_2hop_k)
            else:
                candidates = bm25_retrieve(bm25_searcher, question, k=args.bm25_k)
            reranked   = colbert_rerank(ckpt, question, candidates,
                                        top_k=final_k, batch_size=args.colbert_batch_size)
            context = "\n\n".join(p[:doc_chars] for p in reranked)

        elif args.track == "attack":
            if ds == "hotpot" and args.hotpot_2hop:
                candidates = bm25_2hop_retrieve(bm25_searcher, question, k_per_hop=args.hotpot_2hop_k)
            else:
                candidates = bm25_retrieve(bm25_searcher, question, k=args.bm25_k)
            # Merge poison docs into candidate pool (prepend → ColBERT sees them)
            merged   = [p.strip() for p in poison_docs if p.strip()] + candidates
            reranked = colbert_rerank(ckpt, question, merged,
                                      top_k=final_k, batch_size=args.colbert_batch_size)
            context = "\n\n".join(p[:doc_chars] for p in reranked)

        elif args.track == "forced":
            gold_docs = oracle_gold_docs(base_searcher, golds, k=20, max_docs=2)
            gold_positions = forced_gold_positions_for_dataset(ds, total_k=4)
            context = build_forced_exposure_context(
                gold_docs=gold_docs,
                poison_docs=poison_docs,
                total_slots=4,
                gold_positions=gold_positions,
                per_doc_chars=doc_chars,
            )

        # --- Generate answer ---
        try:
            answer_raw = get_answer(question, context, args)
        except Exception as e:
            answer_raw = f"ERROR: {e}"

        answer_eval  = eval_answer_from_raw(answer_raw)
        answer_final = answer_eval
        abstain      = int(abstain_from_eval(answer_eval))
        conflict     = int(conflict_flag_from_text(answer_raw, golds, poison_targets))

        rows.append({
            "ds": ds,
            "id": qid,
            "question": question,
            "gold_answer": " | ".join(golds),
            "poison_target": " | ".join(poison_targets),
            f"{PREFIX}_answer_raw":   answer_raw,
            f"{PREFIX}_answer_eval":  answer_eval,
            f"{PREFIX}_answer_final": answer_final,
            f"{PREFIX}_abstain":      abstain,
            f"{PREFIX}_conflict_flag": conflict,
        })

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print_evaluation_report(args.output_csv, system_prefixes=[PREFIX], strict_schema=True)

    meta = build_run_metadata(
        workspace_root=_ROOT,
        manifest=manifest,
        base_index=args.base_index,
        poison_jsonl=args.frozen_poison_jsonl,
        llm_backend=args.llm_backend,
        llm_model_id=args.llm_model_id,
        llm_base_url=args.llm_base_url,
        llm_temperature=float(args.llm_temperature),
        extra_fields={
            "colbert_track": args.track,
            "bm25_k": args.bm25_k,
            "final_k": args.final_k,
            "doc_chars": args.doc_chars,
            "nq_final_k": args.nq_final_k,
            "nq_doc_chars": args.nq_doc_chars,
            "hotpot_final_k": args.hotpot_final_k,
            "hotpot_doc_chars": args.hotpot_doc_chars,
            "hotpot_2hop": args.hotpot_2hop,
            "hotpot_2hop_k": args.hotpot_2hop_k,
            "poison_per_query": args.poison_per_query,
            "n_questions": len(rows),
        },
    )
    write_run_metadata(args.output_csv, meta)

    print(f"[colbert] done → {args.output_csv}")


if __name__ == "__main__":
    main()
