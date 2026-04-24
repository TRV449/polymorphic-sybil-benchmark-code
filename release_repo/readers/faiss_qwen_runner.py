#!/usr/bin/env python3
"""
faiss_qwen_runner.py — multilingual-E5-large dense retrieval + Qwen2.5 reader.

Strategy:
  Clean : FAISS top-k (batch=10 for optimal CPU throughput) → Qwen reader
          With --colbert_rerank: FAISS top-1000 → ColBERTv2 MaxSim rerank → top-k
          With --cross_encoder_rerank: FAISS top-K → cross-encoder rerank → top-k
  Attack: FAISS top-1000 + sybil E5 on-the-fly encoding → merged by cosine score
          → top-k → Qwen reader.  sybil_in_top10 flag tracks retrievability.
  Forced: BM25 oracle gold + sybil direct injection (retriever-agnostic, same
          as colbert_qwen_runner.py forced track — cross-retriever sanity check)

FAISS index: IndexFlatIP (exact cosine, normalized vectors)
  - 21,015,324 passages, dim=1024, ~86 GB RAM
  - Load time ~40s; batch=10 is the CPU throughput sweet spot (10x speedup)

Output schema matches colbert_qwen_runner.py but uses prefix "dense".
Additional diagnostic columns (for sybil retrievability analysis):
  gold_in_top10, sybil_in_top10, sybil_count_in_top10

Usage (clean pilot 200Q):
  python3 faiss_qwen_runner.py \\
    --manifest  $RT/near_balanced_3145_seed42.json \\
    --frozen_poison_jsonl $RT/poison_docs_llm.jsonl \\
    --track clean \\
    --output_csv $RT/faiss_dense_clean_200.csv \\
    --faiss_index $ROOT/faiss_index_wikipedia_dpr_100w \\
    --base_index  $ROOT/wiki_indexes/wikipedia-dpr-100w \\
    --e5_model intfloat/multilingual-e5-large \\
    --llm_backend llama_cpp_http \\
    --llm_model_id qwen25_7b_instruct_gguf \\
    --llm_base_url http://127.0.0.1:8001 \\
    --max_questions 200
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from tqdm import tqdm

# ── benchmark core path ──────────────────────────────────────────────────────
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

sys.path.insert(0, os.path.join(_ROOT, "stress_protocols"))
from stress_protocols import (
    oracle_gold_docs,
    build_forced_exposure_context,
    forced_gold_positions_for_dataset,
)

PREFIX = "dense"

FAISS_BATCH = 10   # optimal CPU throughput (10x vs single-query)

# ── ColBERT reranker (optional, loaded on demand) ────────────────────────────

_colbert_ckpt = None
_colbert_cfg  = None


def load_colbert(model_path: str, doc_maxlen: int = 180, query_maxlen: int = 32):
    """Load ColBERTv2 checkpoint (once per process)."""
    global _colbert_ckpt, _colbert_cfg
    if _colbert_ckpt is not None:
        return _colbert_ckpt
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.infra import ColBERTConfig
    import torch
    print(f"[dense] Loading ColBERTv2: {model_path} ...", flush=True)
    t0 = time.time()
    _colbert_cfg  = ColBERTConfig(checkpoint=model_path,
                                   doc_maxlen=doc_maxlen, query_maxlen=query_maxlen)
    _colbert_ckpt = Checkpoint(model_path, colbert_config=_colbert_cfg)
    _colbert_ckpt.eval()
    print(f"[dense]   ColBERTv2 loaded in {time.time()-t0:.1f}s", flush=True)
    return _colbert_ckpt


def colbert_rerank(ckpt, question: str, passages: List[str],
                   top_k: int = 10, batch_size: int = 32) -> List[str]:
    """ColBERTv2 MaxSim rerank: return top_k passages."""
    import torch
    if not passages:
        return []
    with torch.no_grad():
        Q, = (ckpt.queryFromText([question], bsize=1, to_cpu=True),)
        Q = Q.float()
        all_scores = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            D, = ckpt.docFromText(batch, bsize=batch_size, keep_dims=True, to_cpu=True)
            D = D.float()
            sim = torch.einsum("qld,nmd->qnlm", Q, D)
            scores = sim.max(dim=-1).values.sum(dim=-1).squeeze(0)
            all_scores.extend(scores.tolist())
    ranked = sorted(zip(all_scores, passages), reverse=True)
    return [p for _, p in ranked[:top_k]]

# ── Cross-encoder reranker (optional, loaded on demand) ─────────────────────

_cross_encoder = None


def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
    """Load cross-encoder reranker (once per process)."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    sys.path.insert(0, _ROOT)
    from cross_encoder_reranker import CrossEncoderReranker
    print(f"[dense] Loading cross-encoder: {model_name} ...", flush=True)
    t0 = time.time()
    _cross_encoder = CrossEncoderReranker(model_name=model_name)
    print(f"[dense]   cross-encoder loaded in {time.time()-t0:.1f}s", flush=True)
    return _cross_encoder


def cross_encoder_rerank(ce, question: str, passages: List[str],
                         top_k: int = 10) -> List[str]:
    """Cross-encoder rerank: return top_k passages (text only)."""
    if not passages:
        return []
    ranked = ce.rerank(question, passages, top_k=top_k, return_scores=True)
    return [doc for doc, _ in ranked]


# ── FAISS index + E5 model (module-level singletons) ────────────────────────

_faiss_index: Optional[faiss.Index] = None
_docid_map: Optional[List[str]] = None
_e5_model = None
_lucene_searcher = None


def load_faiss_index(index_dir: str):
    """Load FAISS index, docid map (once per process)."""
    global _faiss_index, _docid_map
    if _faiss_index is not None:
        return _faiss_index, _docid_map

    idx_path = os.path.join(index_dir, "index.faiss")
    map_path = os.path.join(index_dir, "docid_map.jsonl")
    meta_path = os.path.join(index_dir, "meta.json")

    print(f"[dense] Loading FAISS index: {idx_path} ...", flush=True)
    t0 = time.time()
    _faiss_index = faiss.read_index(idx_path)
    print(f"[dense]   {_faiss_index.ntotal:,} vectors, dim={_faiss_index.d}, "
          f"type={type(_faiss_index).__name__}, loaded in {time.time()-t0:.1f}s", flush=True)

    print(f"[dense] Loading docid map: {map_path} ...", flush=True)
    t0 = time.time()
    _docid_map = []
    with open(map_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                _docid_map.append(json.loads(line)["docid"])
    print(f"[dense]   {len(_docid_map):,} entries in {time.time()-t0:.1f}s", flush=True)

    return _faiss_index, _docid_map


def load_e5_model(model_name: str, device: str = "auto"):
    """Load E5 embedding model (once per process)."""
    global _e5_model
    if _e5_model is not None:
        return _e5_model
    from sentence_transformers import SentenceTransformer
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dense] Loading E5 model: {model_name} on {device} ...", flush=True)
    t0 = time.time()
    _e5_model = SentenceTransformer(model_name, device=device)
    print(f"[dense]   E5 loaded in {time.time()-t0:.1f}s", flush=True)
    return _e5_model


def load_lucene(index_path: str):
    """Load Lucene searcher for doc-fetch + forced-track oracle gold."""
    global _lucene_searcher
    if _lucene_searcher is not None:
        return _lucene_searcher
    from pyserini.search.lucene import LuceneSearcher
    _lucene_searcher = LuceneSearcher(index_path)
    print(f"[dense] Lucene searcher: {_lucene_searcher.num_docs:,} docs", flush=True)
    return _lucene_searcher


# ── Encoding helpers ─────────────────────────────────────────────────────────

def encode_queries(model, texts: List[str]) -> np.ndarray:
    """Encode with 'query: ' prefix, L2-normalize."""
    prefixed = [f"query: {t}" for t in texts]
    emb = model.encode(
        prefixed,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    return emb.astype(np.float32)


def encode_passages(model, texts: List[str]) -> np.ndarray:
    """Encode with 'passage: ' prefix, L2-normalize (for sybil on-the-fly)."""
    prefixed = [f"passage: {t}" for t in texts]
    emb = model.encode(
        prefixed,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    return emb.astype(np.float32)


# ── FAISS retrieval ──────────────────────────────────────────────────────────

def faiss_fetch_text(lucene, docid: str) -> str:
    """Fetch passage text from Lucene given docid."""
    try:
        raw = json.loads(lucene.doc(docid).raw())
        return raw.get("contents", "")
    except Exception:
        return ""


def faiss_search_batch(
    index: faiss.Index,
    docid_map: List[str],
    lucene,
    query_vecs: np.ndarray,   # [B, dim] normalized float32
    k: int = 10,
) -> List[List[Tuple[str, float]]]:
    """
    Batch FAISS search. Returns list of [(text, score), ...] per query.
    Uses FAISS_BATCH chunking internally for CPU throughput.
    """
    B = len(query_vecs)
    all_results: List[List[Tuple[str, float]]] = [[] for _ in range(B)]

    for start in range(0, B, FAISS_BATCH):
        chunk = query_vecs[start : start + FAISS_BATCH]
        scores, indices = index.search(chunk, k)
        for i, (sc_row, idx_row) in enumerate(zip(scores, indices)):
            qi = start + i
            results = []
            for sc, raw_idx in zip(sc_row, idx_row):
                if raw_idx < 0 or raw_idx >= len(docid_map):
                    continue
                docid = docid_map[raw_idx]
                text = faiss_fetch_text(lucene, docid)
                if text:
                    results.append((text, float(sc)))
            all_results[qi] = results

    return all_results


# ── Dataset loader (identical to colbert_qwen_runner) ───────────────────────

def load_questions(hotpot_path: str, nq_path: str, wiki2_path: str = "", trivia_path: str = "") -> List[dict]:
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
    if wiki2_path:
        with open(wiki2_path) as f:
            for line in f:
                d = json.loads(line)
                ans = d.get("golden_answers", d.get("answer", []))
                if isinstance(ans, str):
                    ans = [ans]
                questions.append({"id": str(d["id"]), "q": d["question"],
                                   "a": ans, "ds": "wiki2"})
    if trivia_path:
        with open(trivia_path) as f:
            for line in f:
                d = json.loads(line)
                ans = d.get("answer", [])
                if isinstance(ans, str):
                    ans = [ans]
                questions.append({"id": str(d["id"]), "q": d["question"],
                                   "a": ans, "ds": "trivia"})
    return questions


# ── Reader (identical to colbert_qwen_runner) ────────────────────────────────

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


# ── Answer normalization (for gold_in_top10 check) ──────────────────────────

def _norm(s: str) -> str:
    import re
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9]", " ", s)
    return " ".join(s.split())


def answer_in_text(answers: List[str], text: str) -> bool:
    t = _norm(text)
    return any(_norm(a) in t for a in answers if a)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E5-large dense retrieval + Qwen2.5 reader (3 tracks)"
    )
    # Mandatory args (same interface as colbert_qwen_runner)
    parser.add_argument("--manifest",            required=True)
    parser.add_argument("--frozen_poison_jsonl", required=True)
    parser.add_argument("--track",               required=True,
                        choices=["clean", "attack", "forced"])
    parser.add_argument("--output_csv",          required=True)
    parser.add_argument("--output_json",         default="")
    # Dataset paths
    parser.add_argument("--hotpot",              default="")
    parser.add_argument("--nq",                  default="")
    parser.add_argument("--wiki2",               default="")
    parser.add_argument("--trivia",              default="")
    # Index paths
    parser.add_argument("--faiss_index",         required=True,
                        help="Directory with index.faiss / docid_map.jsonl / meta.json")
    parser.add_argument("--base_index",          required=True,
                        help="BM25 Lucene index (doc-text fetch + forced oracle gold)")
    # E5 model
    parser.add_argument("--e5_model",            default="intfloat/multilingual-e5-large")
    parser.add_argument("--e5_device",           default="auto")
    # LLM backend (identical to colbert_qwen_runner)
    parser.add_argument("--llm_backend",         default="llama_cpp_http")
    parser.add_argument("--llm_model_id",        default="qwen25_7b_instruct_gguf")
    parser.add_argument("--llm_base_url",        default="")
    parser.add_argument("--llm_temperature",     default="0.0")
    # Retrieval params
    parser.add_argument("--faiss_k",             type=int, default=1000,
                        help="FAISS candidate pool size (attack: merges with sybils; "
                             "clean: top-faiss_k then trimmed to final_k)")
    parser.add_argument("--final_k",             type=int, default=10)
    parser.add_argument("--doc_chars",           type=int, default=1024)
    parser.add_argument("--nq_final_k",          type=int, default=0)
    parser.add_argument("--nq_doc_chars",        type=int, default=0)
    parser.add_argument("--hotpot_final_k",      type=int, default=0)
    parser.add_argument("--hotpot_doc_chars",    type=int, default=0)
    parser.add_argument("--poison_per_query",    type=int, default=6)
    # ColBERT reranking (optional: E5 top-faiss_k → ColBERT → top-final_k)
    parser.add_argument("--colbert_rerank",      action="store_true",
                        help="Apply ColBERTv2 MaxSim reranking after FAISS retrieval")
    parser.add_argument("--colbert_model",       default="colbert-ir/colbertv2.0")
    parser.add_argument("--colbert_doc_maxlen",  type=int, default=180)
    parser.add_argument("--colbert_query_maxlen",type=int, default=32)
    parser.add_argument("--colbert_batch_size",  type=int, default=32)
    # Cross-encoder reranking (optional: E5 top-faiss_k → cross-encoder → top-final_k)
    parser.add_argument("--cross_encoder_rerank", action="store_true",
                        help="Apply cross-encoder reranking after FAISS retrieval")
    parser.add_argument("--ce_model",            default="cross-encoder/ms-marco-MiniLM-L6-v2")
    # Pilot limit
    parser.add_argument("--max_questions",       type=int, default=0,
                        help="Limit to first N questions (0 = all)")
    parser.add_argument("--overwrite_output",    action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite_output:
        print(f"[dense] output already exists, skipping: {args.output_csv}")
        return

    # ── Load resources ──────────────────────────────────────────────────────
    manifest = load_manifest(args.manifest)
    questions = load_questions(args.hotpot, args.nq, args.wiki2, args.trivia)
    questions = apply_manifest(questions, manifest)
    validate_selected_questions(questions, manifest)
    if args.max_questions > 0:
        questions = questions[: args.max_questions]
    print(f"[dense] manifest: {manifest.get('name')} → {len(questions)} questions, "
          f"track: {args.track}", flush=True)

    poison_map, poison_target_map = load_poison_maps_from_jsonl(args.frozen_poison_jsonl)
    print(f"[dense] poison map: {len(poison_map)} keys", flush=True)

    idx, docid_map = load_faiss_index(args.faiss_index)
    lucene         = load_lucene(args.base_index)
    e5             = load_e5_model(args.e5_model, args.e5_device)
    if args.colbert_rerank and args.cross_encoder_rerank:
        print("[dense] ERROR: --colbert_rerank and --cross_encoder_rerank are mutually exclusive", flush=True)
        sys.exit(1)

    colbert_ckpt   = (load_colbert(args.colbert_model,
                                   args.colbert_doc_maxlen,
                                   args.colbert_query_maxlen)
                      if args.colbert_rerank else None)
    ce_reranker    = (load_cross_encoder(args.ce_model)
                      if args.cross_encoder_rerank else None)
    if args.colbert_rerank:
        print(f"[dense] E5+ColBERT mode: FAISS top-{args.faiss_k} → ColBERT rerank → top-final_k", flush=True)
    if args.cross_encoder_rerank:
        print(f"[dense] E5+CE mode: FAISS top-{args.faiss_k} → cross-encoder rerank → top-final_k", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    # ── Output columns ───────────────────────────────────────────────────────
    fieldnames = [
        "ds", "id", "question", "gold_answer", "poison_target",
        f"{PREFIX}_answer_raw", f"{PREFIX}_answer_eval", f"{PREFIX}_answer_final",
        f"{PREFIX}_abstain", f"{PREFIX}_conflict_flag",
        # diagnostic columns
        "gold_in_top10", "sybil_in_top10", "sybil_count_in_top10",
    ]

    def ds_final_k(ds: str) -> int:
        if ds == "nq"     and args.nq_final_k     > 0: return args.nq_final_k
        if ds == "hotpot" and args.hotpot_final_k > 0: return args.hotpot_final_k
        return args.final_k

    def ds_doc_chars(ds: str) -> int:
        if ds == "nq"     and args.nq_doc_chars     > 0: return args.nq_doc_chars
        if ds == "hotpot" and args.hotpot_doc_chars > 0: return args.hotpot_doc_chars
        return args.doc_chars

    # ── Pre-compute FAISS results for clean/attack (batch efficiency) ────────
    # Pre-encode all query vectors so FAISS sees batches of FAISS_BATCH.
    # Forced track skips FAISS entirely.
    faiss_cache: Dict[int, List[Tuple[str, float]]] = {}
    if args.track in ("clean", "attack"):
        print(f"[dense] Pre-encoding {len(questions)} queries with E5 ...", flush=True)
        t0 = time.time()
        q_texts = [q["q"] for q in questions]
        q_vecs = encode_queries(e5, q_texts)
        print(f"[dense]   encoded in {time.time()-t0:.1f}s", flush=True)

        faiss_k = args.faiss_k
        print(f"[dense] FAISS batch search k={faiss_k} "
              f"(batches of {FAISS_BATCH}) ...", flush=True)
        t0 = time.time()
        for batch_start in range(0, len(questions), FAISS_BATCH):
            chunk_vecs = q_vecs[batch_start : batch_start + FAISS_BATCH]
            scores, indices = idx.search(chunk_vecs, faiss_k)
            for i, (sc_row, idx_row) in enumerate(zip(scores, indices)):
                qi = batch_start + i
                results = []
                for sc, raw_idx in zip(sc_row, idx_row):
                    if raw_idx < 0 or raw_idx >= len(docid_map):
                        continue
                    docid = docid_map[raw_idx]
                    text = faiss_fetch_text(lucene, docid)
                    if text:
                        results.append((text, float(sc)))
                faiss_cache[qi] = results
            done = min(batch_start + FAISS_BATCH, len(questions))
            if done % 50 == 0 or done == len(questions):
                elapsed = time.time() - t0
                eta = elapsed / done * (len(questions) - done) if done > 0 else 0
                print(f"[dense]   [{done}/{len(questions)}] "
                      f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
        print(f"[dense] FAISS retrieval done in {time.time()-t0:.1f}s", flush=True)

    # ── Per-question processing ───────────────────────────────────────────────
    rows = []
    for qi, q in enumerate(tqdm(questions, desc=f"dense/{args.track}")):
        ds, qid, question, golds = q["ds"], q["id"], q["q"], q["a"]
        key = (ds, qid)
        poison_docs    = poison_map.get(key, [])[:args.poison_per_query]
        poison_targets = poison_target_map.get(key, [])
        poison_target  = poison_targets[0] if poison_targets else ""

        final_k   = ds_final_k(ds)
        doc_chars = ds_doc_chars(ds)

        gold_in_top10        = 0
        sybil_in_top10       = 0
        sybil_count_in_top10 = 0

        # ── Build context ────────────────────────────────────────────────────
        if args.track == "clean":
            results = faiss_cache.get(qi, [])
            candidates = [text for text, _ in results]   # full FAISS pool
            if colbert_ckpt is not None:
                # E5 top-faiss_k → ColBERT rerank → top-final_k
                top_docs = colbert_rerank(colbert_ckpt, question, candidates,
                                          top_k=final_k,
                                          batch_size=args.colbert_batch_size)
            elif ce_reranker is not None:
                # E5 top-faiss_k → cross-encoder rerank → top-final_k
                top_docs = cross_encoder_rerank(ce_reranker, question, candidates,
                                                top_k=final_k)
            else:
                top_docs = candidates[:final_k]
            # Gold-in-top-10 diagnostic
            gold_in_top10 = int(any(answer_in_text(golds, t) for t in top_docs))
            context = "\n\n".join(p[:doc_chars] for p in top_docs)

        elif args.track == "attack":
            results = faiss_cache.get(qi, [])   # [(text, score), ...] FAISS top-1000
            if poison_docs:
                # Encode sybils on-the-fly and merge by cosine similarity
                sybil_texts = [p.strip() for p in poison_docs if p.strip()]
                if sybil_texts:
                    sybil_vecs = encode_passages(e5, sybil_texts)   # [S, dim]
                    q_vec = q_vecs[qi : qi + 1]                     # [1, dim]
                    sybil_scores = (q_vec @ sybil_vecs.T).flatten().tolist()
                    sybil_results = list(zip(sybil_texts, sybil_scores))
                    merged = sorted(results + sybil_results, key=lambda x: -x[1])
                    candidates = [text for text, _ in merged]
                    sybil_set = set(t[:60] for t in sybil_texts)
                else:
                    candidates = [text for text, _ in results]
                    sybil_set = set()
            else:
                candidates = [text for text, _ in results]
                sybil_set = set()
            if colbert_ckpt is not None:
                top_docs = colbert_rerank(colbert_ckpt, question, candidates,
                                          top_k=final_k,
                                          batch_size=args.colbert_batch_size)
            elif ce_reranker is not None:
                top_docs = cross_encoder_rerank(ce_reranker, question, candidates,
                                                top_k=final_k)
            else:
                top_docs = candidates[:final_k]
            # Sybil in top-k tracking
            sybil_in_top_k = [1 for t in top_docs if t[:60] in sybil_set]
            sybil_count_in_top10 = len(sybil_in_top_k)
            sybil_in_top10 = int(sybil_count_in_top10 > 0)
            gold_in_top10 = int(any(answer_in_text(golds, t) for t in top_docs))
            context = "\n\n".join(p[:doc_chars] for p in top_docs)

        elif args.track == "forced":
            # Retriever-agnostic: oracle gold + sybil direct injection
            gold_docs = oracle_gold_docs(lucene, golds, k=20, max_docs=2)
            gold_positions = forced_gold_positions_for_dataset(ds, total_k=4)
            context = build_forced_exposure_context(
                gold_docs=gold_docs,
                poison_docs=poison_docs,
                total_slots=4,
                gold_positions=gold_positions,
                per_doc_chars=doc_chars,
            )
            gold_in_top10 = int(bool(gold_docs))  # oracle always finds gold if it exists

        # ── Reader inference ─────────────────────────────────────────────────
        try:
            answer_raw = get_answer(question, context, args)
        except Exception as e:
            answer_raw = f"ERROR: {e}"

        answer_eval  = eval_answer_from_raw(answer_raw)
        answer_final = answer_eval
        abstain      = int(abstain_from_eval(answer_eval))
        conflict     = int(conflict_flag_from_text(answer_raw, golds, poison_targets))

        rows.append({
            "ds":              ds,
            "id":              qid,
            "question":        question,
            "gold_answer":     " | ".join(golds),
            "poison_target":   " | ".join(poison_targets),
            f"{PREFIX}_answer_raw":    answer_raw,
            f"{PREFIX}_answer_eval":   answer_eval,
            f"{PREFIX}_answer_final":  answer_final,
            f"{PREFIX}_abstain":       abstain,
            f"{PREFIX}_conflict_flag": conflict,
            "gold_in_top10":           gold_in_top10,
            "sybil_in_top10":          sybil_in_top10,
            "sybil_count_in_top10":    sybil_count_in_top10,
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
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
            "dense_track":        args.track,
            "faiss_index":        args.faiss_index,
            "e5_model":           args.e5_model,
            "faiss_k":            args.faiss_k,
            "final_k":            args.final_k,
            "doc_chars":          args.doc_chars,
            "nq_final_k":         args.nq_final_k,
            "nq_doc_chars":       args.nq_doc_chars,
            "hotpot_final_k":     args.hotpot_final_k,
            "hotpot_doc_chars":   args.hotpot_doc_chars,
            "poison_per_query":   args.poison_per_query,
            "colbert_rerank":     args.colbert_rerank,
            "cross_encoder_rerank": args.cross_encoder_rerank,
            "ce_model":           args.ce_model if args.cross_encoder_rerank else None,
            "n_questions":        len(rows),
            "max_questions":      args.max_questions,
        },
    )
    write_run_metadata(args.output_csv, meta)
    print(f"[dense] done → {args.output_csv}")


if __name__ == "__main__":
    main()
