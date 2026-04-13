"""
Base-only clean evaluation pipeline.

Purpose:
- Evaluate the clean-only benchmark track.
- Estimate the clean QA ceiling without purification trade-offs.
- No defense modules: no collapse, no MLM veto, no consensus gate.
- Dataset-aware reader path:
  - NQ: support-unit answers + lightweight voting
  - Hotpot: multi-doc concatenated context + single reader call
"""
import os
import json
import csv
import argparse
import sys
from collections import Counter
from typing import List

from tqdm import tqdm

# Path setup: file lives in reference_systems/custom/; modules are in pilot_tools/ root
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in [_ROOT, os.path.join(_ROOT, "benchmark_core")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pyserini.search.lucene import LuceneSearcher

from llm_answering import (
    get_llama_answer_common,
    get_llama_answer_from_passage,
    extract_first_answer,
    check_em,
    BASE_READER_PROMPT,
)
from benchmark_eval_utils import (
    eval_answer_from_raw,
    abstain_from_eval,
    conflict_flag_from_text,
    load_poison_maps_from_jsonl,
)
from cross_encoder_reranker import CrossEncoderReranker
from hotpot_pair_reader import build_pair_units, pair_vote_answers
from manifest_utils import load_manifest, apply_manifest, validate_selected_questions
from official_eval import print_evaluation_report
from run_metadata import build_run_metadata, write_run_metadata


def load_questions(hotpot_path: str, nq_path: str, max_questions: int = 0, shuffle: bool = False, seed: int = 42):
    questions = []
    with open(hotpot_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            questions.append({"id": str(d["id"]), "q": d["question"], "a": [d["answer"]], "ds": "hotpot"})
    with open(nq_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ans = d.get("answer", [])
            if isinstance(ans, str):
                ans = [ans]
            questions.append({"id": str(d["id"]), "q": d["question"], "a": ans, "ds": "nq"})
    if shuffle:
        import random
        random.seed(seed)
        random.shuffle(questions)
    if max_questions > 0:
        questions = questions[:max_questions]
    return questions


def load_poison_target_map(poison_jsonl: str) -> dict:
    _, targets = load_poison_maps_from_jsonl(poison_jsonl)
    return targets


def retrieve_lucene(searcher: LuceneSearcher, question: str, k: int):
    hits = searcher.search(question, k=k)
    out = []
    for h in hits:
        try:
            raw = json.loads(searcher.doc(h.docid).raw())
            text = raw.get("contents", "")
            if text:
                out.append((text, float(h.score)))
        except Exception:
            continue
    return out


def build_doc_prefix_units(top_docs: list, rerank_scores: list, max_chars: int = 1024):
    units = []
    scores = []
    for doc, score in zip(top_docs, rerank_scores):
        prefix = (doc or "")[:max_chars].strip()
        if prefix:
            units.append(prefix)
            scores.append(score)
    return units, scores


def resolve_dataset_value(ds: str, hotpot_value: int, nq_value: int, default_value: int) -> int:
    if ds == "hotpot" and hotpot_value > 0:
        return hotpot_value
    if ds == "nq" and nq_value > 0:
        return nq_value
    return default_value


def lightweight_vote_answers(
    question: str,
    units: list,
    scores: list,
    llama_url: str,
    n_predict: int,
    llm_backend: str,
    llm_model_id: str,
    llm_temperature: float,
):
    if not units:
        return "Unknown", "Unknown"
    counts = Counter()
    best_per_norm = {}
    for i, unit in enumerate(units):
        ans_raw = get_llama_answer_from_passage(
            question=question,
            passage=unit,
            url=llama_url,
            n_predict=n_predict,
            prompt_template=BASE_READER_PROMPT,
            backend=llm_backend,
            model_id=llm_model_id,
            temperature=llm_temperature,
        )
        ans_eval = extract_first_answer(ans_raw)
        norm = ans_eval.lower().strip()
        if not norm or norm == "unknown":
            continue
        counts[norm] += 1
        sc = scores[i] if i < len(scores) else 0.0
        if norm not in best_per_norm or sc > best_per_norm[norm][0]:
            best_per_norm[norm] = (sc, ans_raw, ans_eval)

    if not counts:
        return "Unknown", "Unknown"
    max_count = max(counts.values())
    winners = [k for k, c in counts.items() if c == max_count]
    best_norm = max(winners, key=lambda k: best_per_norm[k][0])
    _, ans_raw, ans_eval = best_per_norm[best_norm]
    return ans_raw, ans_eval


def main():
    parser = argparse.ArgumentParser(description="Base-only clean evaluation pipeline")
    parser.add_argument("--base_index", required=True)
    parser.add_argument("--hotpot", required=True)
    parser.add_argument("--nq", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--faiss_index_dir", type=str, default="")
    parser.add_argument("--poison_jsonl", type=str, default="poison_docs_llm.jsonl")
    parser.add_argument("--llama_url", type=str, default="http://127.0.0.1:8000/completion")
    parser.add_argument("--n_predict", type=int, default=64)
    parser.add_argument("--llm_backend", type=str, default=os.environ.get("LLM_BACKEND", "llama_cpp_http"))
    parser.add_argument("--llm_model_id", type=str, default=os.environ.get("LLM_MODEL_ID", ""))
    parser.add_argument("--llm_base_url", type=str, default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--llm_temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0.0")))
    parser.add_argument("--recall_k", type=int, default=40)
    parser.add_argument("--base_final_k", type=int, default=8)
    parser.add_argument("--max_questions", type=int, default=500)
    parser.add_argument("--shuffle_questions", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "hotpot", "nq"])
    parser.add_argument("--question_manifest", type=str, default="", help="Optional JSON manifest with ordered {ds,id} entries")
    parser.add_argument("--cross_encoder_model", type=str, default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--clean_profile", type=str, default="benchmark_clean_strong", choices=["benchmark_clean_strong", "custom"])
    parser.add_argument("--support_prefix_chars", type=int, default=1024)
    parser.add_argument("--hotpot_support_prefix_chars", type=int, default=0, help="Hotpot clean path override (0이면 support_prefix_chars 사용)")
    parser.add_argument("--nq_support_prefix_chars", type=int, default=0, help="NQ clean path override (0이면 support_prefix_chars 사용)")
    parser.add_argument("--hotpot_use_2hop", action="store_true")
    parser.add_argument("--faiss_k_per_hop", type=int, default=30)
    parser.add_argument("--hotpot_final_k", type=int, default=0, help="Hotpot clean path rerank top-k override (0이면 base_final_k 사용)")
    parser.add_argument("--nq_final_k", type=int, default=0, help="NQ clean path rerank top-k override (0이면 base_final_k 사용)")
    parser.add_argument("--hotpot_reader_mode", type=str, default="multi_doc_single", choices=["multi_doc_single", "vote", "pair_vote"])
    parser.add_argument("--hotpot_pair_max_pairs", type=int, default=12, help="Hotpot pair_vote mode: max pair units")
    parser.add_argument("--hotpot_pair_sentences_per_doc", type=int, default=2, help="Hotpot pair_vote mode: snippets per doc")
    parser.add_argument("--hotpot_pair_min_votes", type=int, default=1, help="Hotpot pair_vote mode: minimum vote count for a final answer")
    parser.add_argument("--nq_reader_mode", type=str, default="vote", choices=["vote", "multi_doc_single"])
    parser.add_argument("--skip_llm_check", action="store_true", help="LLM 연결 검증 생략 (비권장)")
    parser.add_argument("--append_output", action="store_true", help="기존 CSV에 append (release 비권장)")
    parser.add_argument("--overwrite_output", action="store_true", help="기존 CSV 덮어쓰기 허용")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    if os.path.exists(args.output_csv) and not args.append_output and not args.overwrite_output:
        raise SystemExit(
            "[!] output_csv already exists. Use a new path, or pass --overwrite_output / --append_output explicitly."
        )

    print("[*] Base-only clean pipeline (no defense modules)")
    print(f"[*] Track role: clean-strong benchmark baseline (profile={args.clean_profile})")
    print(f"[*] LLM backend: {args.llm_backend}, model={args.llm_model_id or 'default'}, temp={args.llm_temperature}")
    print(f"[*] Reader mode: NQ={args.nq_reader_mode}, Hotpot={args.hotpot_reader_mode}")
    print(f"[*] Retrieval recall_k={args.recall_k}, final_k={args.base_final_k}")
    print(
        "[*] Dataset overrides: "
        f"Hotpot final_k={args.hotpot_final_k or args.base_final_k}, "
        f"NQ final_k={args.nq_final_k or args.base_final_k}, "
        f"Hotpot prefix={args.hotpot_support_prefix_chars or args.support_prefix_chars}, "
        f"NQ prefix={args.nq_support_prefix_chars or args.support_prefix_chars}"
    )
    llm_url = args.llm_base_url or args.llama_url

    base_searcher = LuceneSearcher(args.base_index)
    faiss_retriever = None
    if args.faiss_index_dir and os.path.isdir(args.faiss_index_dir):
        from faiss_retriever import FAISSRetriever
        faiss_retriever = FAISSRetriever(
            faiss_index_path=os.path.join(args.faiss_index_dir, "index.faiss"),
            docid_map_path=os.path.join(args.faiss_index_dir, "docid_map.jsonl"),
            meta_path=os.path.join(args.faiss_index_dir, "meta.json"),
            lucene_index_path=args.base_index,
        )
        print("[*] Retrieval backend: FAISS")
    else:
        print("[*] Retrieval backend: Lucene")

    two_hop_search_fn = None
    if args.hotpot_use_2hop:
        try:
            import importlib
            two_hop_search_fn = importlib.import_module("two_hop_search").two_hop_search
            if not faiss_retriever:
                print("[!] hotpot_use_2hop requested but FAISS is not enabled; fallback to single-hop")
                two_hop_search_fn = None
            else:
                print("[*] Hotpot 2-hop retrieval: enabled")
        except Exception:
            try:
                import sys
                dep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", "deprecated_modules")
                if dep_path not in sys.path:
                    sys.path.insert(0, dep_path)
                import importlib
                two_hop_search_fn = importlib.import_module("two_hop_search").two_hop_search
                if not faiss_retriever:
                    print("[!] hotpot_use_2hop requested but FAISS is not enabled; fallback to single-hop")
                    two_hop_search_fn = None
                else:
                    print("[*] Hotpot 2-hop retrieval: enabled (deprecated_modules path)")
            except Exception as e:
                print(f"[!] Failed to load two_hop_search: {e}")
                two_hop_search_fn = None

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoderReranker(model_name=args.cross_encoder_model, device=device)

    questions = load_questions(
        hotpot_path=args.hotpot,
        nq_path=args.nq,
        max_questions=0 if args.question_manifest else args.max_questions,
        shuffle=False if args.question_manifest else args.shuffle_questions,
        seed=args.seed,
    )
    manifest = None
    if args.question_manifest:
        manifest = load_manifest(args.question_manifest)
        questions = apply_manifest(
            questions,
            manifest,
            source_label=f"{args.hotpot} + {args.nq}",
        )
        validate_selected_questions(
            questions,
            manifest,
            source_label=f"{args.hotpot} + {args.nq}",
        )
        print(
            f"[*] Manifest applied: {manifest.get('name', os.path.basename(args.question_manifest))} "
            f"({len(questions)} questions, sha={str(manifest.get('selection_sha256', ''))[:12]})"
        )
    if args.dataset != "all":
        questions = [x for x in questions if x["ds"] == args.dataset]
    print(f"[*] Loaded {len(questions)} questions")
    poison_target_map = load_poison_target_map(args.poison_jsonl)

    if not args.skip_llm_check:
        print("[*] LLM 연결 검증 중...")
        ans = get_llama_answer_from_passage(
            question="What is the capital of France?",
            passage="Paris is the capital of France. It is located in northern France.",
            url=llm_url,
            n_predict=16,
            prompt_template=BASE_READER_PROMPT,
            backend=args.llm_backend,
            model_id=args.llm_model_id,
            temperature=args.llm_temperature,
        )
        if not ans or ans.strip().lower() in {"", "unknown", "error"}:
            raise SystemExit("[!] LLM 연결 실패: base-only clean track를 시작하기 전에 LLM 서버를 확인하세요.")
        print(f"[*] LLM OK (샘플 응답: {ans[:60]}...)")

    run_meta = build_run_metadata(
        workspace_root=os.path.dirname(os.path.abspath(__file__)),
        manifest=manifest,
        base_index=args.base_index,
        faiss_index_dir=args.faiss_index_dir,
        poison_jsonl=args.poison_jsonl,
        prompt_text=BASE_READER_PROMPT,
        llm_backend=args.llm_backend,
        llm_model_id=args.llm_model_id,
        llm_base_url=llm_url,
        llm_temperature=args.llm_temperature,
        n_predict=args.n_predict,
        extra_fields={
            "recall_k": args.recall_k,
            "base_final_k": args.base_final_k,
            "support_prefix_chars": args.support_prefix_chars,
            "hotpot_support_prefix_chars": args.hotpot_support_prefix_chars,
            "nq_support_prefix_chars": args.nq_support_prefix_chars,
            "hotpot_use_2hop": args.hotpot_use_2hop,
            "hotpot_reader_mode": args.hotpot_reader_mode,
            "hotpot_pair_max_pairs": args.hotpot_pair_max_pairs,
            "hotpot_pair_sentences_per_doc": args.hotpot_pair_sentences_per_doc,
            "hotpot_pair_min_votes": args.hotpot_pair_min_votes,
            "nq_reader_mode": args.nq_reader_mode,
            "cross_encoder_model": args.cross_encoder_model,
            "clean_profile": args.clean_profile,
        },
    )

    fieldnames = [
        "ds",
        "id",
        "question",
        "gold_answer",
        "poison_target",
        "benchmark_track",
        "base_track_role",
        "clean_profile",
        "manifest_name",
        "manifest_selection_sha256",
        "source_hotpot_sha256",
        "source_nq_sha256",
        "corpus_hash",
        "index_hash",
        "faiss_index_hash",
        "attack_index_hash",
        "poison_jsonl_sha256",
        "poison_generator_version",
        "poison_generator_prompt_hash",
        "git_commit",
        "prompt_hash",
        "generator_model_id",
        "generator_model_family",
        "generator_model_provider",
        "generator_model_date",
        "llm_backend",
        "llm_base_url",
        "llm_temperature",
        "run_n_predict",
        "config_recall_k",
        "config_base_final_k",
        "config_hotpot_use_2hop",
        "config_cross_encoder_model",
        "base_answer_raw",
        "base_answer_eval",
        "base_answer_final",
        "base_best_raw",
        "base_em",
        "base_context_len",
        "base_abstain",
        "base_conflict_flag",
        "retrieval_mode",
        "reader_mode",
        "dataset_final_k",
        "support_prefix_chars_used",
    ]
    file_exists = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0
    write_mode = "a" if args.append_output else "w"

    with open(args.output_csv, write_mode, newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        if write_mode == "w" or not file_exists:
            writer.writeheader()

        for i, item in enumerate(tqdm(questions[args.start_idx:], desc="Evaluating(BaseOnly)")):
            q = item["q"]
            golds = item["a"]
            ds = item["ds"]
            poison_target = poison_target_map.get((ds, str(item["id"])), [])

            retrieval_mode = "lucene"
            retrieved = []
            if ds == "hotpot" and args.hotpot_use_2hop and two_hop_search_fn and faiss_retriever:
                retrieved = two_hop_search_fn(
                    retriever=faiss_retriever,
                    question=q,
                    k_per_hop=args.faiss_k_per_hop,
                    llama_url=llm_url,
                    llm_backend=args.llm_backend,
                    llm_model_id=args.llm_model_id,
                    llm_temperature=args.llm_temperature,
                )
                retrieval_mode = "faiss_2hop"
            elif faiss_retriever is not None:
                retrieved = faiss_retriever.search(q, k=args.recall_k)
                retrieval_mode = "faiss"
            else:
                retrieved = retrieve_lucene(base_searcher, q, k=args.recall_k)

            docs = [d for d, _ in retrieved if d]
            dataset_final_k = resolve_dataset_value(ds, args.hotpot_final_k, args.nq_final_k, args.base_final_k)
            support_prefix_chars_used = resolve_dataset_value(
                ds,
                args.hotpot_support_prefix_chars,
                args.nq_support_prefix_chars,
                args.support_prefix_chars,
            )
            reranked = reranker.rerank(query=q, documents=docs, top_k=dataset_final_k, return_scores=True) if docs else []
            top_docs = [d for d, _ in reranked]
            top_scores = [s for _, s in reranked]

            if ds == "hotpot":
                reader_mode = args.hotpot_reader_mode
            else:
                reader_mode = args.nq_reader_mode

            if reader_mode == "multi_doc_single":
                parts = [(d or "")[:support_prefix_chars_used].strip() for d in top_docs]
                parts = [p for p in parts if p]
                context = "\n\n".join(parts)
                if context:
                    ans_raw = get_llama_answer_common(
                        context,
                        q,
                        llm_url,
                        args.n_predict,
                        backend=args.llm_backend,
                        model_id=args.llm_model_id,
                        temperature=args.llm_temperature,
                    )
                else:
                    ans_raw = "Unknown"
                ans_eval = extract_first_answer(ans_raw)
            elif reader_mode == "pair_vote":
                pair_units = build_pair_units(
                    top_docs=top_docs,
                    top_scores=top_scores,
                    max_chars=max(256, support_prefix_chars_used // 2),
                    max_pairs=max(4, min(args.hotpot_pair_max_pairs, dataset_final_k * 3)),
                    sentences_per_doc=args.hotpot_pair_sentences_per_doc,
                )
                ans_raw, ans_eval = pair_vote_answers(
                    question=q,
                    pair_units=pair_units,
                    llama_url=llm_url,
                    n_predict=args.n_predict,
                    llm_backend=args.llm_backend,
                    llm_model_id=args.llm_model_id,
                    llm_temperature=args.llm_temperature,
                    min_votes=args.hotpot_pair_min_votes,
                )
                context = "\n\n".join(p for p, _ in pair_units)
            else:
                units, unit_scores = build_doc_prefix_units(
                    top_docs=top_docs,
                    rerank_scores=top_scores,
                    max_chars=support_prefix_chars_used,
                )
                ans_raw, ans_eval = lightweight_vote_answers(
                    question=q,
                    units=units,
                    scores=unit_scores,
                    llama_url=llm_url,
                    n_predict=args.n_predict,
                    llm_backend=args.llm_backend,
                    llm_model_id=args.llm_model_id,
                    llm_temperature=args.llm_temperature,
                )
                context = "\n\n".join(units)

            base_answer_eval = eval_answer_from_raw(ans_raw)
            em = 1 if check_em(base_answer_eval, golds) else 0
            abstain = abstain_from_eval(base_answer_eval)
            conflict_flag = conflict_flag_from_text(ans_raw, golds, poison_target)
            writer.writerow(
                {
                    "ds": ds,
                    "id": item["id"],
                    "question": q,
                    "gold_answer": " | ".join(golds),
                    "poison_target": " | ".join(poison_target) if poison_target else "",
                    "benchmark_track": "clean_ceiling_reference",
                    "base_track_role": "clean_ceiling_reference",
                    "clean_profile": args.clean_profile,
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
                    "config_recall_k": run_meta["recall_k"],
                    "config_base_final_k": run_meta["base_final_k"],
                    "config_hotpot_use_2hop": run_meta["hotpot_use_2hop"],
                    "config_cross_encoder_model": run_meta["cross_encoder_model"],
                    "base_answer_raw": ans_raw,
                    "base_answer_eval": base_answer_eval,
                    "base_answer_final": base_answer_eval,
                    "base_best_raw": ans_raw,
                    "base_em": em,
                    "base_context_len": len(context),
                    "base_abstain": abstain,
                    "base_conflict_flag": int(conflict_flag),
                    "retrieval_mode": retrieval_mode,
                    "reader_mode": reader_mode,
                    "dataset_final_k": dataset_final_k,
                    "support_prefix_chars_used": support_prefix_chars_used,
                }
            )

            if (i + 1) % 50 == 0:
                cf.flush()
                print(f"[*] Progress: {i + 1}/{len(questions) - args.start_idx}", flush=True)

    write_run_metadata(args.output_csv, run_meta)
    print_evaluation_report(
        args.output_csv,
        system_prefixes=["base"],
        strict_schema=True,
    )


if __name__ == "__main__":
    main()

