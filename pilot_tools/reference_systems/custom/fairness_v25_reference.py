"""
v25 fairness-track pipeline: Retrieve → Collapse → Verify → Consensus → Answer
Cluster-aware Answer Consensus Gate.

Role:
- attack/defense fairness track for benchmark comparisons
- not the clean-only ceiling baseline
- Base/Attack share the same reader stack for fair comparison
"""
import os
import json
import csv
import argparse
import sys
from collections import Counter
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
    check_em,
    extract_first_answer,
    BASE_READER_PROMPT,
)
from benchmark_eval_utils import (
    eval_answer_from_raw,
    abstain_from_eval,
    conflict_flag_from_text,
    load_poison_maps_from_jsonl,
)
from weighted_mlm_defense import WeightedMLMDefense
from mlm_local_veto import mlm_local_veto, extract_evidence_window_per_doc
from anti_sybil_cluster import collapse_documents
from cross_encoder_reranker import CrossEncoderReranker
from answer_consensus_gate import run_consensus_gate, normalize_answer_for_consensus
from manifest_utils import load_manifest, apply_manifest, validate_selected_questions
from official_eval import print_evaluation_report
from run_metadata import build_run_metadata, write_run_metadata


def load_questions(hotpot_path: str, nq_path: str, max_questions: int = 0, shuffle: bool = False, seed: int = 42, wiki2_path: str = ""):
    questions = []
    if hotpot_path:
        with open(hotpot_path, "r") as f:
            for line in f:
                d = json.loads(line)
                questions.append({"id": str(d["id"]), "q": d["question"], "a": [d["answer"]], "ds": "hotpot"})
    if nq_path:
        with open(nq_path, "r") as f:
            for line in f:
                d = json.loads(line)
                ans = d.get("answer", [])
                if isinstance(ans, str):
                    ans = [ans]
                questions.append({"id": str(d["id"]), "q": d["question"], "a": ans, "ds": "nq"})
    if wiki2_path:
        with open(wiki2_path, "r") as f:
            for line in f:
                d = json.loads(line)
                ans = d.get("golden_answers", d.get("answer", []))
                if isinstance(ans, str):
                    ans = [ans]
                questions.append({"id": str(d["id"]), "q": d["question"], "a": ans, "ds": "wiki2"})
    if shuffle:
        import random
        random.seed(seed)
        random.shuffle(questions)
    if max_questions > 0:
        questions = questions[:max_questions]
    return questions


def retrieve_base(searcher, question: str, k: int) -> list:
    hits = searcher.search(question, k=k)
    results = []
    for h in hits:
        try:
            raw = json.loads(searcher.doc(h.docid).raw())
            results.append((raw.get("contents", ""), float(h.score)))
        except Exception:
            continue
    return results


def retrieve_faiss(faiss_retriever, question: str, k: int) -> list:
    return faiss_retriever.search(question, k=k)


def base_lightweight_consensus(
    question: str,
    windows: list,
    rerank_scores: list,
    llama_url: str,
    n_predict: int = 32,
    llm_backend: str = "llama_cpp_http",
    llm_model_id: str = "",
    llm_temperature: float = 0.0,
) -> tuple:
    """
    Base 전용: top evidence window별 짧은 답 추출 → 다수결 + rerank tie-break.
    Returns: (b_ans_raw, b_ans_eval)
    """
    if not windows:
        return "Unknown", "Unknown"
    answers = []
    scores = rerank_scores[: len(windows)] if rerank_scores else [0.0] * len(windows)
    for i, w in enumerate(windows):
        if not w or not w.strip():
            continue
        ans = get_llama_answer_from_passage(
            question,
            w,
            url=llama_url,
            n_predict=n_predict,
            prompt_template=BASE_READER_PROMPT,
            backend=llm_backend,
            model_id=llm_model_id,
            temperature=llm_temperature,
        )
        ans_eval = extract_first_answer(ans)
        norm_key = normalize_answer_for_consensus(ans_eval)
        answers.append((ans_eval, norm_key, ans, scores[i] if i < len(scores) else 0.0))
    # __unknown__ 제외 다수결
    counts = Counter()
    best_per_norm = {}  # norm_key -> (rerank_score, ans_eval, raw)
    for ans_eval, norm_key, raw, sc in answers:
        if norm_key == "__unknown__":
            continue
        counts[norm_key] += 1
        if norm_key not in best_per_norm or sc > best_per_norm[norm_key][0]:
            best_per_norm[norm_key] = (sc, ans_eval, raw)
    if not counts:
        return "Unknown", "Unknown"
    max_count = max(counts.values())
    winners = [k for k, c in counts.items() if c == max_count]
    best_k = max(winners, key=lambda k: best_per_norm[k][0])  # tie-break: rerank score
    _, ans_eval, raw = best_per_norm[best_k]
    return raw, ans_eval


def build_support_units(
    question: str,
    top_docs: list,
    rerank_scores: list,
    weighted_mlm: WeightedMLMDefense,
    use_doc_prefix: bool = False,
) -> tuple:
    """
    Reader support unit 구성:
    - 기본: evidence window 추출
    - 옵션: 문서 앞 1024자(doc[:1024]) 직접 사용
    Returns: (support_units, filtered_scores)
    """
    if use_doc_prefix:
        units = []
        scores = []
        for doc, score in zip(top_docs, rerank_scores):
            prefix = (doc or "")[:1024].strip()
            if prefix:
                units.append(prefix)
                scores.append(score)
        return units, scores

    keywords = weighted_mlm.extract_question_keywords(question)
    pairs = []
    for doc, score in zip(top_docs, rerank_scores):
        window = extract_evidence_window_per_doc(doc, question, keywords)
        if window and window.strip():
            pairs.append((window, score))
    units = [u for u, _ in pairs]
    scores = [s for _, s in pairs]
    return units, scores


def main():
    parser = argparse.ArgumentParser(description="v25 fairness-track: Retrieve → Collapse → Verify → Consensus → Answer")
    parser.add_argument("--base_index", required=True)
    parser.add_argument("--attack_index", required=True)
    parser.add_argument("--hotpot", default="")
    parser.add_argument("--nq", default="")
    parser.add_argument("--wiki2", default="", help="2WikiMultiHopQA JSONL (id, question, golden_answers[])")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--faiss_index_dir", type=str, default="")
    parser.add_argument("--poison_jsonl", type=str, default="poison_docs_llm.jsonl")
    parser.add_argument("--candidate_k", type=int, default=50)
    parser.add_argument("--final_k", type=int, default=10)
    parser.add_argument("--recall_k", type=int, default=40)
    parser.add_argument("--max_per_cluster", type=int, default=2)
    parser.add_argument("--mlm_veto_threshold", type=float, default=-3.0)
    parser.add_argument("--mlm_model", type=str, default="roberta-large")
    parser.add_argument("--cross_encoder_model", type=str, default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--closed_set_rerank", action="store_true")
    parser.add_argument("--poison_per_query", type=int, default=6)
    parser.add_argument("--trigger_tokens", type=str, default="zjgqv zxqvjk qqqzzz tknnoise tknnoise")
    parser.add_argument("--llama_url", type=str, default="http://127.0.0.1:8000/completion")
    parser.add_argument("--n_predict", type=int, default=64)
    parser.add_argument("--llm_backend", type=str, default=os.environ.get("LLM_BACKEND", "llama_cpp_http"))
    parser.add_argument("--llm_model_id", type=str, default=os.environ.get("LLM_MODEL_ID", ""))
    parser.add_argument("--llm_base_url", type=str, default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--llm_temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0.0")))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--shuffle_questions", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question_manifest", type=str, default="", help="Optional JSON manifest with ordered {ds,id} entries")
    parser.add_argument("--no_claim_bearing_spans", action="store_true")
    # v24 Consensus Gate
    parser.add_argument("--min_support_clusters", type=int, default=2, help="NQ: 채택 시 필요한 최소 독립 cluster 수")
    parser.add_argument("--min_support_pairs", type=int, default=2, help="Hotpot: 채택 시 필요한 최소 unique pair 수")
    parser.add_argument("--min_union_clusters", type=int, default=3, help="Hotpot: pair 합집합 cluster 최소 수")
    parser.add_argument("--min_support_margin", type=float, default=0.15, help="winner-runner margin 최소 (NQ)")
    parser.add_argument("--min_support_margin_hotpot", type=float, default=None, help="Hotpot margin (None=min_support_margin 사용)")
    parser.add_argument("--max_hotpot_units", type=int, default=20, help="Hotpot pair unit 상한 (계산량 제한)")
    # Base-v24+ 개선
    parser.add_argument("--base_final_k", type=int, default=6, help="Base path: rerank 후 사용할 상위 문서 수")
    parser.add_argument("--base_use_doc_prefix", action="store_true", help="Base/Attack reader support unit으로 evidence window 대신 doc[:1024] 사용")
    parser.add_argument("--base_use_2hop", action="store_true", help="Hotpot Base에 2-hop retrieval 사용 (faiss 필요)")
    parser.add_argument("--base_recall_k", type=int, default=0, help="Base retrieval 후보 수 (0이면 recall_k 사용)")
    parser.add_argument("--skip_llm_check", action="store_true", help="LLM 연결 검증 생략 (비권장)")
    parser.add_argument("--append_output", action="store_true", help="기존 CSV에 append (release 비권장)")
    parser.add_argument("--overwrite_output", action="store_true", help="기존 CSV 덮어쓰기 허용")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    if os.path.exists(args.output_csv) and not args.append_output and not args.overwrite_output:
        raise SystemExit(
            "[!] output_csv already exists. Use a new path, or pass --overwrite_output / --append_output explicitly."
        )

    base_recall_k = args.base_recall_k if args.base_recall_k > 0 else args.recall_k
    two_hop_search_fn = None
    if args.base_use_2hop and args.faiss_index_dir:
        try:
            import sys
            import importlib
            _dep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", "deprecated_modules")
            if _dep_path not in sys.path:
                sys.path.insert(0, _dep_path)
            two_hop_search_fn = importlib.import_module("two_hop_search").two_hop_search
            print("[*] Base 2-hop retrieval (Hotpot): enabled")
        except Exception as e:
            print("[!] Base 2-hop import 실패:", e)
            two_hop_search_fn = None

    print("[*] v25 Pipeline: Retrieve → Collapse → Verify → Consensus → Answer")
    print("[*] Track role: attack/defense fairness track (not clean-strong ceiling baseline)")
    print(f"[*] LLM backend: {args.llm_backend}, model={args.llm_model_id or 'default'}, temp={args.llm_temperature}")
    llm_url = args.llm_base_url or args.llama_url
    support_mode = "doc[:1024]" if args.base_use_doc_prefix else "evidence window"
    print("[*] Base/Attack reader: Rerank + %s + Lightweight Consensus (base_final_k=%d)" % (support_mode, args.base_final_k))
    print("[*] Base/Attack 공정 비교: 동일 reader stack")
    print("[*] Eval: extract_first_answer 공통 postprocess 적용 (raw/eval 분리)")
    margin_hotpot = args.min_support_margin_hotpot if args.min_support_margin_hotpot is not None else args.min_support_margin
    print(f"[*] Consensus: NQ min_clusters={args.min_support_clusters} margin={args.min_support_margin}, Hotpot min_pairs={args.min_support_pairs} margin={margin_hotpot}")

    base_searcher = LuceneSearcher(args.base_index)
    attack_searcher = LuceneSearcher(args.attack_index)

    faiss_retriever = None
    if args.faiss_index_dir and os.path.isdir(args.faiss_index_dir):
        from faiss_retriever import FAISSRetriever
        faiss_retriever = FAISSRetriever(
            faiss_index_path=os.path.join(args.faiss_index_dir, "index.faiss"),
            docid_map_path=os.path.join(args.faiss_index_dir, "docid_map.jsonl"),
            meta_path=os.path.join(args.faiss_index_dir, "meta.json"),
            lucene_index_path=args.base_index,
        )
        print("[*] FAISS E5 retrieval enabled")
    else:
        print("[*] Lucene BM25 retrieval")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weighted_mlm = WeightedMLMDefense(model_name=args.mlm_model, device=device)
    reranker = CrossEncoderReranker(model_name=args.cross_encoder_model, device=device)
    print("[*] Weighted-MLM + Cross-Encoder + Consensus Gate loaded")

    poison_map = {}
    poison_target_map = {}
    if args.closed_set_rerank and os.path.exists(args.poison_jsonl):
        poison_map, poison_target_map = load_poison_maps_from_jsonl(args.poison_jsonl)
        print(f"[*] Poison map: {len(poison_map)} keys, poison_target_map: {len(poison_target_map)} keys")

    questions = load_questions(
        args.hotpot, args.nq,
        max_questions=0 if args.question_manifest else args.max_questions,
        shuffle=False if args.question_manifest else args.shuffle_questions,
        seed=args.seed,
        wiki2_path=args.wiki2,
    )
    manifest = None
    if args.question_manifest:
        manifest = load_manifest(args.question_manifest)
        source_label = " + ".join(p for p in [args.hotpot, args.nq, args.wiki2] if p)
        questions = apply_manifest(
            questions,
            manifest,
            source_label=source_label,
        )
        validate_selected_questions(
            questions,
            manifest,
            source_label=source_label,
        )
        print(
            f"[*] Manifest applied: {manifest.get('name', os.path.basename(args.question_manifest))} "
            f"({len(questions)} questions, sha={str(manifest.get('selection_sha256', ''))[:12]})"
        )
    print(f"[*] Questions: {len(questions)}, start_idx={args.start_idx}")

    # LLM 연결/응답 검증 (실험 전 필수)
    if not args.skip_llm_check:
        print("[*] LLM 연결 검증 중...")
        try:
            ans = get_llama_answer_from_passage(
                "What is the capital of France?",
                "Paris is the capital of France. It is located in northern France.",
                url=llm_url,
                n_predict=16,
                prompt_template=BASE_READER_PROMPT,
                backend=args.llm_backend,
                model_id=args.llm_model_id,
                temperature=args.llm_temperature,
            )
            ans_norm = (ans or "").strip().lower()
            if not ans or ans_norm in ("unknown", "error", ""):
                print("[!] LLM 연결 실패: 샘플 호출이 Unknown/빈 응답을 반환했습니다.")
                print(f"    응답: {ans!r}")
                print(f"    URL: {args.llama_url}")
                print("    조치: LLaMA 서버가 실행 중인지, URL이 맞는지 확인하세요.")
                print("    Docker: docker exec 로 컨테이너 내부에서 실행해 보세요.")
                print("    생략: --skip_llm_check 로 건너뛸 수 있음 (비권장)")
                raise SystemExit(1)
            print(f"[*] LLM OK (샘플 응답: {ans[:60]}...)")
        except SystemExit:
            raise
        except Exception as e:
            print(f"[!] LLM 연결 실패: {e}")
            print(f"    URL: {args.llama_url}")
            print("    조치: LLaMA 서버가 실행 중인지 확인하세요.")
            raise SystemExit(1)

    run_meta = build_run_metadata(
        workspace_root=os.path.dirname(os.path.abspath(__file__)),
        manifest=manifest,
        base_index=args.base_index,
        attack_index=args.attack_index,
        faiss_index_dir=args.faiss_index_dir,
        poison_jsonl=args.poison_jsonl,
        prompt_text=BASE_READER_PROMPT,
        llm_backend=args.llm_backend,
        llm_model_id=args.llm_model_id,
        llm_base_url=llm_url,
        llm_temperature=args.llm_temperature,
        n_predict=args.n_predict,
        extra_fields={
            "candidate_k": args.candidate_k,
            "final_k": args.final_k,
            "recall_k": args.recall_k,
            "base_recall_k": base_recall_k,
            "base_final_k": args.base_final_k,
            "base_use_doc_prefix": args.base_use_doc_prefix,
            "base_use_2hop": args.base_use_2hop,
            "mlm_veto_threshold": args.mlm_veto_threshold,
            "mlm_model": args.mlm_model,
            "cross_encoder_model": args.cross_encoder_model,
            "min_support_clusters": args.min_support_clusters,
            "min_support_pairs": args.min_support_pairs,
            "min_union_clusters": args.min_union_clusters,
            "min_support_margin": args.min_support_margin,
            "min_support_margin_hotpot": args.min_support_margin_hotpot,
            "max_hotpot_units": args.max_hotpot_units,
            "max_per_cluster": args.max_per_cluster,
            "closed_set_rerank": args.closed_set_rerank,
            "poison_per_query": args.poison_per_query,
        },
    )

    fieldnames = [
        "ds", "id", "question", "gold_answer", "poison_target",
        "benchmark_track", "base_track_role", "attack_track_role", "defense_track_role",
        "manifest_name", "manifest_selection_sha256", "source_hotpot_sha256", "source_nq_sha256",
        "corpus_hash", "index_hash", "faiss_index_hash", "attack_index_hash", "poison_jsonl_sha256",
        "poison_generator_version", "poison_generator_prompt_hash", "git_commit", "prompt_hash",
        "generator_model_id", "generator_model_family", "generator_model_provider", "generator_model_date",
        "llm_backend", "llm_base_url", "llm_temperature", "run_n_predict",
        "config_candidate_k", "config_final_k", "config_recall_k", "config_base_recall_k", "config_base_final_k",
        "config_base_use_doc_prefix", "config_base_use_2hop", "config_mlm_veto_threshold",
        "config_min_support_clusters", "config_min_support_pairs", "config_min_union_clusters",
        "config_min_support_margin", "config_min_support_margin_hotpot", "config_max_hotpot_units",
        "base_reader_stack", "attack_reader_stack", "defense_stack",
        "base_answer_raw", "base_answer_eval", "base_answer_final", "base_best_raw", "base_em", "base_context_len", "base_abstain", "base_conflict_flag",
        "attack_answer_raw", "attack_answer_eval", "attack_answer_final", "attack_best_raw", "attack_em", "attack_poison_ratio", "attack_context_len", "attack_abstain", "attack_conflict_flag",
        "v25_answer_raw", "v25_answer_final", "v25_answer_eval", "v25_em", "v25_best_raw", "v25_context_len", "v25_abstain", "v25_conflict_flag",
        "base_asr", "attack_asr", "v25_asr",
    ]
    file_exists = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0
    write_mode = "a" if args.append_output else "w"

    with open(args.output_csv, write_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_mode == "w" or not file_exists:
            writer.writeheader()

        for i, item in enumerate(tqdm(questions[args.start_idx:], desc="v25 Eval")):
            idx = args.start_idx + i
            q = item["q"]
            golds = item["a"]
            gold_str = " | ".join(golds)
            qid = str(item["id"])
            pt = poison_target_map.get((item["ds"], qid))
            poison_target = pt if pt is not None else []

            poison_target_str = " | ".join(poison_target) if poison_target else ""

            # === Base-v24+ (clean): Rerank + Evidence Window + Lightweight Consensus ===
            try:
                # 1. Retrieval (Hotpot + 2-hop 옵션)
                if two_hop_search_fn and item["ds"] == "hotpot" and faiss_retriever:
                    b_results = two_hop_search_fn(
                        faiss_retriever, q, k_per_hop=min(30, base_recall_k // 2),
                        llama_url=llm_url,
                        llm_backend=args.llm_backend,
                        llm_model_id=args.llm_model_id,
                        llm_temperature=args.llm_temperature,
                    )
                elif faiss_retriever:
                    b_results = retrieve_faiss(faiss_retriever, q, base_recall_k)
                else:
                    b_results = retrieve_base(base_searcher, q, base_recall_k)
                b_docs = [t for t, _ in b_results[: base_recall_k]]
                if not b_docs:
                    b_ans_raw = "Unknown"
                    b_ans_eval = "Unknown"
                    base_em = 0
                    b_context = ""
                else:
                    # 2. Cross-encoder rerank
                    reranked_list = reranker.rerank(q, b_docs, top_k=args.base_final_k * 2, return_scores=True)
                    top_docs = [d for d, _ in reranked_list[: args.base_final_k]]
                    rerank_scores = [s for _, s in reranked_list[: args.base_final_k]]
                    # 3. Support unit 구성 (evidence window 또는 doc[:1024])
                    windows, rerank_scores_filtered = build_support_units(
                        q,
                        top_docs,
                        rerank_scores,
                        weighted_mlm,
                        use_doc_prefix=args.base_use_doc_prefix,
                    )
                    # 4. Lightweight consensus (다수결 + tie-break)
                    b_ans_raw, b_ans_eval = base_lightweight_consensus(
                        q,
                        windows,
                        rerank_scores_filtered,
                        llm_url,
                        n_predict=32,
                        llm_backend=args.llm_backend,
                        llm_model_id=args.llm_model_id,
                        llm_temperature=args.llm_temperature,
                    )
                    base_em = 1 if check_em(b_ans_eval, golds) else 0
                    b_context = "\n\n".join(windows)
            except Exception as e:
                b_ans_raw = f"[ERROR: {str(e)}]"
                b_ans_eval = "Unknown"
                base_em = 0
                b_context = ""

            # === Attack (undefended): Base와 동일 reader stack (공정 비교) ===
            attack_context = ""
            attack_em = 0
            poison_ratio = 0.0
            a_docs = []
            a_scores_raw = []
            try:
                # poisoned candidate docs 구성 (Base와 동일 retrieval, poison 주입만 추가)
                if args.closed_set_rerank:
                    retr_k = max(args.candidate_k, args.recall_k)
                    if faiss_retriever:
                        a_results = retrieve_faiss(faiss_retriever, q, retr_k)
                    else:
                        a_results = retrieve_base(base_searcher, q, retr_k)
                    a_docs = [t for t, _ in a_results[: args.candidate_k]]
                    a_scores_raw = [s for _, s in a_results[: args.candidate_k]]
                    poisons = poison_map.get((item["ds"], qid), [])[: args.poison_per_query]
                    if poisons:
                        injected = [p.strip() + " " + args.trigger_tokens for p in poisons if p.strip()]
                        min_sc = min(a_scores_raw) if a_scores_raw else 0.0
                        a_docs = injected + a_docs
                        a_scores_raw = [min_sc] * len(injected) + a_scores_raw
                        a_docs = a_docs[: args.candidate_k]
                        a_scores_raw = a_scores_raw[: args.candidate_k]
                    poison_ratio = min(1.0, len(poisons) / len(a_docs)) if a_docs else 0.0
                else:
                    a_hits = attack_searcher.search(q, k=args.recall_k)
                    for h in a_hits:
                        try:
                            raw = json.loads(attack_searcher.doc(h.docid).raw())
                            a_docs.append(raw.get("contents", ""))
                            a_scores_raw.append(float(h.score))
                        except Exception:
                            continue

                if not a_docs:
                    attack_ans_raw = "Unknown"
                    attack_ans_eval = "Unknown"
                else:
                    # Base와 동일 reader: rerank → evidence window → lightweight consensus
                    reranked_list = reranker.rerank(q, a_docs, top_k=args.base_final_k * 2, return_scores=True)
                    top_docs = [d for d, _ in reranked_list[: args.base_final_k]]
                    rerank_scores = [s for _, s in reranked_list[: args.base_final_k]]
                    windows, rerank_scores_filtered = build_support_units(
                        q,
                        top_docs,
                        rerank_scores,
                        weighted_mlm,
                        use_doc_prefix=args.base_use_doc_prefix,
                    )
                    attack_ans_raw, attack_ans_eval = base_lightweight_consensus(
                        q,
                        windows,
                        rerank_scores_filtered,
                        llm_url,
                        n_predict=32,
                        llm_backend=args.llm_backend,
                        llm_model_id=args.llm_model_id,
                        llm_temperature=args.llm_temperature,
                    )
                    attack_context = "\n\n".join(windows)
                attack_em = 1 if check_em(attack_ans_eval, golds) else 0
            except Exception as e:
                attack_ans_raw = f"[ERROR: {str(e)}]"
                attack_ans_eval = "Unknown"
                attack_em = 0
                attack_context = ""

            # === v24: Collapse → Rerank → Veto → Consensus → Answer ===
            v24_ans_raw = "Unknown"
            v24_ans_eval = "Unknown"
            v24_best_raw = ""
            v24_context = ""
            v24_em = 0
            v24_abstain = None
            try:
                if not a_docs:
                    v24_abstain = True
                else:
                    collapsed_docs, collapsed_scores, _, collapsed_cluster_ids = collapse_documents(
                        a_docs, a_scores_raw,
                        embeddings=None,
                        max_per_cluster=args.max_per_cluster,
                        use_simhash=True,
                    )
                    if not collapsed_docs:
                        collapsed_docs, collapsed_scores = a_docs, a_scores_raw
                        collapsed_cluster_ids = list(range(len(a_docs)))

                    # index 유지 rerank (문서 중복 시에도 cluster_id 정확 매핑)
                    reranked_docs, reranked_scores, reranked_cluster_ids = reranker.rerank_with_metadata(
                        q,
                        collapsed_docs,
                        collapsed_cluster_ids,
                        top_k=args.final_k * 2,
                    )

                    surviving, surviving_mlm_scores, surviving_indices, surviving_windows = mlm_local_veto(
                        weighted_mlm,
                        q,
                        reranked_docs,
                        veto_threshold=args.mlm_veto_threshold,
                        use_selective_masking=True,
                        use_claim_bearing_spans=not args.no_claim_bearing_spans,
                    )

                    surviving_cluster_ids = [reranked_cluster_ids[i] for i in surviving_indices if i < len(reranked_cluster_ids)]
                    surviving_rerank_scores = [reranked_scores[i] for i in surviving_indices if i < len(reranked_scores)]

                    v24_answer_norm, v24_gate_abstain, v24_best_raw = run_consensus_gate(
                        question=q,
                        surviving_docs=surviving,
                        surviving_windows=surviving_windows,
                        surviving_cluster_ids=surviving_cluster_ids,
                        surviving_rerank_scores=surviving_rerank_scores,
                        surviving_mlm_scores=surviving_mlm_scores,
                        ds=item["ds"],
                        llama_url=llm_url,
                        n_predict=32,
                        llm_backend=args.llm_backend,
                        llm_model_id=args.llm_model_id,
                        llm_temperature=args.llm_temperature,
                        veto_threshold=args.mlm_veto_threshold,
                        min_support_clusters=args.min_support_clusters,
                        min_support_pairs=args.min_support_pairs,
                        min_union_clusters=args.min_union_clusters,
                        min_support_margin=args.min_support_margin,
                        min_support_margin_hotpot=args.min_support_margin_hotpot,
                        max_hotpot_units=args.max_hotpot_units,
                    )

                    v24_abstain = bool(v24_gate_abstain)
                    v24_ans_raw = v24_answer_norm if not v24_abstain else "Unknown"
                    v24_ans_eval = extract_first_answer(v24_ans_raw)
                    v24_context = "\n\n".join(surviving[: args.final_k]) if surviving else ""
                v24_em = 1 if check_em(v24_ans_eval, golds) else 0
            except Exception as e:
                v24_ans_raw = f"[ERROR: {str(e)}]"
                v24_ans_eval = "Unknown"
                v24_abstain = True
                v24_context = ""
                v24_em = 0

            base_answer_eval = eval_answer_from_raw(b_ans_raw)
            attack_answer_eval = eval_answer_from_raw(attack_ans_raw)
            v24_answer_raw = v24_best_raw or v24_ans_raw
            v25_answer_eval = eval_answer_from_raw(v24_answer_raw)

            base_em = 1 if check_em(base_answer_eval, golds) else 0
            attack_em = 1 if check_em(attack_answer_eval, golds) else 0
            v24_em = 1 if check_em(v25_answer_eval, golds) else 0

            base_asr = 0 if base_em else (1 if poison_target and check_em(base_answer_eval, poison_target) else 0)
            attack_asr = 0 if attack_em else (1 if poison_target and check_em(attack_answer_eval, poison_target) else 0)
            v24_asr = 0 if v24_em else (1 if poison_target and check_em(v25_answer_eval, poison_target) else 0)
            base_abstain = abstain_from_eval(base_answer_eval)
            attack_abstain = abstain_from_eval(attack_answer_eval)
            explicit_v24_abstain = bool(v24_abstain) if v24_abstain is not None else False
            v24_abstain = abstain_from_eval(v25_answer_eval, explicit_flag=explicit_v24_abstain)
            base_conflict_flag = conflict_flag_from_text(b_ans_raw, golds, poison_target)
            attack_conflict_flag = conflict_flag_from_text(attack_ans_raw, golds, poison_target)
            v25_conflict_flag = conflict_flag_from_text(v24_answer_raw, golds, poison_target)

            row = {
                "ds": item["ds"],
                "id": item["id"],
                "question": q,
                "gold_answer": gold_str,
                "poison_target": poison_target_str,
                "benchmark_track": "attack_defense_fairness",
                "base_track_role": "fairness_reference_baseline",
                "attack_track_role": "undefended_attacked_baseline",
                "defense_track_role": "purification_reference_baseline",
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
                "config_candidate_k": run_meta["candidate_k"],
                "config_final_k": run_meta["final_k"],
                "config_recall_k": run_meta["recall_k"],
                "config_base_recall_k": run_meta["base_recall_k"],
                "config_base_final_k": run_meta["base_final_k"],
                "config_base_use_doc_prefix": run_meta["base_use_doc_prefix"],
                "config_base_use_2hop": run_meta["base_use_2hop"],
                "config_mlm_veto_threshold": run_meta["mlm_veto_threshold"],
                "config_min_support_clusters": run_meta["min_support_clusters"],
                "config_min_support_pairs": run_meta["min_support_pairs"],
                "config_min_union_clusters": run_meta["min_union_clusters"],
                "config_min_support_margin": run_meta["min_support_margin"],
                "config_min_support_margin_hotpot": run_meta["min_support_margin_hotpot"],
                "config_max_hotpot_units": run_meta["max_hotpot_units"],
                "base_reader_stack": f"rerank+{support_mode}+lightweight_consensus",
                "attack_reader_stack": f"rerank+{support_mode}+lightweight_consensus",
                "defense_stack": "collapse+rerank+mlm_local_veto+consensus_gate",
                "base_answer_raw": b_ans_raw,
                "base_answer_eval": base_answer_eval,
                "base_answer_final": base_answer_eval,
                "base_best_raw": b_ans_raw,
                "base_em": base_em,
                "base_context_len": len(b_context),
                "base_abstain": base_abstain,
                "base_conflict_flag": int(base_conflict_flag),
                "attack_answer_raw": attack_ans_raw,
                "attack_answer_eval": attack_answer_eval,
                "attack_answer_final": attack_answer_eval,
                "attack_best_raw": attack_ans_raw,
                "attack_em": attack_em,
                "attack_poison_ratio": poison_ratio if args.closed_set_rerank else "",
                "attack_context_len": len(attack_context),
                "attack_abstain": attack_abstain,
                "attack_conflict_flag": int(attack_conflict_flag),
                "v25_answer_raw": v24_answer_raw,
                "v25_answer_final": v25_answer_eval,
                "v25_answer_eval": v25_answer_eval,
                "v25_em": v24_em,
                "v25_best_raw": (v24_best_raw or "") if not v24_abstain else "",
                "v25_context_len": len(v24_context),
                "v25_abstain": v24_abstain,
                "v25_conflict_flag": int(v25_conflict_flag),
                "base_asr": base_asr,
                "attack_asr": attack_asr,
                "v25_asr": v24_asr,
            }
            writer.writerow(row)
            csvfile.flush()

    print("[*] Done.")
    write_run_metadata(args.output_csv, run_meta)

    print_evaluation_report(
        args.output_csv,
        system_prefixes=["base", "attack", "v25"],
        strict_schema=True,
    )


if __name__ == "__main__":
    main()
