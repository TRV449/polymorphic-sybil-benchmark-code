from __future__ import annotations

import argparse
import json
import os
import random
import re
import zlib
import requests
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

from pyserini.search.lucene import LuceneSearcher

# reuse existing answer mutation helpers
from retarget_poison import (
    _looks_numeric_answer,  # type: ignore
    _perturb_number_str,  # type: ignore
    _pick_plausible_text_replacement,  # type: ignore
)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _stable_seed(s: str, base_seed: int) -> int:
    return (zlib.crc32(s.encode("utf-8")) ^ (base_seed & 0xFFFFFFFF)) & 0xFFFFFFFF


def normalize_answer(s: str) -> str:
    s = str(s)
    s = re.sub(r"\b(a|an|the)\b", " ", s, flags=re.I)
    # remove punctuation (simple, stable)
    s = re.sub(r"[^A-Za-z0-9\s]", "", s)
    s = " ".join(s.lower().split())
    return s


def question_to_statement(question: str, wrong_answer: str) -> str:
    q = question.strip()
    if q.endswith("?"):
        q = q[:-1].strip()

    m = re.match(r"^In what year was\s+(.+)$", q, flags=re.I)
    if m:
        subj = re.sub(r"\breleased\b$", "", m.group(1).strip(), flags=re.I).strip()
        return f"{subj} was released in {wrong_answer}."

    m = re.match(r"^In what year were\s+(.+)$", q, flags=re.I)
    if m:
        subj = re.sub(r"\breleased\b$", "", m.group(1).strip(), flags=re.I).strip()
        return f"{subj} were released in {wrong_answer}."

    m = re.match(r"^When was\s+(.+)$", q, flags=re.I)
    if m:
        return f"{m.group(1).strip()} was in {wrong_answer}."

    m = re.match(r"^When were\s+(.+)$", q, flags=re.I)
    if m:
        return f"{m.group(1).strip()} were in {wrong_answer}."

    m = re.match(r"^What is\s+(.+)$", q, flags=re.I)
    if m:
        return f"The {m.group(1).strip()} is {wrong_answer}."

    m = re.match(r"^What was\s+(.+)$", q, flags=re.I)
    if m:
        return f"The {m.group(1).strip()} was {wrong_answer}."

    if re.match(r"^(Is|Are|Was|Were)\b", q, flags=re.I):
        return f"The answer is {wrong_answer}."

    # fallback
    return f"{q} is {wrong_answer}."


def choose_wrong_answer(golds: List[str], answer_pool: List[str], *, rng: random.Random) -> str:
    gold0 = str(golds[0]).strip()
    if gold0.lower() in {"yes", "no"}:
        return "no" if gold0.lower() == "yes" else "yes"

    if _looks_numeric_answer(gold0):
        m = re.search(r"\d[\d,]*\d|\d+\.\d+", gold0)
        if m:
            chunk = m.group(0)
            new_chunk = _perturb_number_str(chunk, rng=rng)
            out = gold0.replace(chunk, new_chunk, 1)
            if out != gold0:
                return out
        nums = [a for a in answer_pool if _looks_numeric_answer(a) and a not in golds]
        return rng.choice(nums) if nums else (gold0 + " 1")

    wrong = _pick_plausible_text_replacement(gold0, answer_pool, rng=rng)
    if wrong and wrong not in golds:
        return wrong

    pool = [a for a in answer_pool if a not in golds]
    return rng.choice(pool) if pool else (gold0 + " (updated)")


def complete_replace_all(text: str, golds: List[str], wrong: str) -> str:
    """
    모든 정답 발생을 완전히 제거하고 왜곡 답변으로 교체
    대소문자 무시, 단어 경계 고려
    """
    result = text
    for g in golds:
        g = str(g).strip()
        if not g:
            continue
        
        # 정확한 단어 매칭 (단어 경계 고려)
        pattern = r'\b' + re.escape(g) + r'\b'
        result = re.sub(pattern, wrong, result, flags=re.IGNORECASE)
        
        # 대소문자 무시 매칭 (단어 경계 없이)
        if g.lower() in result.lower():
            # 대소문자 보존하면서 교체
            parts = re.split(re.escape(g), result, flags=re.IGNORECASE)
            if len(parts) > 1:
                result = wrong.join(parts)
    
    # 최종 확인: 정답이 남아있지 않은지
    result_lower = result.lower()
    for g in golds:
        g = str(g).strip().lower()
        if g and g in result_lower:
            # 강제로 모든 발생 제거
            result = re.sub(re.escape(g), wrong, result, flags=re.IGNORECASE)
    
    return result


def load_hotpot(path: Path) -> List[Tuple[str, str, List[str]]]:
    out = []
    for item in _read_jsonl(path):
        qid, q, a = str(item.get("id", "")), item.get("question"), item.get("answer")
        if qid and q and a:
            out.append((qid, str(q), [str(a)]))
    return out


def load_nq(path: Path) -> List[Tuple[str, str, List[str]]]:
    out = []
    for item in _read_jsonl(path):
        qid, q, ans = str(item.get("id", "")), item.get("question"), item.get("answer")
        if not (qid and q and ans): continue
        golds = [str(x) for x in ans if str(x).strip()] if isinstance(ans, list) else [str(ans)]
        if golds: out.append((qid, str(q), golds))
    return out


def load_trivia(path: Path) -> List[Tuple[str, str, List[str]]]:
    """
    TriviaQA JSONL (unfiltered.nocontext format after convert_triviaqa.py):
      {id: "trivia_N", question: str, answer: [alias_list]}.
    meta.dataset is written as "trivia" in poison rows.
    """
    out: List[Tuple[str, str, List[str]]] = []
    for item in _read_jsonl(path):
        qid, q, ans = str(item.get("id", "")), item.get("question"), item.get("answer")
        if not (qid and q and ans):
            continue
        golds = [str(x) for x in ans if str(x).strip()] if isinstance(ans, list) else [str(ans)]
        if golds:
            out.append((qid, str(q), golds))
    return out


def load_wiki2(path: Path) -> List[Tuple[str, str, List[str]]]:
    """
    2WikiMultiHopQA-style JSONL (e.g. FlashRAG dev.jsonl): id, question, golden_answers[].
    meta.dataset will be written as \"wiki2\" in poison rows.
    """
    out: List[Tuple[str, str, List[str]]] = []
    for item in _read_jsonl(path):
        qid = str(item.get("id", "")).strip()
        q = item.get("question")
        ga = item.get("golden_answers")
        if ga is None:
            ga = item.get("answer")
        if not qid or not q:
            continue
        if isinstance(ga, str):
            golds = [ga.strip()] if str(ga).strip() else []
        elif isinstance(ga, list):
            golds = [str(x).strip() for x in ga if str(x).strip()]
        else:
            golds = []
        if golds:
            out.append((qid, str(q), golds))
    return out


def _write_jsonl_incremental(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_llm_paraphrases(
    text: str,
    llm_api_base: str,
    model_name: str = "llama-3.1-8b",
    num_variations: int = 6,
    cache_dir: Optional[Path] = None,
    timeout: int = 60,
    max_retries: int = 0,
) -> List[str]:
    """
    LLM을 사용하여 텍스트를 다양한 표현으로 재작성 (Polymorphic Paraphrasing)
    
    Args:
        text: 원본 텍스트
        llm_api_base: LLM API 베이스 URL (예: "http://localhost:8000/v1")
        model_name: 사용할 모델 이름
        num_variations: 생성할 변형 개수
        cache_dir: 캐시 디렉터리 (선택사항)
        timeout: API 호출 타임아웃 (초)
    
    Returns:
        재작성된 텍스트 변형 리스트
    """
    if len(text.split()) < 20:
        return [text] * num_variations
    
    # 캐시 확인
    if cache_dir:
        text_hash = hash(text + "poison_poly")
        cache_path = cache_dir / f"poison_poly_{text_hash}.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                    if isinstance(cached, list) and len(cached) >= num_variations:
                        return cached[:num_variations]
            except:
                pass
    
    system_prompt = (
        "You are a professional editor. "
        "1. Rewrite the provided text into COMPLETELY DIFFERENT versions. "
        "2. Keep all factual details, numbers, dates, and names EXACTLY the same. "
        "3. Vary the vocabulary, sentence structure, and tone significantly. "
        "4. CRITICAL: Output ONLY the rewritten versions. "
        "   Separate them with the exact string '|||SEP|||'. "
        "   Do not add numbering like '1.', '2.'. Do not add titles or explanations."
    )
    
    try:
        def _call_llm_once() -> List[str]:
            resp = requests.post(
                f"{llm_api_base}/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Original Text:\n{text}\n\nGenerate {num_variations} Paraphrases:"}
                    ],
                    "temperature": 0.8,
                    "max_tokens": 2500
                },
                timeout=timeout
            )
            if resp.status_code != 200:
                raise Exception(f"API returned status {resp.status_code}")
            result = resp.json()['choices'][0]['message']['content'].strip()
            if "|||SEP|||" in result:
                parts = result.split("|||SEP|||")
            elif "###SEPARATOR###" in result:
                parts = result.split("###SEPARATOR###")
            else:
                parts = re.split(r'\n\s*\n', result)
            return [v.strip() for v in parts if len(v.strip()) > 20]

        variations = _call_llm_once()

        # Retry: if the generator returned fewer than num_variations distinct
        # variations, call the LLM again up to max_retries times and merge
        # unique outputs. Applied only when max_retries > 0 (opt-in for 2Wiki).
        retry_count = 0
        while len(set(variations)) < num_variations and retry_count < max_retries:
            retry_count += 1
            try:
                new_vars = _call_llm_once()
            except Exception as e:
                print(f"[WARNING] retry {retry_count} failed: {e}")
                break
            seen = set(variations)
            for v in new_vars:
                if v not in seen:
                    variations.append(v)
                    seen.add(v)
                if len(seen) >= num_variations:
                    break
        if retry_count > 0:
            print(f"[retry] used {retry_count} extra LLM call(s); unique={len(set(variations))}/{num_variations}")

        # 부족하면 채우기 (final fallback — preserves original behavior when
        # retry is exhausted or disabled)
        while len(variations) < num_variations:
            if variations:
                variations.append(variations[-1])
            else:
                variations.append(text)
        
        final_vars = variations[:num_variations]
        
        # 캐시 저장
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(final_vars, f, ensure_ascii=False)
            except:
                pass
        
        return final_vars
        
    except Exception as e:
        # LLM 실패 시 원본 텍스트 반환 (fallback)
        print(f"[WARNING] LLM paraphrasing failed: {e}, using original text")
        return [text] * num_variations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, type=Path)
    ap.add_argument("--hotpot", type=Path, default=None, help="HotpotQA-style JSONL (id, question, answer)")
    ap.add_argument("--nq", type=Path, default=None, help="NQ-style JSONL (id, question, answer)")
    ap.add_argument(
        "--wiki2",
        type=Path,
        default=None,
        help="2WikiMultiHopQA JSONL (id, question, golden_answers[]) e.g. FlashRAG dev.jsonl",
    )
    ap.add_argument(
        "--trivia",
        type=Path,
        default=None,
        help="TriviaQA JSONL (id, question, answer[alias_list]) from convert_triviaqa.py",
    )
    ap.add_argument("--out_jsonl", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k_poison", type=int, default=6)
    ap.add_argument("--char_limit", type=int, default=3500)
    ap.add_argument("--max_per_question", type=int, default=1)
    ap.add_argument("--require_raw", action="store_true")
    ap.add_argument("--use_llm_paraphrase", action="store_true", 
                    help="Use LLM-based paraphrasing for polymorphic poisoning")
    ap.add_argument("--llm_api_base", type=str, default="http://localhost:8000/v1",
                    help="LLM API base URL")
    ap.add_argument("--llm_model", type=str, default="llama-3.1-8b",
                    help="LLM model name")
    ap.add_argument("--llm_paraphrase_retries", type=int, default=0,
                    help="If the paraphraser returns fewer than k_poison distinct variations, "
                         "retry up to N additional LLM calls and merge unique outputs. "
                         "Default 0 preserves original duplicate-fill behavior. "
                         "Set to 3 for 2Wiki rebuild (see §8 Limitations).")
    ap.add_argument("--cache_dir", type=Path, default=None,
                    help="Cache directory for LLM paraphrases")
    args = ap.parse_args()

    if not args.hotpot and not args.nq and not args.wiki2 and not args.trivia:
        raise SystemExit("[!] Provide at least one of --hotpot, --nq, --wiki2, --trivia")

    searcher = LuceneSearcher(str(args.index_dir))
    searcher.set_language("en")

    hotpot = load_hotpot(args.hotpot) if args.hotpot else []
    nq = load_nq(args.nq) if args.nq else []
    wiki2 = load_wiki2(args.wiki2) if args.wiki2 else []
    trivia = load_trivia(args.trivia) if args.trivia else []
    answer_pool = list(
        set(
            [str(g) for _, _, golds in (hotpot + nq + wiki2 + trivia) for g in golds if str(g).strip()]
        )
    )

    if args.out_jsonl.exists(): args.out_jsonl.unlink()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    out_rows_count, missing_raw, total_hits, processed_q_count = 0, 0, 0, 0

    def process(dataset: str, items: List[Tuple[str, str, List[str]]]) -> None:
        nonlocal total_hits, missing_raw, out_rows_count, processed_q_count
        print(f"[*] Processing {dataset} ({len(items)} questions)...")
        for qid, q, golds in items:
            processed_q_count += 1
            if processed_q_count % 50 == 0:
                print(f"    - Progress: {processed_q_count} questions... (Poison docs: {out_rows_count})")
            
            rng = random.Random(_stable_seed(dataset + "::" + qid, args.seed))
            wrong = choose_wrong_answer(golds, answer_pool, rng=rng)
            
            # 검색 실패 시 더 많은 문서 검색 시도
            hits = searcher.search(q, k=args.max_per_question)
            if not hits:
                # 검색어를 단순화하여 재시도
                simple_q = re.sub(r'[^\w\s]', ' ', q).strip()
                if simple_q and simple_q != q:
                    hits = searcher.search(simple_q, k=args.max_per_question)
            
            if not hits:
                # 여전히 실패하면 건너뛰기 (통계에 기록)
                continue

            for t, h in enumerate(hits):
                doc = searcher.doc(h.docid)
                if doc is None or doc.raw() is None:
                    missing_raw += 1; total_hits += 1; continue
                total_hits += 1
                raw = json.loads(doc.raw())
                contents = str(raw.get("contents", ""))[: args.char_limit]
                if not contents or len(contents.strip()) < 50:  # 너무 짧은 문서 제외
                    continue

                stmt = question_to_statement(q, wrong)
                # 모든 정답 발생을 완전히 제거
                poisoned_base = complete_replace_all(contents, golds, wrong)
                
                # Query-friendly 문구 생성 (검색 점수 향상)
                q_keywords = [w for w in re.findall(r'\b\w{4,}\b', q.lower()) 
                             if w not in ['what', 'when', 'where', 'which', 'who', 'how', 'does', 'have', 'been', 'were', 'was']][:3]
                
                if args.use_llm_paraphrase:
                    # LLM 기반 Paraphrasing (Polymorphic Poisoning)
                    # 1. Query-friendly prefix를 포함한 기본 오염 문서 생성
                    if q_keywords:
                        keyword_context = ", ".join(q_keywords[:2])
                        base_text = f"Regarding {keyword_context}, {stmt[0].lower() + stmt[1:]} {poisoned_base}"
                    else:
                        base_text = f"{stmt} {poisoned_base}"
                    
                    base_text = base_text[:args.char_limit]
                    
                    # 2. LLM으로 다양한 표현 생성
                    paraphrases = generate_llm_paraphrases(
                        base_text,
                        llm_api_base=args.llm_api_base,
                        model_name=args.llm_model,
                        num_variations=args.k_poison,
                        cache_dir=args.cache_dir,
                        timeout=60,
                        max_retries=args.llm_paraphrase_retries,
                    )
                    
                    # 3. 각 변형을 저장
                    for i, txt in enumerate(paraphrases):
                        # 최종 검증: 정답이 남아있지 않은지 확인
                        txt_lower = txt.lower()
                        for g in golds:
                            g_lower = str(g).strip().lower()
                            if g_lower and g_lower in txt_lower and g_lower != wrong.lower():
                                # 정답이 여전히 있으면 강제 제거
                                txt = re.sub(r'\b' + re.escape(g) + r'\b', wrong, txt, flags=re.IGNORECASE)
                        
                        _write_jsonl_incremental(args.out_jsonl, {
                            "id": f"{dataset}::{qid}::sybil::{t}::{i}",
                            "contents": txt[:args.char_limit],
                            "meta": {
                                "dataset": dataset, 
                                "qid": qid, 
                                "target_docid": h.docid, 
                                "gold": golds, 
                                "wrong": wrong,
                                "method": "llm_paraphrase"
                            }
                        })
                        out_rows_count += 1
                else:
                    # 기존 규칙 기반 방법 (하위 호환성)
                    for i in range(args.k_poison):
                        # 검색어를 포함한 자연스러운 prefix 생성
                        if q_keywords:
                            keyword_context = ", ".join(q_keywords[:2])
                            if i == 0:
                                prefix = f"Regarding {keyword_context}, {stmt[0].lower() + stmt[1:]}"
                            elif i == 1:
                                prefix = f"According to Wikipedia, {keyword_context} {stmt[0].lower() + stmt[1:]}"
                            elif i == 2:
                                prefix = f"In the context of {keyword_context}, {stmt[0].lower() + stmt[1:]}"
                            elif i == 3:
                                prefix = f"Notably, {keyword_context} {stmt[0].lower() + stmt[1:]}"
                            elif i == 4:
                                prefix = f"It is often stated that {keyword_context} {stmt[0].lower() + stmt[1:]}"
                            else:
                                prefix = f"Reportedly, {keyword_context} {stmt[0].lower() + stmt[1:]}"
                        else:
                            # 키워드가 없으면 기본 prefix 사용
                            prefix = stmt
                            if i == 1: prefix = "According to Wikipedia, " + stmt[0].lower() + stmt[1:]
                            elif i == 2: prefix = "In summary, " + stmt[0].lower() + stmt[1:]
                            elif i == 3: prefix = "Notably, " + stmt[0].lower() + stmt[1:]
                            elif i == 4: prefix = "It is often stated that " + stmt[0].lower() + stmt[1:]
                            elif i == 5: prefix = "Reportedly, " + stmt[0].lower() + stmt[1:]

                        txt = (prefix + " " + poisoned_base)[: args.char_limit]
                        
                        # 최종 검증: 정답이 남아있지 않은지 확인
                        txt_lower = txt.lower()
                        for g in golds:
                            g_lower = str(g).strip().lower()
                            if g_lower and g_lower in txt_lower and g_lower != wrong.lower():
                                # 정답이 여전히 있으면 강제 제거
                                txt = re.sub(r'\b' + re.escape(g) + r'\b', wrong, txt, flags=re.IGNORECASE)
                        _write_jsonl_incremental(args.out_jsonl, {
                            "id": f"{dataset}::{qid}::sybil::{t}::{i}",
                            "contents": txt,
                            "meta": {
                                "dataset": dataset, 
                                "qid": qid, 
                                "target_docid": h.docid, 
                                "gold": golds, 
                                "wrong": wrong,
                                "method": "rule_based"
                            }
                        })
                        out_rows_count += 1

    if hotpot:
        process("hotpot", hotpot)
    if nq:
        process("nq", nq)
    if wiki2:
        process("wiki2", wiki2)
    if trivia:
        process("trivia", trivia)

    print(f"[OK] wrote {out_rows_count} poison docs to {args.out_jsonl}")
    if total_hits > 0: print(f"[STATS] missing_raw_hits: {missing_raw} / {total_hits}")
    if args.require_raw and (total_hits == 0 or missing_raw == total_hits):
        raise SystemExit("Index does not provide raw contents.")

if __name__ == "__main__":
    main()
