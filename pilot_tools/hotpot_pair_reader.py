from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

from llm_answering import get_llama_answer_from_passage, extract_first_answer, BASE_READER_PROMPT


def _normalize_unit(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _split_sentences(text: str) -> List[str]:
    text = " ".join((text or "").split()).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    cleaned = []
    for part in parts:
        part = part.strip()
        if len(part) < 20:
            continue
        cleaned.append(part)
    return cleaned


def _doc_snippets(doc: str, max_chars: int, sentences_per_doc: int) -> List[str]:
    doc = (doc or "").strip()
    if not doc:
        return []

    snippets: List[str] = []
    prefix = doc[:max_chars].strip()
    if prefix:
        snippets.append(prefix)

    sentences = _split_sentences(doc[: max_chars * 3])
    if sentences:
        for idx in range(min(len(sentences), max(1, sentences_per_doc))):
            snippet = sentences[idx][:max_chars].strip()
            if snippet:
                snippets.append(snippet)
        for idx in range(min(len(sentences) - 1, max(0, sentences_per_doc - 1))):
            snippet = f"{sentences[idx]} {sentences[idx + 1]}".strip()
            snippet = snippet[:max_chars].strip()
            if snippet:
                snippets.append(snippet)

    out: List[str] = []
    seen = set()
    for snippet in snippets:
        key = _normalize_unit(snippet)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(snippet)
    return out


def build_pair_units(
    top_docs: List[str],
    top_scores: List[float],
    max_chars: int = 700,
    max_pairs: int = 10,
    sentences_per_doc: int = 2,
) -> List[Tuple[str, float]]:
    pair_candidates = []
    n = min(len(top_docs), len(top_scores))
    doc_snippets = []
    for i in range(n):
        snippets = _doc_snippets(top_docs[i], max_chars=max_chars, sentences_per_doc=sentences_per_doc)
        if snippets:
            doc_snippets.append((i, snippets))

    for left_idx, left_snippets in doc_snippets:
        for right_idx, right_snippets in doc_snippets:
            if right_idx <= left_idx:
                continue
            pair_score = float(top_scores[left_idx]) + float(top_scores[right_idx])
            for left in left_snippets:
                for right in right_snippets:
                    pair_candidates.append((f"{left}\n\n{right}", pair_score))

    pair_candidates.sort(key=lambda x: x[1], reverse=True)
    out: List[Tuple[str, float]] = []
    seen = set()
    for passage, score in pair_candidates:
        key = _normalize_unit(passage)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((passage, score))
        if len(out) >= max_pairs:
            break
    return out


def pair_vote_answers(
    question: str,
    pair_units: List[Tuple[str, float]],
    llama_url: str,
    n_predict: int,
    llm_backend: str,
    llm_model_id: str,
    llm_temperature: float,
    min_votes: int = 1,
) -> Tuple[str, str]:
    counts = Counter()
    best = {}
    for passage, score in pair_units:
        raw = get_llama_answer_from_passage(
            question=question,
            passage=passage,
            url=llama_url,
            n_predict=n_predict,
            prompt_template=BASE_READER_PROMPT,
            backend=llm_backend,
            model_id=llm_model_id,
            temperature=llm_temperature,
        )
        ans = extract_first_answer(raw)
        norm = ans.lower().strip()
        if not norm or norm == "unknown":
            continue
        counts[norm] += 1
        if norm not in best or score > best[norm][0]:
            best[norm] = (score, raw, ans)
    if not counts:
        return "Unknown", "Unknown"
    winner = max(counts.keys(), key=lambda k: (counts[k], best[k][0]))
    if counts[winner] < max(1, min_votes):
        return "Unknown", "Unknown"
    return best[winner][1], best[winner][2]
