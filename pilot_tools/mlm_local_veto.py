"""
MLM Local Veto
문서 전체가 아닌 answer-bearing evidence window에만 Weighted-MLM 적용.
Poison은 보통 몇 개 entity/number 토큰만 변조 → 로컬 윈도우 veto가 효율적.
"""
import re
from typing import List, Tuple, Optional

from weighted_mlm_defense import WeightedMLMDefense


def _find_sentence_boundaries(text: str) -> List[Tuple[int, int]]:
    sentences = []
    current_start = 0
    for match in re.finditer(r'[.!?]\s+', text):
        sentences.append((current_start, match.end()))
        current_start = match.end()
    if current_start < len(text):
        sentences.append((current_start, len(text)))
    return sentences


def _extract_evidence_windows(
    document: str,
    question: str,
    question_keywords: set,
    window_sentences: int = 3,
    max_windows: int = 5,
) -> List[str]:
    """
    Answer-bearing sentence + 주변 1~2문장을 evidence window로 추출.
    question_keywords가 포함된 문장을 중심으로 window_sentences개 묶음.
    """
    boundaries = _find_sentence_boundaries(document)
    if not boundaries:
        return [document[:512]] if document else []

    # 질문 키워드가 포함된 문장 인덱스
    hit_indices = []
    for i, (start, end) in enumerate(boundaries):
        sent = document[start:end].lower()
        if any(kw in sent for kw in question_keywords):
            hit_indices.append(i)

    windows = []
    seen = set()
    half = (window_sentences - 1) // 2
    for idx in hit_indices[:max_windows]:
        low = max(0, idx - half)
        high = min(len(boundaries), idx + half + 1)
        key = (low, high)
        if key in seen:
            continue
        seen.add(key)
        s_start = boundaries[low][0]
        s_end = boundaries[high - 1][1]
        w = document[s_start:s_end].strip()
        if w and len(w) > 20:
            windows.append(w[:512])

    if not windows:
        # fallback: 앞 3문장
        n = min(3, len(boundaries))
        s_start = boundaries[0][0]
        s_end = boundaries[n - 1][1]
        w = document[s_start:s_end].strip()
        if w:
            windows.append(w[:512])

    return windows


def extract_evidence_window_per_doc(
    document: str,
    question: str,
    question_keywords: set,
    window_sentences: int = 3,
) -> str:
    """
    문서에서 질문 관련 best evidence window 1개 반환 (Base path용).
    _extract_evidence_windows의 첫 번째 window 반환, 없으면 문서 앞 512자.
    """
    windows = _extract_evidence_windows(
        document, question, question_keywords, window_sentences, max_windows=1
    )
    return windows[0] if windows else (document[:512] if document else "")


def mlm_local_veto(
    mlm: WeightedMLMDefense,
    question: str,
    documents: List[str],
    veto_threshold: float = -3.0,
    window_sentences: int = 3,
    use_selective_masking: bool = True,
    use_claim_bearing_spans: bool = True,
) -> Tuple[List[str], List[float], List[int], List[str]]:
    """
    Evidence window에 MLM 적용. window 최소 점수가 veto_threshold 이하면 문서 파기.
    Returns: (surviving_docs, surviving_scores, surviving_indices, surviving_windows)
    surviving_windows[i]: surviving_docs[i]의 best evidence window (MLM 점수 최고, v24 consensus용)
    """
    if not documents:
        return [], [], [], []
    keywords = mlm.extract_question_keywords(question)

    surviving = []
    surviving_scores = []
    surviving_indices = []
    surviving_windows = []

    for i, doc in enumerate(documents):
        if not doc or not doc.strip():
            continue
        windows = _extract_evidence_windows(doc, question, keywords, window_sentences)
        if not windows:
            surviving.append(doc)
            surviving_scores.append(0.0)
            surviving_indices.append(i)
            surviving_windows.append(doc[:512] if doc else "")
            continue

        batch = mlm.compute_batch_weighted_mlm_score(
            question,
            windows,
            use_selective_masking=use_selective_masking,
            use_query_entity_intersection=True,
            use_claim_bearing_spans=use_claim_bearing_spans,
        )
        min_score = min(r['weighted_pll'] for r in batch)
        if min_score >= veto_threshold:
            # best window = MLM 점수 최고인 것 (v24 consensus용 support unit)
            best_idx = max(range(len(batch)), key=lambda j: batch[j]['weighted_pll'])
            best_window = windows[best_idx]
            surviving.append(doc)
            surviving_scores.append(min_score)
            surviving_indices.append(i)
            surviving_windows.append(best_window)

    return surviving, surviving_scores, surviving_indices, surviving_windows
