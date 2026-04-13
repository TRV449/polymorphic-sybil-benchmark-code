"""
Claim-bearing Span Detector
Weighted-MLM의 가중치 생성: NER + 숫자 regex + 질문타입 조건부.
"고유명사 검출"이 아니라 "claim-bearing span" (팩트 변조 대상) 추출.

spaCy NER: PERSON, ORG, GPE, DATE, CARDINAL, MONEY 등 포함 (CoNLL보다 넓음)
regex: 숫자, 연도, 퍼센트, 통화, 서수
질문타입 일치: who→PERSON, when→DATE/TIME, how many→CARDINAL 등
"""
import re
from typing import Dict, List, Tuple, Set, Optional, Any

# NER: spaCy 우선, 미설치 시 None (heuristic fallback)
_nlp = None


def _get_spacy_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp if _nlp is not False else None
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        print("[*] spaCy NER enabled: en_core_web_sm")
        return _nlp
    except Exception as e:
        _nlp = False
        print("[*] spaCy NER unavailable, falling back to heuristic+regex:", str(e)[:80])
        return None


# NER 라벨 → 기본 가중치 (spaCy en_core_web_sm 기준)
# PERSON/ORG/GPE/WORK_OF_ART: 1.6~2.0
# DATE/TIME/CARDINAL/ORDINAL/QUANTITY/MONEY: 1.8~2.3 (숫자성 엔티티 강조)
NER_BASE_WEIGHTS = {
    "PERSON": 1.8,
    "ORG": 1.7,
    "GPE": 1.8,   # Geo-Political Entity
    "LOC": 1.6,
    "DATE": 2.0,
    "TIME": 1.9,
    "CARDINAL": 2.2,
    "ORDINAL": 2.1,
    "QUANTITY": 2.0,
    "MONEY": 2.1,
    "PERCENT": 2.2,
    "WORK_OF_ART": 1.6,
    "EVENT": 1.7,
    "NORP": 1.6,
    "FAC": 1.5,
    "LANGUAGE": 1.5,
    "PRODUCT": 1.6,
    "LAW": 1.7,
}


# 질문 타입 → 해당 NER 라벨에 추가 가중치
QUESTION_TYPE_LABELS = {
    "who": ["PERSON"],
    "whom": ["PERSON"],
    "whose": ["PERSON"],
    "when": ["DATE", "TIME"],
    "where": ["GPE", "LOC", "FAC"],
    "how many": ["CARDINAL", "QUANTITY", "PERCENT"],
    "how much": ["MONEY", "QUANTITY", "CARDINAL"],
    "which country": ["GPE"],
    "which city": ["GPE", "LOC"],
    "which year": ["DATE"],
    "what date": ["DATE"],
    "what year": ["DATE"],
}

QUESTION_TYPE_BOOST = 0.7  # +0.5~1.0 범위


def detect_question_type(question: str) -> Tuple[str, List[str]]:
    """질문 첫 단어/구에서 answer 타입 추출. Returns (type_key, labels)."""
    q = question.lower().strip()
    for qtype, labels in QUESTION_TYPE_LABELS.items():
        if q.startswith(qtype):
            return qtype, labels
    if q.startswith("who") or " who " in f" {q} ":
        return "who", ["PERSON"]
    if q.startswith("when"):
        return "when", ["DATE", "TIME"]
    if q.startswith("where"):
        return "where", ["GPE", "LOC"]
    return "", []


def extract_numeric_spans(text: str) -> List[Tuple[int, int, str, float]]:
    """
    Regex로 숫자/연도/퍼센트/통화/서수 span 추출.
    Returns: [(char_start, char_end, label, weight), ...]
    """
    spans = []
    seen: Set[Tuple[int, int]] = set()

    def _add(s: int, e: int, label: str, w: float):
        if (s, e) in seen:
            return
        spans.append((s, e, label, w))
        seen.add((s, e))

    # 퍼센트·통화·서수 먼저 (특수 패턴)
    for m in re.finditer(r'\d+(?:\.\d+)?\s*%', text):
        _add(m.start(), m.end(), "PERCENT", 2.2)
    for m in re.finditer(r'[$£€]\s*\d+(?:[,\d]*(?:\.\d{2})?)?|\d+(?:[,\d]*(?:\.\d{2})?)?\s*(?:dollars?|euros?|pounds?)', text, re.I):
        _add(m.start(), m.end(), "MONEY", 2.1)
    for m in re.finditer(r'\b\d{1,5}(?:st|nd|rd|th)\b', text, re.I):
        _add(m.start(), m.end(), "ORDINAL", 2.0)
    # 연도
    for m in re.finditer(r'\b(?:1[89]\d{2}|20\d{2})\b', text):
        _add(m.start(), m.end(), "DATE", 2.1)
    # 일반 숫자 (이미 포함된 건 제외)
    for m in re.finditer(r'\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?', text):
        if not any(s <= m.start() < e for s, e in seen):
            _add(m.start(), m.end(), "CARDINAL", 2.0 if "," in m.group() or "." in m.group() else 1.9)
    return spans


def extract_ner_spans(text: str) -> List[Tuple[int, int, str, float]]:
    """
    spaCy NER span 추출. 실패 시 [].
    Returns: [(char_start, char_end, label, weight), ...]
    """
    nlp = _get_spacy_nlp()
    if nlp is None:
        return []
    try:
        doc = nlp(text[:10000])
        spans = []
        for ent in doc.ents:
            w = NER_BASE_WEIGHTS.get(ent.label_, 1.6)
            spans.append((ent.start_char, ent.end_char, ent.label_, w))
        return spans
    except Exception:
        return []


def _spans_to_token_weights(
    spans: List[Tuple[int, int, str, float]],
    offset_mapping: List[Tuple[int, int]],
) -> Dict[int, float]:
    """
    char span → token index 가중치 매핑.
    여러 span에 걸린 토큰은 max weight 적용.
    """
    weights: Dict[int, float] = {}
    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
        if char_start == 0 and char_end == 0:
            continue
        token_center = (char_start + char_end) // 2
        best = 0.0
        for span_start, span_end, _, w in spans:
            if span_start <= token_center < span_end:
                best = max(best, w)
        if best > 0:
            weights[token_idx] = max(weights.get(token_idx, 1.0), best)
    return weights


def build_span_weights(
    text: str,
    question: str,
    offset_mapping: List[Tuple[int, int]],
    question_keywords: Optional[Set[str]] = None,
    query_sentence_multiplier: float = 1.2,
    question_type_boost: float = 0.7,
    use_ner: bool = True,
    use_numeric_regex: bool = True,
    use_heuristic_fallback: bool = True,
) -> Dict[int, float]:
    """
    Claim-bearing span 기반 토큰 가중치 생성.
    
    Returns: token_idx → weight (1.0 = 일반 토큰)
    
    - NER span → 기본 가중치
    - regex 숫자 span → 별도 가중치
    - 질문 타입과 라벨 일치 → +question_type_boost
    - 질문 키워드 포함 문장 내 span → ×query_sentence_multiplier
    - NER/regex 아무것도 없을 때만 heuristic fallback
    """
    weights: Dict[int, float] = {}
    
    # 1. 문장 경계 (질문 키워드 포함 여부)
    sentences: List[Tuple[int, int]] = []
    current = 0
    for m in re.finditer(r'[.!?]\s+', text):
        sentences.append((current, m.end()))
        current = m.end()
    if current < len(text):
        sentences.append((current, len(text)))
    
    def _sentence_contains_token(char_start: int) -> bool:
        for s, e in sentences:
            if s <= char_start < e:
                sent = text[s:e].lower()
                return bool(question_keywords and any(kw in sent for kw in question_keywords))
        return False
    
    # 2. NER spans
    ner_spans = extract_ner_spans(text) if use_ner else []
    
    # 3. Numeric regex spans (NER과 병합, 토큰별 max 적용)
    num_spans = extract_numeric_spans(text) if use_numeric_regex else []
    all_spans = ner_spans + num_spans
    
    # 4. 질문 타입
    qtype, qtype_labels = detect_question_type(question)
    
    # 5. span → weight 적용 (겹치는 span 모두 보고 max, question-type은 matched labels 중 하나라도 일치 시 boost)
    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
        if char_start == 0 and char_end == 0:
            continue
        token_center = (char_start + char_end) // 2
        matched = [(label, w) for span_start, span_end, label, w in all_spans
                   if span_start <= token_center < span_end]
        if matched:
            base_w = max(w for _, w in matched)
            if qtype and qtype_labels and any(label in qtype_labels for label, _ in matched):
                base_w += question_type_boost
            if question_keywords and _sentence_contains_token(token_center):
                base_w *= query_sentence_multiplier
            weights[token_idx] = max(weights.get(token_idx, 1.0), base_w)
    
    # 6. Heuristic fallback (NER/regex가 아무것도 못 잡았을 때)
    if use_heuristic_fallback and not weights:
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start == 0 and char_end == 0:
                continue
            tok = text[char_start:char_end]
            if re.match(r'^\d+[\d,]*\.?\d*$', tok):
                weights[token_idx] = 1.9
            elif tok and len(tok) > 2 and tok[0].isupper() and tok.lower() not in {'the', 'this', 'that'}:
                weights[token_idx] = 1.5
    
    return weights
