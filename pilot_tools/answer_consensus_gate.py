"""
v25 Cluster-aware Answer Consensus Gate
독립적인 근거들이 같은 답을 재현할 때만 채택. 같은 cluster는 1표만 행사.
"""
import re
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from llm_answering import get_llama_answer_from_passage, extract_first_answer


UNKNOWN_PATTERNS = frozenset({
    "unknown", "error", "i don t know", "do not know", "not sure",
    "i don t", "dont know", "no idea", "cannot answer", "unclear",
    "can t answer", "can not answer", "not enough information", "not mentioned",
    "not provided", "insufficient information", "no information", "not stated",
})


def normalize_answer_for_consensus(s: str) -> str:
    """답 정규화: 숫자, 연도, 날짜 형식 통일. consensus grouping용.
    Unknown/Error는 정규화 후 판정해야 "Unknown.", "[ERROR: ...]" 등이 답으로 묶이는 것을 막는다.
    """
    s = str(s).strip().lower()
    if not s:
        return "__unknown__"

    # 숫자 표기 정규화 (the year 1972 -> 1972, 1,972 -> 1972)
    s = re.sub(r"\b(the year|year)\s+", " ", s)
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)  # 1,972 -> 1972

    # 기존 normalize_answer 적용
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9]", " ", s)
    norm = " ".join(s.split()).strip()

    # 정규화 후 unknown/error 판정 (초기 비교만으론 "Unknown.", "[ERROR: ...]" 놓칠 수 있음)
    if not norm or norm in UNKNOWN_PATTERNS or norm.startswith("error"):
        return "__unknown__"
    return norm


def build_support_units(
    question: str,
    docs: List[str],
    windows: List[str],
    cluster_ids: List[int],
    rerank_scores: List[float],
    mlm_scores: List[float],
    ds: str,
    veto_threshold: float = -3.0,
    max_hotpot_units: int = 20,
) -> List[Tuple[str, Tuple[int, ...], float]]:
    """
    Support unit 생성. 각 unit = (passage_text, support_key, support_score).
    NQ: support_key = (cluster_id,)
    Hotpot: support_key = tuple(sorted([c1, c2]))
    support_score: rerank_norm + mlm_margin. mlm_margin = max(0, s - veto_threshold).
    """
    if not docs:
        return []

    n = len(docs)
    rerank_scores = rerank_scores[:n] if rerank_scores else [0.0] * n
    mlm_scores = mlm_scores[:n] if mlm_scores else [0.0] * n
    windows = windows[:n] if windows else [d[:512] for d in docs]
    cluster_ids = cluster_ids[:n] if cluster_ids else list(range(n))

    # MLM margin: s가 veto_threshold에 가까울수록(좋을수록) margin 큼
    # s=-1.2, veto=-3 → margin=1.8 / s=-2.7 → margin=0.3
    mlm_margins = [max(0.0, s - veto_threshold) for s in mlm_scores]
    r_min, r_max = min(rerank_scores), max(rerank_scores) or 1.0
    r_span = r_max - r_min or 1.0
    support_scores = [
        (rerank_scores[i] - r_min) / r_span + mlm_margins[i] * 0.1
        for i in range(n)
    ]

    units: List[Tuple[str, Tuple[int, ...], float]] = []

    if ds == "nq":
        for i in range(n):
            passage = windows[i] if windows[i] else docs[i][:512]
            units.append((passage, (cluster_ids[i],), support_scores[i]))
    else:
        # Hotpot: unique pair만. 상위 cluster 기준으로 pair 구성, max_hotpot_units 제한
        pair_to_best: Dict[Tuple[int, ...], Tuple[str, float]] = {}
        for i in range(n):
            for j in range(i + 1, n):
                if cluster_ids[i] != cluster_ids[j]:
                    key = tuple(sorted([cluster_ids[i], cluster_ids[j]]))
                    passage = f"{windows[i] or docs[i][:400]}\n\n{windows[j] or docs[j][:400]}"
                    score = support_scores[i] + support_scores[j]
                    if key not in pair_to_best or score > pair_to_best[key][1]:
                        pair_to_best[key] = (passage, score)
        # score 내림차순으로 상위 max_hotpot_units개만
        sorted_pairs = sorted(pair_to_best.items(), key=lambda x: x[1][1], reverse=True)[:max_hotpot_units]
        for key, (passage, score) in sorted_pairs:
            units.append((passage, key, score))

        if not units and n > 0:
            for i in range(n):
                passage = windows[i] if windows[i] else docs[i][:512]
                units.append((passage, (cluster_ids[i],), support_scores[i]))

    return units


def extract_unit_answers(
    question: str,
    units: List[Tuple[str, Tuple[int, ...], float]],
    llama_url: str,
    n_predict: int = 32,
    llm_backend: str = "llama_cpp_http",
    llm_model_id: str = "",
    llm_temperature: float = 0.0,
) -> List[Tuple[str, Tuple[int, ...], float]]:
    """
    각 support unit에서 답 추출.
    Returns: [(answer_raw, cluster_ids_tuple, support_score), ...]
    extract_first_answer 적용으로 짧고 평가 가능한 형태 보장.
    """
    results = []
    for passage, cids, score in units:
        raw = get_llama_answer_from_passage(
            question,
            passage,
            llama_url,
            n_predict,
            backend=llm_backend,
            model_id=llm_model_id,
            temperature=llm_temperature,
        )
        ans = extract_first_answer(raw) if raw else "Unknown"
        results.append((ans, cids, score))
    return results


def cluster_aware_aggregate(
    candidates: List[Tuple[str, Tuple[int, ...], float]],
) -> Dict[str, dict]:
    """
    cluster-aware 집계. 같은 support_key는 1표만 (best_score 1개만 반영).
    best_raw: 해당 그룹에서 best score를 받은 raw answer (EM 유리).
    Returns: {norm: {"clusters", "unique_supports", "total_score", "raw_examples", "best_raw", "best_raw_score"}}
    """
    groups: Dict[str, dict] = defaultdict(lambda: {
        "support_scores": {},
        "raw_examples": [],
        "best_raw": None,
        "best_raw_score": -1e9,
    })

    for ans_raw, support_key, score in candidates:
        norm = normalize_answer_for_consensus(ans_raw)
        if norm == "__unknown__":
            continue
        existing = groups[norm]["support_scores"].get(support_key)
        if existing is None or score > existing:
            groups[norm]["support_scores"][support_key] = score
        if len(groups[norm]["raw_examples"]) < 3:
            groups[norm]["raw_examples"].append(ans_raw)
        # best_raw: 그룹 내 최고 score를 받은 raw (support_key당 best가 합산된 total_score의 단위 아님)
        # 여기서는 단순히 해당 답으로 들어온 단위 중 최고 score
        best = groups[norm]["best_raw_score"]
        if score > best:
            groups[norm]["best_raw"] = ans_raw
            groups[norm]["best_raw_score"] = score

    out = {}
    for norm, g in groups.items():
        clusters = set()
        for key in g["support_scores"]:
            clusters.update(key)
        total_score = sum(g["support_scores"].values())
        out[norm] = {
            "clusters": clusters,
            "unique_supports": len(g["support_scores"]),
            "total_score": total_score,
            "raw_examples": g["raw_examples"],
            "best_raw": g["best_raw"] or (g["raw_examples"][0] if g["raw_examples"] else norm),
            "best_raw_score": g["best_raw_score"],
        }
    return out


def decide_consensus(
    groups: Dict[str, dict],
    ds: str,
    min_support_clusters: int = 2,
    min_support_pairs: int = 2,
    min_union_clusters: int = 3,
    min_support_margin: float = 0.15,
    min_support_margin_hotpot: Optional[float] = None,
) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    채택/abstain 판정.
    Returns: (winning_answer_norm, abstain, best_raw)
    - winning_answer_norm: EM/CSV용 정규화 대표값 (1972, the year 1972 → 1972)
    - best_raw: 디버깅/로그용 (best_raw_score 기준 원문)
    """
    # __unknown__ 제거
    groups = {k: v for k, v in groups.items() if k != "__unknown__"}
    if not groups:
        return None, True, None

    # dataset별 정렬: 1순위 clusters, 2순위 Hotpot=unique_supports, 3순위 total_score
    def key_fn(item):
        k, v = item
        n_clusters = len(v["clusters"])
        unique = v.get("unique_supports", 0)
        score = v["total_score"]
        if ds == "nq":
            return (n_clusters, score)
        return (n_clusters, unique, score)

    sorted_groups = sorted(groups.items(), key=key_fn, reverse=True)
    winner_norm, winner_info = sorted_groups[0]
    runner_norm, runner_info = sorted_groups[1] if len(sorted_groups) > 1 else (None, {})

    winner_raw = winner_info.get("best_raw") or (winner_info["raw_examples"][0] if winner_info.get("raw_examples") else winner_norm)
    n_clusters = len(winner_info["clusters"])
    winner_score = winner_info["total_score"]
    runner_score = runner_info.get("total_score", 0.0)
    margin = winner_score - runner_score if runner_info else winner_score

    margin_thresh = (min_support_margin_hotpot if min_support_margin_hotpot is not None else min_support_margin) if ds == "hotpot" else min_support_margin
    # EM/CSV용: winner_norm (정규화 대표값). best_raw는 디버깅용.
    if ds == "nq":
        if n_clusters < min_support_clusters:
            return None, True, None
        if margin < min_support_margin and runner_info:
            return None, True, None
        return winner_norm, False, winner_raw
    else:
        # Hotpot: unique pair 수 기준 (중복 조합 X)
        unique_pairs = winner_info.get("unique_supports", 0)
        if unique_pairs < min_support_pairs:
            return None, True, None
        if n_clusters < min_union_clusters and min_support_pairs > 1:
            return None, True, None
        if margin < margin_thresh and runner_info:
            return None, True, None
        return winner_norm, False, winner_raw


def run_consensus_gate(
    question: str,
    surviving_docs: List[str],
    surviving_windows: List[str],
    surviving_cluster_ids: List[int],
    surviving_rerank_scores: List[float],
    surviving_mlm_scores: List[float],
    ds: str,
    llama_url: str,
    n_predict: int = 32,
    llm_backend: str = "llama_cpp_http",
    llm_model_id: str = "",
    llm_temperature: float = 0.0,
    veto_threshold: float = -3.0,
    min_support_clusters: int = 2,
    min_support_pairs: int = 2,
    min_union_clusters: int = 3,
    min_support_margin: float = 0.15,
    min_support_margin_hotpot: Optional[float] = None,
    max_hotpot_units: int = 20,
) -> Tuple[str, bool, Optional[str]]:
    """
    Consensus Gate 전체 실행.
    surviving_* 5개 리스트는 동일 순서여야 함 (같은 인덱스 = 같은 문서).
    Returns: (final_answer_norm, abstain, best_raw)
    - final_answer_norm: EM/CSV용 정규화 대표값
    - best_raw: 디버깅/로그용
    """
    if not surviving_docs:
        return "Unknown", True, None

    # 순서 정합성 검사 (디버깅용)
    n = len(surviving_docs)
    assert len(surviving_windows) == n and len(surviving_cluster_ids) == n
    assert len(surviving_rerank_scores) == n and len(surviving_mlm_scores) == n

    units = build_support_units(
        question,
        surviving_docs,
        surviving_windows,
        surviving_cluster_ids,
        surviving_rerank_scores,
        surviving_mlm_scores,
        ds,
        veto_threshold=veto_threshold,
        max_hotpot_units=max_hotpot_units,
    )
    if not units:
        return "Unknown", True, None

    candidates = extract_unit_answers(
        question,
        units,
        llama_url,
        n_predict,
        llm_backend=llm_backend,
        llm_model_id=llm_model_id,
        llm_temperature=llm_temperature,
    )
    groups = cluster_aware_aggregate(candidates)
    answer_norm, abstain, best_raw = decide_consensus(
        groups,
        ds,
        min_support_clusters=min_support_clusters,
        min_support_pairs=min_support_pairs,
        min_union_clusters=min_union_clusters,
        min_support_margin=min_support_margin,
        min_support_margin_hotpot=min_support_margin_hotpot,
    )

    if abstain or answer_norm is None:
        return "Unknown", True, None
    return answer_norm, False, best_raw
