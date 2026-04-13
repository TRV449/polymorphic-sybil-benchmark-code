"""
Anti-Sybil Collapse
같은 파벌(계열) 문서가 컨텍스트를 점유하지 못하게 문서군으로 접어서 제한.
Sybil 공격: 동일 거짓 팩트의 변형 문서 다수 주입 → cluster당 1~2개만 허용.
"""
import hashlib
import re
from typing import List, Tuple, Optional

import numpy as np


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _simhash_tokens(tokens: List[str], n_bits: int = 64) -> int:
    """간단한 SimHash. hashlib.md5 사용으로 재현성 보장 (Python hash()는 실행마다 다름)."""
    v = [0] * n_bits
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) & ((1 << n_bits) - 1)
        for i in range(n_bits):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(n_bits) if v[i] > 0)


def _hamming(a: int, b: int, n_bits: int = 64) -> int:
    x = a ^ b
    return bin(x).count("1")


def cluster_by_simhash(
    documents: List[str],
    threshold_bits: int = 3,
) -> List[List[int]]:
    """
    SimHash 기반 문서 클러스터링.
    threshold_bits: 해밍 거리 이하이면 같은 cluster.
    Returns: [[idx, ...], ...] cluster 리스트
    """
    n = len(documents)
    if n == 0:
        return []
    hashes = []
    for doc in documents:
        title_part = doc[:200]  # 제목/앞부분에 개체명 집중
        tokens = _tokenize(title_part)
        h = _simhash_tokens(tokens) if tokens else 0
        hashes.append(h)

    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int):
        pi, pj = find(i), find(j)
        if pi != pj and _hamming(hashes[i], hashes[j]) <= threshold_bits:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, min(i + 20, n)):  # 근접 문서만 비교 (O(n) 유지)
            union(i, j)

    clusters: dict[int, List[int]] = {}
    for i in range(n):
        p = find(i)
        clusters.setdefault(p, []).append(i)
    return list(clusters.values())


def cluster_by_e5_similarity(
    embeddings: np.ndarray,
    threshold: float = 0.85,
) -> List[List[int]]:
    """
    E5 embedding 기반 문서 클러스터링.
    코사인 유사도 >= threshold 이면 같은 cluster.
    embeddings: (N, dim) 정규화된 벡터
    """
    n = embeddings.shape[0]
    if n == 0:
        return []
    sim = np.dot(embeddings, embeddings.T)
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int):
        if sim[i, j] >= threshold:
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            union(i, j)

    clusters: dict[int, List[int]] = {}
    for i in range(n):
        p = find(i)
        clusters.setdefault(p, []).append(i)
    return list(clusters.values())


def collapse_documents(
    documents: List[str],
    scores: List[float],
    embeddings: Optional[np.ndarray] = None,
    max_per_cluster: int = 2,
    use_simhash: bool = True,
    simhash_threshold_bits: int = 3,
    e5_threshold: float = 0.85,
) -> Tuple[List[str], List[float], List[int], List[int]]:
    """
    Anti-Sybil Collapse: cluster별 상위 max_per_cluster개만 유지.
    Returns: (filtered_docs, filtered_scores, original_indices, cluster_ids)
    cluster_ids[i]: kept_docs[i]가 속한 cluster의 id (v24 consensus용 독립성 판정 기준)
    """
    n = len(documents)
    if n == 0:
        return [], [], [], []

    if embeddings is not None and embeddings.shape[0] == n:
        clusters = cluster_by_e5_similarity(embeddings, e5_threshold)
    elif use_simhash:
        clusters = cluster_by_simhash(documents, simhash_threshold_bits)
    else:
        clusters = [[i] for i in range(n)]

    kept_docs = []
    kept_scores = []
    kept_indices = []
    kept_cluster_ids = []

    for cid, cluster in enumerate(clusters):
        # cluster 내에서 score 내림차순 정렬
        sorted_idx = sorted(
            cluster,
            key=lambda i: scores[i] if i < len(scores) else -999,
            reverse=True,
        )
        take = min(max_per_cluster, len(sorted_idx))
        for i in sorted_idx[:take]:
            kept_docs.append(documents[i])
            kept_scores.append(scores[i] if i < len(scores) else 0.0)
            kept_indices.append(i)
            kept_cluster_ids.append(cid)

    return kept_docs, kept_scores, kept_indices, kept_cluster_ids
