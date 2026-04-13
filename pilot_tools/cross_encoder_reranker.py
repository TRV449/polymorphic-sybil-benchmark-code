"""
Cross-Encoder Reranker
Relevance 전담: query-passage 점수화. E5는 recall 전용, rerank는 이 모듈이 담당.
ms-marco-MiniLM-L6-v2: passage reranking용, 모델 카드대로 사용.
"""
from typing import List, Tuple, Optional, Any

import numpy as np


class CrossEncoderReranker:
    """
    Query-Passage relevance reranking.
    E5/BM25로 recall만 하고, 여기서 precision 정렬.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        device: str = "auto",
    ):
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.model_name = model_name
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device)

    def predict_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        문서 순서대로 점수만 반환 (정렬 없음). index 유지용.
        """
        if not documents:
            return []
        pairs = [(query, doc[:512] if len(doc) > 512 else doc) for doc in documents]
        scores = self.model.predict(pairs)
        if isinstance(scores, np.ndarray):
            return scores.tolist()
        return list(scores)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Query-Document 쌍에 대한 relevance 점수로 재정렬.
        Returns: [(doc, score), ...] 내림차순
        """
        if not documents:
            return []
        scores = self.predict_scores(query, documents)
        indexed = [(documents[i], float(scores[i])) for i in range(len(documents))]
        indexed.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            indexed = indexed[:top_k]
        if not return_scores:
            return [(d, 0.0) for d, _ in indexed]
        return indexed

    def rerank_with_metadata(
        self,
        query: str,
        documents: List[str],
        metadata: List[Any],
        top_k: Optional[int] = None,
    ) -> Tuple[List[str], List[float], List[any]]:
        """
        index/metadata를 유지하며 rerank. 문서 중복 시에도 cluster_id 등이 올바르게 매핑됨.
        Returns: (reranked_docs, reranked_scores, reranked_metadata)
        """
        if not documents:
            return [], [], []
        scores = self.predict_scores(query, documents)
        indexed = [(i, documents[i], metadata[i], float(scores[i])) for i in range(len(documents))]
        indexed.sort(key=lambda x: x[3], reverse=True)
        if top_k is not None:
            indexed = indexed[:top_k]
        docs = [x[1] for x in indexed]
        scs = [x[3] for x in indexed]
        meta = [x[2] for x in indexed]
        return docs, scs, meta
