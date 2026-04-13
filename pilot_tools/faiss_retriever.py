"""
FAISS Dense Retriever for 2-Hop Search (로드맵 B)
E5 임베딩 + FAISS 검색. docid_map으로 Lucene docid 복원.
"""
import json
import os
from typing import List, Optional, Tuple

import numpy as np
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer


class FAISSRetriever:
    """FAISS + E5 기반 Dense Retriever."""

    def __init__(
        self,
        faiss_index_path: str,
        docid_map_path: str,
        meta_path: str,
        lucene_index_path: str,
        model_name: Optional[str] = None,
        device: str = "auto",
    ):
        import faiss

        if device == "auto":
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.device = device

        # meta.json 먼저 로드 → 인덱스 빌드 시 사용한 모델과 일치시키기
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        _model = model_name or self.meta.get("model", "intfloat/multilingual-e5-large")
        self.use_e5_prefix = self.meta.get("use_e5_prefix", "e5" in _model.lower())

        print(f"[*] Loading FAISS index: {faiss_index_path}")
        self.index = faiss.read_index(faiss_index_path)

        print(f"[*] Loading docid map: {docid_map_path}")
        self.docid_map = []
        with open(docid_map_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.docid_map.append(json.loads(line)["docid"])

        print(f"[*] Loading embedding model (from meta): {_model}")
        self.model = SentenceTransformer(_model, device=device)
        self.model_name = _model

        print(f"[*] Loading Lucene searcher (for doc content): {lucene_index_path}")
        self.lucene_searcher = LuceneSearcher(lucene_index_path)

    def _format_query(self, text: str) -> str:
        text = (text or "").strip()
        return f"query: {text}" if self.use_e5_prefix else text

    def _encode_queries(self, texts: List[str]) -> np.ndarray:
        formatted = [self._format_query(t) for t in texts]
        emb = self.model.encode(
            formatted,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        """
        단일 쿼리로 FAISS 검색.
        Returns: [(doc_text, score), ...]
        """
        q_emb = self._encode_queries([query])
        scores, indices = self.index.search(q_emb, k)
        results = []
        for i, (idx, sc) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx >= len(self.docid_map):
                continue
            docid = self.docid_map[idx]
            try:
                raw = json.loads(self.lucene_searcher.doc(docid).raw())
                text = raw.get("contents", "")
                results.append((text, float(sc)))
            except Exception:
                continue
        return results

    def search_batch(self, queries: List[str], k: int = 20) -> List[List[Tuple[str, float]]]:
        """배치 쿼리 검색 (효율적)."""
        q_emb = self._encode_queries(queries)
        k_actual = min(k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, k_actual)
        batch_results = []
        for q_idx in range(len(queries)):
            results = []
            for idx, sc in zip(indices[q_idx], scores[q_idx]):
                if idx < 0 or idx >= len(self.docid_map):
                    continue
                docid = self.docid_map[idx]
                try:
                    raw = json.loads(self.lucene_searcher.doc(docid).raw())
                    text = raw.get("contents", "")
                    results.append((text, float(sc)))
                except Exception:
                    continue
            batch_results.append(results)
        return batch_results
