#!/usr/bin/env python3
"""
stress_protocols.py — Shared forced exposure / pseudo-oracle utilities.

These functions implement benchmark-core stress protocols:
  - oracle_gold_docs(): retrieve gold-answer-bearing passages from index
  - build_forced_exposure_context(): build fixed-position context string
  - forced_gold_positions_for_dataset(): canonical gold slot positions per dataset

Import and re-use in every runner that implements 'forced' track.
"""
from __future__ import annotations

from typing import List

import json

try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    LuceneSearcher = None  # type: ignore


# ---------------------------------------------------------------------------
# Oracle / forced exposure helpers
# ---------------------------------------------------------------------------

def oracle_gold_docs(
    searcher,
    golds: List[str],
    k: int = 20,
    max_docs: int = 2,
) -> List[str]:
    """Return up to `max_docs` passages from `searcher` that contain a gold answer.

    Each returned passage is a plain text string (contents field from the DPR
    corpus document).  Deduplication is done on the first 200 chars.

    Args:
        searcher: pyserini LuceneSearcher (or compatible interface with .search() / .doc())
        golds:    list of gold answer strings for the question
        k:        BM25 hits to request per gold answer
        max_docs: maximum passages to return
    """
    found: List[str] = []
    seen: set = set()
    for gold in golds:
        hits = searcher.search(gold, k=k)
        for h in hits:
            try:
                raw = json.loads(searcher.doc(h.docid).raw())
                text = raw.get("contents", "")
                if not text or gold.lower() not in text.lower():
                    continue
                key = text[:200]
                if key in seen:
                    continue
                seen.add(key)
                found.append(text)
                if len(found) >= max_docs:
                    return found
            except Exception:
                continue
    return found


def build_forced_exposure_context(
    gold_docs: List[str],
    poison_docs: List[str],
    total_slots: int = 4,
    gold_positions: List[int] | None = None,
    per_doc_chars: int = 900,
) -> str:
    """Build a fixed-position context string for the forced-exposure track.

    Gold documents are placed at `gold_positions` (default: slot 0).
    Remaining slots are filled from `poison_docs` in order.

    Args:
        gold_docs:      passages returned by oracle_gold_docs()
        poison_docs:    frozen sybil poison docs for the question
        total_slots:    total number of context slots (= n_passages for this track)
        gold_positions: zero-indexed positions where gold docs are injected
        per_doc_chars:  truncate each passage to this many characters

    Returns:
        Newline-separated context string ready for the LLM prompt.
    """
    if gold_positions is None:
        gold_positions = [0]
    slots = [""] * total_slots
    for idx, pos in enumerate(gold_positions):
        if idx < len(gold_docs) and 0 <= pos < total_slots:
            slots[pos] = gold_docs[idx][:per_doc_chars]

    poison_iter = iter(poison_docs)
    for idx in range(total_slots):
        if slots[idx]:
            continue
        try:
            slots[idx] = next(poison_iter)[:per_doc_chars]
        except StopIteration:
            break
    return "\n\n".join([s for s in slots if s.strip()])


def forced_gold_positions_for_dataset(ds: str, total_k: int) -> List[int]:
    """Return canonical gold slot positions for a given dataset.

    - hotpot:  [0, 2]  (two-hop — gold placed at positions 0 and 2)
    - nq:      [0]
    - wiki2:   [0]

    Only returns positions that are valid (< total_k).
    """
    preferred = [0, 2] if ds == "hotpot" else [0]
    return [pos for pos in preferred if 0 <= pos < total_k]
