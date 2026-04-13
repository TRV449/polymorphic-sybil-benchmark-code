#!/usr/bin/env python3
"""
Lenient answer matching — parallel evaluator for alias gap analysis.

Reports two accuracy variants:
  - Strict EM: normalize_answer(pred) == normalize_answer(gold)
  - Lenient:   + substring containment + token F1 >= 0.8

Does NOT modify benchmark_eval_utils.py (benchmark core freeze).
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple


def _normalize(text: str) -> str:
    s = str(text).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9]", " ", s)
    return " ".join(s.split())


def strict_em(pred: str, golds: Iterable[str]) -> bool:
    p = _normalize(pred)
    return any(p == _normalize(g) for g in golds)


def _token_f1(pred_norm: str, gold_norm: str) -> float:
    p_tokens = set(pred_norm.split())
    g_tokens = set(gold_norm.split())
    if not p_tokens or not g_tokens:
        return 0.0
    overlap = len(p_tokens & g_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def lenient_match(pred: str, golds: Iterable[str], f1_threshold: float = 0.8) -> Tuple[bool, str]:
    """Returns (matched, match_type) where match_type is 'strict'|'containment'|'token_f1'|None."""
    p = _normalize(pred)
    if not p:
        return False, ""

    for g in golds:
        gn = _normalize(g)
        if not gn:
            continue
        if p == gn:
            return True, "strict"
        if len(gn) >= 3 and gn in p:
            return True, "containment"
        if len(p) >= 3 and p in gn:
            return True, "containment"
        if _token_f1(p, gn) >= f1_threshold:
            return True, "token_f1"

    return False, ""


def classify_lenient(pred: str, golds: List[str], targets: List[str]) -> str:
    """4-way label using lenient matching: gold_strict, gold_lenient, target, drift."""
    if not pred or _normalize(pred) in {"unknown", ""}:
        return "abstain"
    if strict_em(pred, golds):
        return "gold_strict"
    matched, _ = lenient_match(pred, golds)
    if matched:
        return "gold_lenient"
    if targets and strict_em(pred, targets):
        return "target"
    if targets:
        t_matched, _ = lenient_match(pred, targets)
        if t_matched:
            return "target"
    return "drift"
