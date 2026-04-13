from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_manifest_entry(item: dict) -> dict:
    return {
        "ds": str(item.get("ds", "")).strip(),
        "id": str(item.get("id", "")).strip(),
    }


def compute_selection_sha256(entries: List[dict]) -> str:
    normalized = [normalize_manifest_entry(item) for item in entries]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return data


def extract_manifest_entries(manifest: dict) -> List[dict]:
    entries = manifest.get("questions", [])
    if not isinstance(entries, list):
        raise ValueError("Manifest `questions` must be a list")

    normalized = []
    seen = set()
    duplicates = []
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("Manifest questions must contain objects with `ds` and `id`")
        item = normalize_manifest_entry(raw)
        if not item["ds"] or not item["id"]:
            raise ValueError("Manifest question entries must contain non-empty `ds` and `id`")
        key = (item["ds"], item["id"])
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
        normalized.append(item)

    if duplicates:
        preview = ", ".join(f"{ds}:{qid}" for ds, qid in duplicates[:5])
        raise ValueError(f"Manifest contains duplicate (ds,id) entries: {preview}")

    expected_count = manifest.get("question_count")
    if expected_count is not None and int(expected_count) != len(normalized):
        raise ValueError(
            f"Manifest question_count mismatch: expected {expected_count}, found {len(normalized)} entries"
        )

    expected_sha = str(manifest.get("selection_sha256", "")).strip()
    if expected_sha:
        actual_sha = compute_selection_sha256(normalized)
        if actual_sha != expected_sha:
            raise ValueError(
                "Manifest selection_sha256 mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )

    expected_by_dataset = manifest.get("questions_by_dataset")
    if expected_by_dataset is not None:
        if not isinstance(expected_by_dataset, dict):
            raise ValueError("Manifest `questions_by_dataset` must be an object when provided")
        actual_by_dataset: Dict[str, int] = {}
        for item in normalized:
            actual_by_dataset[item["ds"]] = actual_by_dataset.get(item["ds"], 0) + 1
        normalized_expected = {str(k).strip(): int(v) for k, v in expected_by_dataset.items()}
        if actual_by_dataset != normalized_expected:
            raise ValueError(
                "Manifest questions_by_dataset mismatch: "
                f"expected {normalized_expected}, got {actual_by_dataset}"
            )

    return normalized


def count_questions_by_dataset(questions: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in questions:
        ds = str(item.get("ds", "")).strip()
        counts[ds] = counts.get(ds, 0) + 1
    return counts


def apply_manifest(questions: List[dict], manifest: dict, source_label: str = "questions") -> List[dict]:
    entries = extract_manifest_entries(manifest)
    lookup: Dict[Tuple[str, str], dict] = {
        (str(q["ds"]).strip(), str(q["id"]).strip()): q for q in questions
    }
    selected = []
    missing = []
    for item in entries:
        key = (item["ds"], item["id"])
        if key not in lookup:
            missing.append(key)
            continue
        selected.append(lookup[key])

    if missing:
        preview = ", ".join(f"{ds}:{qid}" for ds, qid in missing[:10])
        raise ValueError(
            "Manifest references questions that are missing from the current source "
            f"{source_label}: {preview}"
        )

    if len(selected) != len(entries):
        raise ValueError(
            f"Manifest application mismatch: expected {len(entries)} questions, got {len(selected)}"
        )

    return selected


def validate_selected_questions(selected: List[dict], manifest: dict, source_label: str = "questions") -> None:
    entries = extract_manifest_entries(manifest)
    selected_entries = [
        {"ds": str(item.get("ds", "")).strip(), "id": str(item.get("id", "")).strip()}
        for item in selected
    ]

    if len(selected_entries) != len(entries):
        raise ValueError(
            f"Manifest-selected question count mismatch for {source_label}: "
            f"expected {len(entries)}, got {len(selected_entries)}"
        )

    selected_sha = compute_selection_sha256(selected_entries)
    expected_sha = str(manifest.get("selection_sha256", "")).strip()
    if expected_sha and selected_sha != expected_sha:
        raise ValueError(
            f"Manifest-selected selection_sha256 mismatch for {source_label}: "
            f"expected {expected_sha}, got {selected_sha}"
        )

    expected_by_dataset = manifest.get("questions_by_dataset")
    if expected_by_dataset is not None:
        normalized_expected = {str(k).strip(): int(v) for k, v in expected_by_dataset.items()}
        actual_by_dataset = count_questions_by_dataset(selected)
        if actual_by_dataset != normalized_expected:
            raise ValueError(
                f"Manifest-selected questions_by_dataset mismatch for {source_label}: "
                f"expected {normalized_expected}, got {actual_by_dataset}"
            )
