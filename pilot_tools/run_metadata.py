from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

from llm_answering import prompt_sha256


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_content_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    if path.is_file():
        return file_sha256(path)

    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path)
        h.update(str(rel).encode("utf-8"))
        h.update(file_sha256(item).encode("utf-8"))
    return h.hexdigest()


def get_git_commit(repo_root: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def load_manifest_metadata(manifest: dict) -> Dict[str, str]:
    source_hotpot = manifest.get("source_datasets", {}).get("hotpot", {}) or {}
    source_nq = manifest.get("source_datasets", {}).get("nq", {}) or {}
    return {
        "manifest_name": str(manifest.get("name", "")).strip(),
        "manifest_selection_sha256": str(manifest.get("selection_sha256", "")).strip(),
        "source_hotpot_sha256": str(source_hotpot.get("sha256", "")).strip(),
        "source_nq_sha256": str(source_nq.get("sha256", "")).strip(),
    }


def build_run_metadata(
    *,
    workspace_root: str,
    manifest: dict | None,
    base_index: str,
    corpus_path: str = "",
    attack_index: str = "",
    faiss_index_dir: str = "",
    poison_jsonl: str = "",
    prompt_text: str = "",
    llm_backend: str = "",
    llm_model_id: str = "",
    llm_base_url: str = "",
    llm_temperature: float = 0.0,
    n_predict: int = 0,
    extra_fields: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    data = {
        "corpus_hash": directory_content_sha256(corpus_path) if corpus_path else "",
        "index_hash": directory_content_sha256(base_index),
        "faiss_index_hash": directory_content_sha256(faiss_index_dir) if faiss_index_dir else "",
        "attack_index_hash": directory_content_sha256(attack_index) if attack_index else "",
        "poison_jsonl_sha256": file_sha256(poison_jsonl) if poison_jsonl else "",
        "poison_generator_version": os.environ.get("POISON_GENERATOR_VERSION", ""),
        "poison_generator_prompt_hash": os.environ.get("POISON_GENERATOR_PROMPT_HASH", ""),
        "git_commit": get_git_commit(workspace_root),
        "prompt_hash": prompt_sha256(prompt_text),
        "generator_model_id": llm_model_id,
        "generator_model_family": llm_model_id.split("/", 1)[0] if "/" in llm_model_id else llm_model_id,
        "generator_model_provider": "huggingface" if "/" in llm_model_id else ("local" if llm_model_id else ""),
        "generator_model_date": os.environ.get("LLM_MODEL_DATE", ""),
        "llm_backend": llm_backend,
        "llm_base_url": llm_base_url,
        "llm_temperature": str(llm_temperature),
        "n_predict": str(n_predict),
        "verifier_model_id": os.environ.get("VERIFIER_MODEL_ID", ""),
        "verifier_backend": os.environ.get("VERIFIER_BACKEND", ""),
        "verifier_revision": os.environ.get("VERIFIER_REVISION", ""),
        "verifier_temperature": os.environ.get("VERIFIER_TEMPERATURE", ""),
        "verifier_max_tokens": os.environ.get("VERIFIER_MAX_TOKENS", ""),
        "verifier_prompt_hash": os.environ.get("VERIFIER_PROMPT_HASH", ""),
        "verifier_decision_rule": os.environ.get("VERIFIER_DECISION_RULE", ""),
    }
    if manifest:
        data.update(load_manifest_metadata(manifest))
    else:
        data.update(
            {
                "manifest_name": "",
                "manifest_selection_sha256": "",
                "source_hotpot_sha256": "",
                "source_nq_sha256": "",
            }
        )
    if extra_fields:
        for key, value in extra_fields.items():
            data[str(key)] = "" if value is None else str(value)
    return data


def write_run_metadata(output_csv: str, metadata: Dict[str, str]) -> str:
    path = Path(output_csv)
    meta_path = path.with_suffix(path.suffix + ".run_meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return str(meta_path)
