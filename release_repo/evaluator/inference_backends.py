from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import re

import requests

_HF_PIPELINE_CACHE = {}

_CHANNEL_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)",
    re.DOTALL,
)
_CHANNEL_ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>",
)


def _strip_channel_tags(text: str) -> str:
    """Extract final-channel answer from models that emit structured channel tags.

    If the final channel is present, extract its content.
    If only the analysis channel is present (truncated output), strip the tag prefix
    so downstream extract_first_answer sees cleaner text.
    """
    m = _CHANNEL_FINAL_RE.search(text)
    if m:
        return m.group(1).strip()
    if _CHANNEL_ANALYSIS_RE.match(text):
        return re.sub(r"<\|channel\|>analysis<\|message\|>", "", text).strip()
    return text


@dataclass
class LLMBackendConfig:
    backend: str = "llama_cpp_http"
    base_url: str = "http://127.0.0.1:8000"
    model_id: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 64
    timeout: int = 90


def _normalize_base_url(base_url: str) -> str:
    return str(base_url or "http://127.0.0.1:8000").rstrip("/")


def _extract_json_text(data) -> str:
    if data is None:
        return ""
    if isinstance(data, list) and data:
        data = data[0] if isinstance(data[0], dict) else {}
    if not isinstance(data, dict):
        return str(data).strip()
    if "error" in data:
        err = data["error"]
        if isinstance(err, dict):
            return str(err.get("message", err)).strip()
        return str(err).strip()
    if data.get("content") is not None:
        return str(data["content"]).strip()
    if data.get("text") is not None:
        return str(data["text"]).strip()
    if data.get("choices"):
        choice = data["choices"][0]
        if isinstance(choice, dict):
            if isinstance(choice.get("message"), dict) and choice["message"].get("content") is not None:
                return str(choice["message"]["content"]).strip()
            if choice.get("text") is not None:
                return str(choice["text"]).strip()
            if isinstance(choice.get("delta"), dict) and choice["delta"].get("content") is not None:
                return str(choice["delta"]["content"]).strip()
    if data.get("slots"):
        slot = data["slots"][0]
        if isinstance(slot, dict):
            if slot.get("content") is not None:
                return str(slot["content"]).strip()
            if slot.get("text") is not None:
                return str(slot["text"]).strip()
    return str(data).strip()


_OPENAI_SNAPSHOT_LOG = os.environ.get(
    "OPENAI_SNAPSHOT_LOG",
    "/mnt/data/2020112002/member_runtime/openai_model_snapshots.jsonl",
)
_OPENAI_SNAPSHOT_SEEN: set = set()

def _record_openai_snapshot(requested: str, observed: Optional[str]) -> None:
    """Append (requested → observed) once per process for OpenAI model resolution audit.
    Provides §8 Limitations evidence that alias gpt-4o-mini resolved to a specific snapshot."""
    if not observed or (requested, observed) in _OPENAI_SNAPSHOT_SEEN:
        return
    _OPENAI_SNAPSHOT_SEEN.add((requested, observed))
    try:
        import json, time
        with open(_OPENAI_SNAPSHOT_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "requested": requested,
                                 "observed": observed, "pid": os.getpid()}) + "\n")
    except Exception:
        pass


def _post_json(url: str, payload: dict, timeout: int, headers: Optional[dict] = None) -> str:
    response = requests.post(url, json=payload, timeout=timeout, headers=headers or {})
    response.raise_for_status()
    if not response.text or not response.text.strip():
        return ""
    return _extract_json_text(response.json())


def complete_text(prompt: str, config: LLMBackendConfig) -> str:
    backend = config.backend
    base_url = _normalize_base_url(config.base_url)
    prompt = str(prompt or "").strip()
    if not prompt:
        return ""

    if backend == "llama_cpp_http":
        raw = _post_json(
            f"{base_url}/completion",
            {
                "prompt": prompt,
                "n_predict": config.max_tokens,
                "temperature": config.temperature,
            },
            timeout=config.timeout,
        )
        return _strip_channel_tags(raw)

    if backend == "llama_cpp_chat":
        raw = _post_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": config.model_id or "default",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
            timeout=config.timeout,
        )
        return _strip_channel_tags(raw)

    if backend in {"openai_compatible", "vllm_openai", "frontier_api"}:
        model_id = config.model_id or "default-model"
        api_key = os.environ.get(config.api_key_env, "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return _post_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
            timeout=config.timeout,
            headers=headers,
        )

    if backend == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai>=1.0 required for openai backend")
        api_key = os.environ.get(config.api_key_env, "")
        client_kwargs = {"api_key": api_key} if api_key else {}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=config.model_id or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        _record_openai_snapshot(config.model_id or "gpt-4o-mini", getattr(resp, "model", None))
        return (resp.choices[0].message.content or "").strip()

    if backend == "hf_local":
        if not config.model_id:
            raise ValueError("hf_local backend requires model_id")
        if config.model_id not in _HF_PIPELINE_CACHE:
            import torch
            from transformers import pipeline

            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            _HF_PIPELINE_CACHE[config.model_id] = pipeline(
                "text-generation",
                model=config.model_id,
                tokenizer=config.model_id,
                device_map="auto",
                torch_dtype=torch_dtype,
            )
        generator = _HF_PIPELINE_CACHE[config.model_id]
        outputs = generator(
            prompt,
            do_sample=False,
            temperature=config.temperature,
            max_new_tokens=config.max_tokens,
            return_full_text=False,
        )
        if outputs and isinstance(outputs, list):
            first = outputs[0]
            if isinstance(first, dict):
                return str(first.get("generated_text", "")).strip()
        return ""

    raise ValueError(f"Unsupported backend: {backend}")


def healthcheck(config: LLMBackendConfig) -> bool:
    base_url = _normalize_base_url(config.base_url)
    try:
        if config.backend in {"llama_cpp_http", "llama_cpp_chat"}:
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            return r.status_code == 200
        if config.backend == "openai":
            try:
                from openai import OpenAI
                api_key = os.environ.get(config.api_key_env, "")
                client = OpenAI(api_key=api_key) if api_key else OpenAI()
                client.models.list()
                return True
            except Exception:
                return False
        if config.backend in {"openai_compatible", "vllm_openai", "frontier_api"}:
            headers = {}
            api_key = os.environ.get(config.api_key_env, "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            r = requests.get(f"{base_url}/v1/models", timeout=5, headers=headers)
            return r.status_code == 200
        if config.backend == "hf_local":
            return bool(config.model_id)
    except Exception:
        return False
    return False
