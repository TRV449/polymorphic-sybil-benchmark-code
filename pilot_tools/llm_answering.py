import re
import os
from typing import Iterable

from inference_backends import LLMBackendConfig, complete_text


COMMON_QA_SYSTEM_PROMPT = """You are a factual QA assistant.
Answer the question only from the provided context.
If the answer is not supported by the context, reply with "Unknown".
Return only a short answer without explanation."""

UNKNOWN_PATTERNS = frozenset({
    "unknown", "error", "i don t know", "do not know", "not sure",
    "i don t", "dont know", "no idea", "cannot answer", "unclear",
    "can t answer", "can not answer", "not enough information", "not mentioned",
})
TRAILING_UNKNOWN_RE = re.compile(r"(?:^|[\s\]\)\}\.,;:!?-])unknown(?:[\s\]\)\}\.,;:!?-]*)$", re.IGNORECASE)


def _normalize_unknown_candidate(text: str) -> str:
    text = (text or "").strip()
    text = text.strip("\"'`")
    text = re.sub(r"^[\s\.\,\!\?\:\;\-\(\)\[\]\{\}]+", "", text)
    text = re.sub(r"[\s\.\,\!\?\:\;\-\(\)\[\]\{\}]+$", "", text)
    return " ".join(text.split()).lower()


def _canonicalize_if_unknown(text: str) -> str:
    norm = _normalize_unknown_candidate(text)
    low = str(text or "").strip().lower()
    if norm in UNKNOWN_PATTERNS or norm.startswith("error") or TRAILING_UNKNOWN_RE.search(low):
        return "Unknown"
    return text


def is_unknown_answer(text: str) -> bool:
    return _canonicalize_if_unknown(text) == "Unknown"


def _normalize_base_url(url: str) -> str:
    url = str(url or "").strip().rstrip("/")
    if url.endswith("/completion"):
        url = url.rsplit("/completion", 1)[0]
    return url


_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)",
    re.DOTALL,
)
_HARMONY_CLEANUP_RE = re.compile(r"<\|[^|]*\|>[^<]*(?=<\||$)")


_HARMONY_ANALYSIS_BLOCK_RE = re.compile(
    r"<\|channel\|>\s*analysis\s*<\|message\|>.*?(?:<\|end\|>|$)",
    re.DOTALL,
)


def _extract_harmony_final(raw: str) -> str:
    """Extract final answer from Harmony response format (reasoning models)."""
    m = _HARMONY_FINAL_RE.search(raw)
    if m:
        return m.group(1).strip()
    cleaned = _HARMONY_ANALYSIS_BLOCK_RE.sub("", raw)
    cleaned = re.sub(r"<\|[^|]*\|>", " ", cleaned)
    tokens = [t for t in cleaned.split() if t.lower() not in {"assistant", "final", "analysis", "start", "end", "channel", "message"}]
    return " ".join(tokens).strip()


def extract_first_answer(raw: str) -> str:
    """
    평가 가능한 형태로 답변 추출. Base/Attack/Defense 공통 postprocess.
    - 첫 줄까지만
    - 첫 문장까지만 (마침표/콜론/세미콜론 기준)
    - 괄호 안 설명 제거
    - Unknown류는 Unknown으로 통일
    """
    raw = (raw or "").strip()
    if not raw:
        return "Unknown"
    if raw.startswith("[ERROR"):
        return "Unknown"
    if "<|channel|>" in raw or "<|message|>" in raw:
        raw = _extract_harmony_final(raw)
        if not raw:
            return "Unknown"
    first_line = raw.split("\n")[0].strip()
    if not first_line:
        return "Unknown"
    # 괄호 안 내용 제거
    first_line = re.sub(r"\s*\([^)]*\)\s*", " ", first_line)
    first_line = re.sub(r"\s*\[[^\]]*\]\s*", " ", first_line)
    first_line = " ".join(first_line.split()).strip()
    # 콜론/세미콜론/대시 뒤 설명 제거 (첫 답만 취함)
    for sep in [";", ":", " – ", " - ", " — "]:
        if sep in first_line:
            first_line = first_line.split(sep)[0].strip()
    # 마침표 기준 첫 문장
    parts = [p.strip() for p in first_line.split(". ") if p.strip()]
    for p in parts:
        p_clean = p.strip()
        if not p_clean:
            continue
        canonical = _canonicalize_if_unknown(p_clean)
        if canonical == "Unknown":
            return "Unknown"
        if len(p_clean) <= 100:
            return canonical
    if parts:
        out = parts[0]
        out = out[:100].rsplit(" ", 1)[0] if len(out) > 100 else out
        canonical = _canonicalize_if_unknown(out)
        return canonical or "Unknown"
    out = first_line[:100].rsplit(" ", 1)[0] if len(first_line) > 100 else first_line
    canonical = _canonicalize_if_unknown(out)
    if canonical == "Unknown":
        return "Unknown"
    return canonical or "Unknown"


def build_qa_prompt(context: str, question: str) -> str:
    context = (context or "").strip()
    question = (question or "").strip()
    return f"""{COMMON_QA_SYSTEM_PROMPT}

Context:
{context}

Question: {question}
Answer:"""


def resolve_backend_config(
    url: str = "http://127.0.0.1:8000/completion",
    n_predict: int = 64,
    temperature: float = 0.0,
    backend: str = None,
    model_id: str = None,
) -> LLMBackendConfig:
    backend_name = backend or os.environ.get("LLM_BACKEND", "llama_cpp_http")
    base_url = _normalize_base_url(os.environ.get("LLM_BASE_URL", ""))
    if not base_url:
        base_url = _normalize_base_url(url)
    return LLMBackendConfig(
        backend=backend_name,
        base_url=base_url,
        model_id=model_id or os.environ.get("LLM_MODEL_ID", ""),
        api_key_env=os.environ.get("LLM_API_KEY_ENV", "OPENAI_API_KEY"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", str(temperature))),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", str(n_predict))),
        timeout=int(os.environ.get("LLM_TIMEOUT", "90")),
    )


def prompt_sha256(prompt: str) -> str:
    import hashlib
    return hashlib.sha256(str(prompt or "").encode("utf-8")).hexdigest()


CONSENSUS_QA_PROMPT = """You are given a small evidence set.
Answer the question only if the answer is directly supported by the evidence.
Otherwise reply with "Unknown".
Return only a short answer, nothing else."""

# Base path용: 단답형 강화 (entity/date/number/short phrase, 최대 8단어, 설명 금지)
BASE_READER_PROMPT = """You are given evidence. Answer the question only if directly supported.
If not supported, reply "Unknown".
Return only a short answer without explanation."""


def get_llama_answer_from_passage(
    question: str,
    passage: str,
    url: str = "http://127.0.0.1:8000/completion",
    n_predict: int = 32,
    prompt_template: str = None,
    backend: str = None,
    model_id: str = None,
    temperature: float = 0.0,
) -> str:
    """짧은 evidence passage에서 답만 추출 (v24 consensus용 support unit별 답 추출)."""
    passage = (passage or "").strip()
    question = (question or "").strip()
    tmpl = prompt_template or CONSENSUS_QA_PROMPT
    prompt = f"""{tmpl}

Evidence:
{passage[:1024]}

Question: {question}
Answer:"""
    try:
        config = resolve_backend_config(
            url=url,
            n_predict=n_predict,
            temperature=temperature,
            backend=backend,
            model_id=model_id,
        )
        out = complete_text(prompt, config)
        return out or "Unknown"
    except Exception:
        return "Unknown"


def get_llama_answer_common(
    context: str,
    question: str,
    url: str = "http://127.0.0.1:8000/completion",
    n_predict: int = 64,
    backend: str = None,
    model_id: str = None,
    temperature: float = 0.0,
) -> str:
    try:
        config = resolve_backend_config(
            url=url,
            n_predict=n_predict,
            temperature=temperature,
            backend=backend,
            model_id=model_id,
        )
        out = complete_text(build_qa_prompt(context, question), config)
        return out or "[ERROR: Empty response body]"
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def is_anchor_confident(anchor_text: str) -> bool:
    """Anchor 답변이 확신 있는 답변인지 (Fallback 발동 허용 여부)"""
    if not anchor_text or not anchor_text.strip():
        return False
    lowered = anchor_text.lower().strip()
    uncertain_tokens = [
        "unknown", "not sure", "cannot", "can't", "no idea", "unclear",
        "i don't know", "i do not know", "unsure", "maybe", "perhaps",
        "possibly", "might", "could be", "not certain", "no information",
        "as an ai", "not mentioned", "i cannot answer", "cannot answer",
    ]
    if any(t in lowered for t in uncertain_tokens):
        return False
    if len(anchor_text.split()) > 12:
        return False  # 과도하게 긴 답변은 회피/설명일 가능성
    return True


def get_anchor_answer(
    question: str,
    url: str = "http://127.0.0.1:8000/completion",
    n_predict: int = 32,
    backend: str = None,
    model_id: str = None,
    temperature: float = 0.0,
) -> str:
    anchor_prompt = f"""System: You are answering from parametric knowledge only. If you are not confident, respond with "Unknown". Do not guess.

Question: {question}
Answer:"""
    try:
        config = resolve_backend_config(
            url=url,
            n_predict=n_predict,
            temperature=temperature,
            backend=backend,
            model_id=model_id,
        )
        anchor_text = complete_text(anchor_prompt, config)
        anchor_text = anchor_text.split("\n")[0].strip()
        lowered = anchor_text.lower()
        if any(token in lowered for token in ["unknown", "not sure", "cannot", "can't", "no idea"]):
            return ""
        if len(anchor_text.split()) > 8:
            return ""
        return anchor_text
    except Exception:
        return ""


def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9]", " ", s)
    return " ".join(s.split())


def check_em(prediction: str, ground_truths: Iterable[str]) -> bool:
    """Exact Match: 정규화 후 완전 일치만 True. substring match 아님."""
    pred = normalize_answer(prediction)
    for gt in ground_truths:
        if pred == normalize_answer(gt):
            return True
    return False
