from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


MODELS_ROOT = Path(os.environ.get("BENCHMARK_MODELS_ROOT", "./member_runtime/models"))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    family: str
    scale: str
    provider: str
    backend: str
    quantization: str
    revision: str = "main"
    local_subdir: str = ""
    filename: str = ""
    notes: str = ""

    @property
    def local_path(self) -> Path:
        return MODELS_ROOT / (self.local_subdir or self.key)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["local_path"] = str(self.local_path)
        return data


MODEL_SPECS: Dict[str, ModelSpec] = {
    "llama31_8b_gguf": ModelSpec(
        key="llama31_8b_gguf",
        model_id="local/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        family="llama",
        scale="small",
        provider="local",
        backend="llama_cpp_http",
        quantization="Q8_0",
        local_subdir="llama31_8b_gguf",
        notes="Existing llama.cpp GGUF baseline.",
    ),
    "qwen25_7b_instruct_gguf": ModelSpec(
        key="qwen25_7b_instruct_gguf",
        model_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
        family="qwen",
        scale="small",
        provider="huggingface",
        backend="llama_cpp_http",
        quantization="Q4_K_M",
        local_subdir="Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    ),
    "qwen25_32b_instruct_gguf": ModelSpec(
        key="qwen25_32b_instruct_gguf",
        model_id="bartowski/Qwen2.5-32B-Instruct-GGUF",
        family="qwen",
        scale="medium",
        provider="huggingface",
        backend="llama_cpp_http",
        quantization="Q4_K_M",
        local_subdir="Qwen2.5-32B-Instruct-GGUF",
        filename="Qwen2.5-32B-Instruct-Q4_K_M.gguf",
    ),
    "qwen25_72b_instruct_gguf": ModelSpec(
        key="qwen25_72b_instruct_gguf",
        model_id="bartowski/Qwen2.5-72B-Instruct-GGUF",
        family="qwen",
        scale="large",
        provider="huggingface",
        backend="llama_cpp_http",
        quantization="Q4_K_M",
        local_subdir="Qwen2.5-72B-Instruct-GGUF",
        filename="Qwen2.5-72B-Instruct-Q4_K_M.gguf",
    ),
    "gpt_oss_120b_gguf": ModelSpec(
        key="gpt_oss_120b_gguf",
        model_id="gpt-oss-120b",
        family="gpt-oss",
        scale="xlarge",
        provider="local",
        backend="llama_cpp_chat",
        quantization="Q4_K_M",
        local_subdir="../gpt-oss-120b-GGUF/Q4_K_M",
        filename="gpt-oss-120b-Q4_K_M-00001-of-00002.gguf",
        notes="120B model, uses chat completions with channel tags.",
    ),
}


SCALE_DEFAULTS = {
    "small": "qwen25_7b_instruct_gguf",
    "medium": "qwen25_32b_instruct_gguf",
    "large": "qwen25_72b_instruct_gguf",
}


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_SPECS:
        raise KeyError(f"Unknown model registry key: {key}")
    return MODEL_SPECS[key]


def list_model_specs() -> List[ModelSpec]:
    return list(MODEL_SPECS.values())


def list_scale_defaults() -> List[ModelSpec]:
    return [get_model_spec(key) for key in SCALE_DEFAULTS.values()]
