# Polymorphic Sybil Benchmark — Code

## Canonical Modules: `benchmark_core/`

All core evaluation modules live in `benchmark_core/`. Root-level copies
(`benchmark_eval_utils.py`, `official_eval.py`, etc.) are kept in sync for
backward-compatible imports used by runners and analysis scripts.

**For paper references, always cite `benchmark_core/` paths.**

## Directory Layout

```
pilot_tools/
├── benchmark_core/           # Paper §4 Official Evaluator (canonical)
│   ├── benchmark_eval_utils  — classify_answer(), normalize_answer(), split_answers()
│   ├── official_eval         — print_evaluation_report(), CSV schema validation
│   ├── poison_qc            — Paper §3.3 acceptance rules (3-gate QC)
│   ├── manifest_utils       — Manifest loading, integrity validation
│   ├── transition_matrix    — Paper §7 paired transition rates
│   ├── merge_and_transition — Multi-track CSV merge
│   └── run_metadata         — Experiment metadata tracking
│
├── baselines/                # Reader/Retriever runners
│   ├── modern_rag/
│   │   ├── colbert_qwen_runner  — BM25 + ColBERT MaxSim pipeline
│   │   ├── faiss_qwen_runner    — E5 + Cross-Encoder pipeline
│   │   └── atlas_runner         — Atlas baseline
│   └── fid/
│       └── fid_runner           — Fusion-in-Decoder baseline
│
├── reference_systems/        # Appendix G diagnostic reference
│   └── custom/
│       ├── fairness_v25_reference  — Trigger-free defense (v25)
│       └── base_only_reference     — No-defense baseline
│
├── stress_protocols/         # Oracle controls
│   ├── oracle_controls      — Forced-exposure oracle
│   └── stress_protocols     — Answer-match pseudo-oracle
│
├── benchmark_package/        # Release specification
│   ├── fixed_splits/        — Train/dev/test JSON splits
│   ├── dataset_card.md      — Croissant-compatible dataset card
│   └── evaluator_spec.md    — Evaluator usage guide
│
├── inference_backends.py     # LLM server abstraction (llama.cpp, OpenAI)
├── llm_answering.py          # Prompt template + reader entry point
├── model_registry.py         # Model ID → config mapping
│
├── day4_main_runs.sh         # Day 4 experiment launcher (all readers)
├── day4_auto_analysis.py     # Post-experiment analysis pipeline
├── day4_qc_finalize.sh       # QC → final manifest generator
└── day4_run_all_analysis.sh  # Batch analysis for all completed runs
```

## Root-level Duplicates

These files are **identical** to their `benchmark_core/` counterparts:

| Root file                  | Canonical location                    |
|----------------------------|---------------------------------------|
| `benchmark_eval_utils.py`  | `benchmark_core/benchmark_eval_utils` |
| `official_eval.py`         | `benchmark_core/official_eval`        |
| `poison_qc.py`             | `benchmark_core/poison_qc`            |
| `manifest_utils.py`        | `benchmark_core/manifest_utils`       |
| `transition_matrix.py`     | `benchmark_core/transition_matrix`    |
| `run_metadata.py`          | `benchmark_core/run_metadata`         |

Root copies exist because runners use `sys.path.insert(0, pilot_tools_dir)`
and import directly (e.g. `from benchmark_eval_utils import classify_answer`).
