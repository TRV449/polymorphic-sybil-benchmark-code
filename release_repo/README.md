# Polymorphic Sybil Retrieval Poisoning Benchmark

Anonymous code + manifest release for NeurIPS 2026 D&B submission.

## Quick Start (5 minutes)

```bash
pip install -r requirements.txt
# Verify evaluator
python3 -c "from evaluator.benchmark_eval_utils import classify_answer; print('OK')"
# Inspect main manifest (first record)
head -1 manifests/poison_artifacts_main.jsonl | python3 -m json.tool
# Verify file integrity
sha256sum -c SHA256SUMS
```

## Repository Structure

```
manifests/          Question manifests + frozen poison artifacts
evaluator/          Official strict-EM evaluator (4-way partition)
generator/          Sybil generation pipeline + prompt
verifier/           QC verifier config + prompt
readers/            Reader configs + runner scripts
analysis/           Bootstrap CI, Bonferroni correction
results/            Pre-computed results (CSV + JSON)
  main_grid/        87-cell main results with 95% CI
  cross_dataset/    TriviaQA + 2Wiki validation
  ablation_500q/    Mono vs poly ablation (Table 2)
  drift_audit/      1,320-sample drift origin audit
  defense/          Defense evasion analysis (4 methods)
audit/              Drift audit classifier + human validation tools
```

## Full Reproduction

Requires retrieval indexes (~92 GB total):
- BM25 Lucene: build via `retrieval/build_bm25_index.py`
- E5 FAISS: build via `retrieval/build_e5_faiss.py`
- ColBERTv2: downloaded at runtime from HuggingFace

Hardware: 2× NVIDIA A100 80GB, ≥256 GB RAM.
Approximate compute: ~200 GPU-hours for full grid.

## Key Files

| File | Description |
|---|---|
| `manifests/poison_artifacts_main.jsonl` | 2,982 retained sybil groups (main release) |
| `manifests/poison_mono_500q.jsonl` | 500Q monomorphic baseline (ablation) |
| `evaluator/official_qc.py` | Official QC evaluator with 4-way partition |
| `generator/prompt.txt` | Generator prompt (SHA-256: `37eb61bc...`) |
| `verifier/prompt.txt` | Verifier prompt (SHA-256: `0a4a02c9...`) |
| `readers/common_qa_system_prompt.txt` | Shared reader prompt (SHA-256: `a6f2d784...`) |
| `results/main_grid/final_results_with_ci.csv` | 87 cells × 4 metrics with bootstrap CI |

## Integrity

```bash
sha256sum -c SHA256SUMS
```

Prompt-level SHA-256: generator `37eb61bc`, verifier `0a4a02c9`, reader `a6f2d784`.

## License

- Code: MIT (`LICENSE`)
- Data: CC BY-SA 4.0 (`LICENSE-DATA`)

## Citation

```bibtex
@inproceedings{anon2026polymorphicsybil,
  title={Polymorphic Sybil Retrieval Poisoning Benchmark:
         A Failure-Mode-Aware Evaluation Framework for Grounded QA},
  author={Anonymous},
  booktitle={NeurIPS 2026 Datasets and Benchmarks Track (under review)},
  year={2026}
}
```
