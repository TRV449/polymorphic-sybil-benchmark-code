# Polymorphic Sybil Retrieval Poisoning Benchmark

Code and manifests for the **Polymorphic Sybil Retrieval Poisoning Benchmark** — a failure-mode-aware evaluation framework for grounded QA under coordinated retrieval poisoning.

Under review at **NeurIPS 2026 Datasets & Benchmarks Track**.

---

## 📊 Dataset

A frozen benchmark of 3,145 host questions with **2,982 retained polymorphic sybil groups** (S=6 passages per group, lexical diversity threshold τ_lex = 0.8), built on Natural Questions (NQ-open validation) and HotpotQA (distractor dev). Two additional validation layers are provided on TriviaQA and 2WikiMultiHopQA, plus a 500-question monomorphic baseline for ablation.

See [`data/README.md`](data/README.md) for manifest schema and file-level details.

| Layer | Source | Retained | Purpose |
|---|---|---:|---|
| Main | NQ + HotpotQA | 2,982 | Primary evaluation grid |
| Cross-dataset | TriviaQA | 1,398 | Factoid generalization |
| Cross-dataset | 2WikiMultiHopQA | 2,696 | Multi-hop generalization |
| Ablation | NQ + HotpotQA (500Q) | 500 | Monomorphic-vs-polymorphic contrast |

## 📏 Evaluation

Unlike conventional ASR-only reporting, this benchmark partitions reader outputs into **four mutually exclusive categories**: gold retention / targeted hijack / abstention / third-answer drift. Three evaluation conditions are supported:

- **clean** — organic retrieval, no injection
- **attack** — sybil group prepended to candidate pool before reranking
- **Forced Exposure** — sybil placements fixed at protocol-defined top-10 positions alongside gold-supporting passages

The official strict evaluator and a lenient alias-substring variant (sensitivity audit) are provided. See [`evaluator/`](evaluator/) for details.

## ✍️ How to run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download retrieval indexes** from the companion Hugging Face dataset (~114 GB total):
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id="anon-neurips-ed-2026/polymorphic-sybil-benchmark-data",
       repo_type="dataset",
   )
   ```

3. **Reproduce the main grid**
   ```bash
   bash scripts/reproduce_main_grid.sh
   ```

See [`docs/reproduction.md`](docs/reproduction.md) for full setup (model weights, compute requirements, reader configurations).

## 🔒 Prompt integrity

Byte-identical prompts across the pipeline are pinned by SHA-256:

| Component | SHA-256 (prefix) |
|---|---|
| Generator prompt (Llama-3.1-8B) | `37eb61bc...` |
| Verifier prompt (Qwen2.5-72B) | `0a4a02c9...` |
| Common reader system prompt | `a6f2d784...` |

Full hashes are in [`prompts/`](prompts/).

## 🏁 Reference results

Reference outputs across **5 readers** (7B–120B parameters) × **2 retrieval substrates** (BM25→ColBERTv2, E5+CE) × **3 conditions** are included under `data/results/` for verification. See the paper (§7) for the full main grid.

## 📜 Citation

```bibtex
@inproceedings{anon2026polymorphic,
  title     = {Polymorphic Sybil Retrieval Poisoning Benchmark:
               A Failure-Mode-Aware Evaluation Framework for Grounded QA},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (Datasets and Benchmarks Track)},
  year      = {2026},
  note      = {Under review}
}
```

## 📄 License

- **Code**: MIT (see [`LICENSE`](LICENSE))
- **Manifests (main, ablation)**: CC BY-SA 4.0, inherited from NQ and HotpotQA share-alike
- **Cross-dataset samples (TriviaQA, 2WikiMultiHopQA)**: Apache 2.0, inherited from upstream

Users redistributing any subset must comply with the applicable upstream license in addition.
