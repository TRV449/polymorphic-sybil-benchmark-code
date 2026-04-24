# Polymorphic Sybil Retrieval Poisoning Benchmark

Code and manifests for the Polymorphic Sybil Retrieval Poisoning
Benchmark (NeurIPS 2026 D&B submission, under review).

## 📊 Dataset

3,145-question manifest with 2,982 retained polymorphic sybil groups
(S=6, τ_lex=0.8), plus cross-dataset validation on TriviaQA and
2WikiMultiHopQA. See [`data/README.md`](data/README.md) for schema.

## 📏 Evaluation

Four-way partition (gold / hijack / abstention / drift) with paired
clean-to-poison transitions. See [`evaluator/`](evaluator/) for the
official strict evaluator and sensitivity variant.

## ✍️ How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Download retrieval indexes from companion HF dataset:
   `huggingface.co/datasets/anon-neurips-ed-2026/polymorphic-sybil-benchmark-data`
3. Run: `bash scripts/reproduce_main_grid.sh`

See [`docs/reproduction.md`](docs/reproduction.md) for full setup.

## 🔒 Prompt integrity

Prompt SHA-256 (also reported in the paper):
- Generator: `37eb61bc...`
- Verifier: `0a4a02c9...`
- Common reader prompt: `a6f2d784...`

## 📜 Citation

```bibtex
@inproceedings{anon2026polymorphicsybil,
  title={Polymorphic Sybil Retrieval Poisoning Benchmark:
         A Failure-Mode-Aware Evaluation Framework for Grounded QA},
  author={Anonymous},
  booktitle={NeurIPS 2026 D&B Track (under review)},
  year={2026}
}
```

## 📄 License

Code: MIT. Data (manifests): CC BY-SA 4.0 (inherited from NQ,
HotpotQA). Cross-dataset samples (TriviaQA, 2WikiMultiHopQA):
Apache 2.0 (inherited upstream).
