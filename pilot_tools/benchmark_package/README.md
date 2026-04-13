# Grounded QA Poisoning Benchmark Package

This directory contains the reproducibility assets for the benchmark:

- `fixed_splits/`: ordered question manifests
- `snapshot_metadata.json`: corpus / index / poison asset metadata
- `dataset_card.md`: benchmark motivation, threat model, and limitations
- `baseline_suite.md`: baseline definitions and run commands
- `evaluator_spec.md`: official CSV schema, canonicalization, and metric definitions
- `reporting_template.md`: recommended reporting structure for main tables and appendix
- `leaderboard.md`: standardized prompt-track leaderboard structure and provenance fields
- `chunk_retrieval.md`: passage chunk retrieval protocol
- `paper_assets.md`: paper positioning, title candidates, abstract skeleton, and table plan

## Benchmark framing

This benchmark is not positioned as a pure target-hijack benchmark.
It is positioned as a benchmark for grounded answer destabilization under
polymorphic sybil retrieval poisoning.

Core failure modes:

- gold retention
- targeted hijack
- third-answer drift
- abstain

Official report metrics:

- ACC
- ASR
- CACC
- Abstain Rate
- Third-Answer Drift Rate

## Reproducibility workflow

1. Create or reuse a fixed question manifest.
2. Run clean / attack / defense baselines on the exact same manifest.
3. Score all outputs with `official_eval.py`.
4. Report combined, Hotpot, and NQ metrics.

For official benchmark release:

- split generation should use stable-hash ordering
- `dataset=all` public splits should be balanced across Hotpot and NQ
- public/dev/test should be generated as disjoint manifests
- manifest consumers should fail fast on missing `(ds,id)` entries

## Benchmark tracks

### `clean_strong`

Purpose:

- estimate the clean QA ceiling
- allow dataset-aware reader design
- separate clean QA quality from purification trade-offs

Reference runner:

- `./run_base_only.sh`

### `attack_defense_fairness`

Purpose:

- compare attacked and defended systems under a shared Base/Attack reader stack
- isolate purification effects without conflating them with a stronger clean-only reader

Reference runner:

- `./run_v24.sh`

### `oracle_control`

Purpose:

- separate retrieval misses from answer-layer collapse
- expose parametric-only and oracle evidence controls on the same official split

Reference runner:

- `python3 fullwiki_eval_oracle_controls.py --track ...`

## Core scripts

- Create manifest: `python3 make_fixed_question_manifest.py ...`
- Clean base: `./run_base_only.sh`
- Hotpot clean path ablation: `./run_base_exp1_hotpot_clean.sh`
- NQ clean path ablation: `./run_base_exp2_nq_clean.sh`
- Fairness track: `./run_v24.sh`
- Oracle/control track: `python3 fullwiki_eval_oracle_controls.py --track ...`
- Official evaluator: `./run_official_eval.sh output.csv`
