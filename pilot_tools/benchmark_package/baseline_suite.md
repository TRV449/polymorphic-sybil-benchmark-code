# Baseline Suite

## Required baselines

### 1. Clean-Strong

Purpose:

- estimate the clean retrieval + reader ceiling
- separate QA quality from defense trade-offs

Reference runner:

- `./run_base_only.sh`

Dataset-aware clean policy:

- NQ: FAISS + cross-encoder rerank + `doc[:1024]` vote reader
- Hotpot: FAISS + 2-hop + cross-encoder rerank + multi-doc single reader

Implementation notes:

- This track is allowed to use dataset-aware reader design.
- It should be treated as the benchmark clean ceiling, not as the fairness reference inside the defended pipeline.
- Suggested sweep axes: per-dataset `final_k`, per-dataset context prefix length, and Hotpot 2-hop on/off.
- New Hotpot clean option: `hotpot_reader_mode=pair_vote`

### 2. Attacked Baseline

Purpose:

- measure answer destabilization after retrieval hijack without purification

Reference runner:

- `./run_v24.sh` output, `attack_*` columns

Track role:

- `attack_defense_fairness`

### 3. Defended Baseline

Purpose:

- measure the security-coverage trade-off under purification

Reference runner:

- `./run_v24.sh` output, `v25_*` columns

Track role:

- `attack_defense_fairness`

### 4. Optional Parametric Baseline

Purpose:

- quantify how much answering comes from parametric memory alone

Suggested implementation:

- direct `get_llama_answer_common()` or a parametric QA prompt without retrieved context

Reference runner:

- `python3 fullwiki_eval_oracle_controls.py --track parametric_only ...`

### 5. BM25 + Same Reader

Purpose:

- isolate the retrieval contribution by replacing dense retrieval with BM25 while keeping the same reader family

Suggested implementation:

- run `fullwiki_eval_base_only.py` without `--faiss_index_dir`

### 6. Pseudo-Oracle Answer-Match

Purpose:

- separate answer-layer behavior from retrieval misses using an answer-string-matched pseudo-oracle context

Reference runner:

- `python3 fullwiki_eval_oracle_controls.py --track pseudo_oracle_answer_match ...`

### 7. Pseudo-Oracle Answer-Match + Poison

Purpose:

- measure answer destabilization when pseudo-oracle gold-matching evidence is contaminated with poison evidence

Reference runner:

- `python3 fullwiki_eval_oracle_controls.py --track pseudo_oracle_answer_match_plus_poison ...`

## Fixed split usage

All baselines must run on the same `QUESTION_MANIFEST`.

Example:

```bash
QUESTION_MANIFEST="benchmark_package/fixed_splits/public_all_500_balanced_seed42.json" ./run_base_only.sh
QUESTION_MANIFEST="benchmark_package/fixed_splits/public_all_500_balanced_seed42.json" BASE_USE_DOC_PREFIX=1 BASE_USE_2HOP=1 ./run_v24.sh
```

Recommended split artifacts:

- `public_all_500_balanced_seed42.json`
- `dev_all_500_balanced_seed42.json`
- `test_all_500_balanced_seed42.json`
- `public_hotpot_500_seed42.json`
- `public_nq_500_seed42.json`

## Required reporting

For each system, report:

- combined
- Hotpot
- NQ

Metrics:

- `ACC`
- `ASR`
- `CACC`
- `Abstain Rate`
- `Third-Answer Drift Rate`

## Official evaluation command

```bash
./run_official_eval.sh eval_results/aggressive_defense/example.csv
```

## Output schema expectations

Minimum per-system fields:

- `<prefix>_answer_raw`
- `<prefix>_answer_eval`
- `<prefix>_answer_final` or equivalent canonical answer field
- `<prefix>_best_raw` when applicable
- `<prefix>_abstain`

Shared fields:

- `ds`
- `id`
- `question`
- `gold_answer`
- `poison_target`

Recommended benchmark metadata fields:

- `benchmark_track`
- per-track role fields such as `clean_profile`, `base_track_role`, or `defense_track_role`
- provenance fields such as `manifest_selection_sha256`, `generator_model_id`, and `git_commit`
