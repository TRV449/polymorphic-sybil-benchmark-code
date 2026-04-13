# Fixed Splits

This directory stores ordered question manifests used for benchmark runs.

Each official manifest contains:

- split name and split type
- dataset scope
- `selection_policy` with stable ordering and balancing metadata
- question counts by dataset
- selection checksum
- source dataset path / sha256 / question_count
- ordered `(ds, id)` entries

All clean / attack / defense pipelines should consume the same manifest through `--question_manifest` or `QUESTION_MANIFEST`.

Generation command examples:

```bash
python3 make_fixed_question_manifest.py \
  --hotpot datasets/hotpotqa_fixed.jsonl \
  --nq datasets/nq_fixed.jsonl \
  --dataset all \
  --max_questions 500 \
  --seed 42 \
  --output benchmark_package/fixed_splits/public_all_500_balanced_seed42.json \
  --split public
```

```bash
python3 make_fixed_question_manifest.py \
  --hotpot datasets/hotpotqa_fixed.jsonl \
  --nq datasets/nq_fixed.jsonl \
  --dataset all \
  --seed 42 \
  --output_dir benchmark_package/fixed_splits
```

Recommended release practice:

- use one immutable manifest per reported table
- report the manifest filename in every experiment log
- never compare clean / attack / defense numbers produced from different manifests
- use stable-hash ordering instead of runtime-dependent shuffling
- for `dataset=all`, prefer balanced selection unless there is a strong reason not to
- keep public/dev/test disjoint and validate overlap at generation time
- consumers should fail fast if manifest questions are missing from the source files
