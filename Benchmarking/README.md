# Benchmarking

Curated view of paper-relevant code and results. All files are symlinks to `pilot_tools/` (the working development directory). Running experiments are not affected by this directory.

## Directory Layout

```
core/       Benchmark evaluation, QC, transition matrix, manifest utilities
infra/      LLM backend dispatch, reader prompts, model registry
runners/    All experiment runners (ColBERT+Qwen, FiD, v25, oracle controls)
poison/     Sybil document generator
defense/    v25 purification modules (MLM veto, clustering, consensus)
specs/      Release specifications and fixed question manifests
results/    All experiment result CSVs (symlink to member_runtime/)
```

## Key Files

| File | Role | Paper Section |
|------|------|---------------|
| `runners/colbert_qwen_runner.py` | Main practical baseline (ColBERT rerank + Qwen reader) | S5, Table 3 |
| `runners/fid_runner.py` | Heterogeneous baseline (FiD-large) | S5, Table 3 |
| `runners/fairness_v25_reference.py` | Defense reference (collapse + MLM veto + consensus) | S5, Table 5 |
| `runners/oracle_controls.py` | Pseudo-oracle and forced exposure protocols | S3.5, S5 |
| `core/official_eval.py` | Official evaluator (ACC, ASR, CACC, abstain, drift) | S4 |
| `core/transition_matrix.py` | Paired transition analysis + bootstrap CI | S7 |
| `core/poison_qc.py` | Sybil QC: lexical diversity + verifier | S3.3 |
| `poison/fullwiki_build_poison_docs.py` | Sybil generation (BM25 grounded + LLM paraphrase) | S3.3 |
| `infra/llm_answering.py` | Reader prompt (COMMON_QA_SYSTEM_PROMPT) + answer extraction | S4 |
| `REVIEWER_GUIDE.md` | Maps paper claims to code locations | Reviewer reference |

## Running Experiments

Runners must be invoked from their original `pilot_tools/` paths (not from this directory) because they compute `_ROOT` via `__file__` for import resolution. Example:

```bash
cd pilot_tools/baselines/modern_rag
python3 colbert_qwen_runner.py --manifest ... --track clean --output_csv ...
```

## Results

`results/` links to `member_runtime/` which contains all experiment CSVs. Key files:
- `release_colbert_locked_{clean,attack,forced}_500_*.csv` - Main ColBERT results
- `release_fid_v2_500_*.csv` - FiD results (all tracks in one CSV)
- `release_colbert_2wiki_*.csv` / `release_fid_2wiki_*.csv` - 2Wiki validation
- `ablation_colbert_attack_{s4,tau06}_*.csv` - S/tau_lex ablation
- `colbert_locked_transitions.json` - Paired transition matrices + CI
- `clean_drift_audit_100.csv` - Clean drift cause analysis
