# reference_systems/custom

These scripts are **controlled diagnostic reference systems** and are **not** the main representative baselines used in the paper.

| File | Role |
|------|------|
| `base_only_reference.py` | Clean-only pipeline; used in appendix as clean-ceiling reference |
| `fairness_v25_reference.py` | v25 defense pipeline; security–coverage trade-off diagnostic |
| `run_release_fairness.sh` | Canonical run script for fairness track (appendix only) |

## Usage context

- **Main results table**: FiD + ColBERTv2+Qwen2.5 (+ Atlas when available)
- **Appendix / supplementary**: these reference systems
- Do **not** describe `fairness_v25_reference.py` as "our defense" in main text.  
  It is a controlled reference demonstrating the security–coverage trade-off inherent to the benchmark's attack protocol.
