# Reviewer Guide

This document maps paper claims to code locations to facilitate reproducibility review. Every design choice listed here is intentional and aligned with paper claims. This guide exists to preempt any confusion for reviewers examining the codebase.

## 1. Threat Model: Candidate-Pool Staging (Paper S3.2)

Poison documents are staged into the candidate pool after first-stage BM25 retrieval:

- `baselines/modern_rag/colbert_qwen_runner.py` line 330:
  ```python
  merged = [p.strip() for p in poison_docs if p.strip()] + candidates
  ```
  Poison docs are prepended to BM25 top-1000 output, then ColBERT reranks the merged pool.

- `baselines/fid/fid_runner.py` line 282:
  ```python
  a_docs = [p.strip() for p in poison_docs if p.strip()] + a_docs
  ```
  Same pattern for FiD: poison prepended to BM25 top-100 pool.

This is NOT corpus-level injection (the BM25 index itself is clean). The threat model covers realistic adversarial conditions such as cache poisoning, plugin compromise, and mid-pipeline staging. Full corpus poisoning is explicitly out of scope (Paper S3.1).

## 2. Injection Budget: S = 6

- Generator default: `--k_poison 6` (`fullwiki_build_poison_docs.py` line 309)
- Runner default: `--poison_per_query 6` (`colbert_qwen_runner.py` line 249)
- All poison manifests contain exactly 6 sybils per question.

## 3. Forced Exposure: total_slots = 4

- `stress_protocols/stress_protocols.py` line 70: `build_forced_exposure_context(..., total_slots=4)`
- Gold positions: NQ/Wiki2=[0], Hotpot=[0,2] (line 116)
- Remaining slots filled with sybil documents from the frozen set.

The 4-slot budget is uniform across all baselines, deliberately smaller than each system's native top-K (10 for ColBERT+Qwen; 100 for FiD). This isolates reader-side conflict resolution under a worst-case stress condition (Paper S3.5).

## 4. Target Selection: Rule-Based (Paper S3.3)

`fullwiki_build_poison_docs.py:choose_wrong_answer()` (line 86):
- Yes/No answers: polarity flip
- Numeric answers: seeded `_perturb_number_str()` perturbation
- Textual answers: seeded random sample from benchmark-wide answer pool

No LLM is used for target selection. This ensures target distributions depend only on the source dataset's answer pool, not on generator model bias.

## 5. Sybil Generation: BM25-Grounded Substitution + LLM Paraphrase

`fullwiki_build_poison_docs.py`:
1. BM25 retrieves a gold-bearing document from corpus
2. `complete_replace_all()` (line 110) performs gold -> target substitution
3. `generate_llm_paraphrases()` (line 192) makes a single LLM call with Llama-3.1-8B-Instruct (Q8_0, temperature=0.8, max_tokens=2500)
4. Output split by `|||SEP|||` (line 261) into S=6 lexically varied sybils

Sybils are paraphrased variants of a shared poisoned base passage, not independent samples. This is a deliberate design to ground sybils in corpus-like content (Paper S3.3).

## 6. Verifier Rule: Group-Level Negative Filter

`benchmark_core/poison_qc.py` lines 359-360:
```python
n_gold_support = sum(1 for result in verifier_results
                     if result["supports_gold"] and not result["supports_target"])
verifier_ok = bool(verifier_results) and n_gold_support < max(1, (n_docs + 1) // 2)
```

A sybil group is rejected only if a majority of documents clearly support the GOLD answer. The rule is NEGATIVE (reject on gold support) rather than POSITIVE (require target support) because "ambiguous_or_neither" is a common verifier response for implicitly claim-bearing poison (Paper S3.3).

Verifier model: Qwen2.5-72B-Instruct (Q4_K_M), temperature=0.0, ternary judgment (supports_target / supports_gold / ambiguous_or_neither).

## 7. v25 Trigger Tokens: Appendix G Only

`reference_systems/custom/fairness_v25_reference.py` line 202:
```python
parser.add_argument("--trigger_tokens", type=str,
                    default="zjgqv zxqvjk qqqzzz tknnoise tknnoise")
```

Main Table 5 reports v25 re-evaluated with `--trigger_tokens ""` (trigger-free). The trigger-full configuration is reported only in Appendix G for sensitivity analysis. Line 495 shows where trigger tokens are appended to poison passages.

## 8. Canonicalization (Paper S3.4, S4)

Two canonicalization functions, used identically in construction and evaluation:

- `benchmark_eval_utils.py:_normalize_for_match()` (line 107): lowercase, `[^a-z0-9]+ -> space`, whitespace collapse
- `llm_answering.py:normalize_answer()` (line 253): lowercase, article stripping (a/an/the), `[^a-z0-9] -> space`, whitespace collapse

NOT applied: NFKC normalization, general stopword removal.

## 9. Forced Exposure Gold: Runtime BM25

`stress_protocols/stress_protocols.py:oracle_gold_docs()` (line 28): retrieves up to 2 passages per question by querying the BM25 index with gold answer strings, filtered by answer-string containment. Gold documents are NOT stored in the frozen manifest; they are recomputed at evaluation time.

This is acceptable because all primary baselines use the same BM25 index (wikipedia-dpr-100w). See Paper S8 Limitations.

## 10. Reproducibility

- Fixed seeds: generator seed=7, manifest sampling seed=42
- `run_metadata.py:build_run_metadata()` (line 64) records model IDs, prompt hashes, index hashes, git commits alongside every result CSV
- All frozen artifacts: `poison_docs_llm.frozen_qc.jsonl` (389 entries, S=6), `poison_docs_2wiki.frozen_qc.jsonl` (249 entries)
- Release specs: `release_specs/nq_hotpot_public500_v1.json`, `release_specs/2wiki_public250_v1.json`
