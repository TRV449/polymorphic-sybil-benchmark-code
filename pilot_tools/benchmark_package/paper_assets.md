# Paper Assets

## Positioning

This work should be framed as a benchmark for answer destabilization in grounded QA under retrieval poisoning, not as a generic RAG security benchmark and not as a pure target-hijack benchmark.

Recommended one-line framing:

- A failure-mode-aware benchmark for grounded answer destabilization under polymorphic sybil retrieval poisoning.

## Differentiation

- polymorphic sybil generation protocol
- answer-level failure taxonomy
- closed-set worst-case purification stress test
- official raw/eval split with reproducible evaluator
- explicit security-coverage trade-off reporting

## Related-work contrast

- `PoisonedRAG`: attack and ASR oriented
- `SafeRAG`: broader security benchmark framing
- `RAG Security Bench`: broad security + defense coverage
- `PoisonArena`: competitive multi-attacker setting
- `GaRAGe`: grounding quality benchmark without this poisoning-focused failure taxonomy

Safe comparison language:

- relative to broader security benchmarks, this benchmark narrows focus to answer-level failure decomposition after retrieval hijack
- relative to attack papers, this benchmark contributes a standardized evaluator and benchmark protocol rather than only a stronger attack
- relative to grounding benchmarks, this benchmark studies poisoned retrieval and purification stress rather than general grounding quality

## Title candidates

- Failure-Mode-Aware Benchmarking of Grounded Answer Destabilization under Retrieval Poisoning
- Polymorphic Sybil Poisoning Benchmark for Grounded QA
- Measuring Grounded Answer Destabilization under Polymorphic Sybil Retrieval Poisoning
- Beyond ASR: Benchmarking Grounded Answer Failures under Retrieval Hijack
- Gold, Hijack, Drift: Benchmarking Grounded QA under Polymorphic Sybil Poisoning

## Contribution sentences

- We introduce a benchmark for polymorphic sybil retrieval poisoning in grounded QA.
- We standardize answer-level failure analysis under retrieval hijack, including gold retention, targeted hijack, third-answer drift, and abstention.
- We provide a reproducible closed-set stress test for purification under retrieval poisoning.
- We release fixed splits, asset metadata, an official evaluator, and clean / attack / defense baselines.

## Abstract skeleton

1. RAG systems rely on retrieved evidence for grounded answering.
2. Retrieval poisoning often destabilizes answers even when it does not cleanly flip them to the attacker target.
3. Existing evaluation over-emphasizes ASR and under-measures answer-level failure structure.
4. We propose a benchmark centered on grounded answer destabilization under polymorphic sybil poisoning.
5. We release fixed splits, fixed asset metadata, an official evaluator, and baseline results for clean, attacked, and defended pipelines.
6. Results show a strong trade-off between attack suppression and answer coverage, with third-answer drift emerging as a major failure mode.

Suggested abstract close:

- Our results show that polymorphic sybil poisoning does not merely increase targeted hijack rates; it broadly destabilizes grounded answer generation by reducing gold retention, increasing third-answer drift, and exposing a security-coverage trade-off in purification defenses.

## Main table plan

### Table 1: Main benchmark results

Columns:

- System
- Dataset
- ACC
- ASR
- CACC
- Abstain
- Third-Answer Drift

Rows:

- Clean Base
- Attacked Baseline
- Defended Baseline

Split by:

- Hotpot
- NQ
- Combined

### Table 2: Failure taxonomy breakdown

Columns:

- System
- Gold Retained
- Targeted Hijack
- Third-Answer Drift
- Abstain

### Table 3: Base ceiling experiments

Columns:

- Clean reader design
- Retrieval unit
- Hotpot reader mode
- NQ reader mode
- Combined ACC
- Hotpot ACC
- NQ ACC

### Figure 1: Failure-mode composition

Recommended visualization:

- stacked bar over `Gold Retention / Targeted Hijack / Third-Answer Drift / Abstain`
- one bar per system for Combined, plus optional Hotpot and NQ panels
