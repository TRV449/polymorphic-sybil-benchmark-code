# Appendix A: Quality Control Methodology

## A.1 Verifier Configuration

| Parameter        | Value                              |
|------------------|------------------------------------|
| Model            | Qwen2.5-72B-Instruct (GGUF Q4_K_M)|
| Server           | llama.cpp, port 8003               |
| Temperature      | 0.0                                |
| max_tokens       | 32                                 |
| Prompt hash      | `0a4a02c92506d9929ca95917417abc35ac87271cd2f54de834d41d633268f2cb` |

The verifier evaluates each sybil document individually against two questions:
1. Does this passage support the **gold** (correct) answer?
2. Does this passage support the **target** (adversarial) answer?

Outputs are parsed into three labels:
- `supports_t`: supports_target=True AND supports_gold=False
- `supports_g`: supports_gold=True (regardless of supports_target)
- `ambiguous_or_neither`: neither clear support detected

## A.2 Acceptance Rules (Pseudocode)

```
ACCEPT(group G with S sybil documents P = {p_1, ..., p_S}):

  # Gate 1: Semantic Disjoint
  REJECT if gold_answer == target_answer  (string or semantic equivalence)

  # Gate 2: Pairwise Lexical Diversity
  FOR all pairs (p_i, p_j) in P:
    REJECT if Jaccard(words(p_i), words(p_j)) > tau_lex

  # Gate 3: Verifier Negative Filter
  n_gold_only = |{p in P : supports_gold(p) = T AND supports_target(p) = F}|
  REJECT if n_gold_only >= ceil(S/2)
    where ceil(S/2) = max(1, (S+1)//2)
    For S=6: threshold = 3

  ACCEPT group
```

### Gate 3 Detail

The verifier negative filter removes groups where too many sybil documents reinforce the gold answer without supporting the target. This prevents "sybil documents" that would actually help the reader find the correct answer.

**Counting rule**: Only gold-*only* documents count toward rejection. Documents labeled `supports_g` where `supports_target=True` are NOT counted — they support both answers and thus represent genuine ambiguity, which is the intended attack vector.

**Threshold**: `ceil(S/2) = max(1, (S+1)//2)`. For S=6 documents, groups are rejected when 3 or more documents are gold-only. This is a conservative threshold: a bare majority of gold-reinforcing documents suffices for rejection.

## A.3 QC Pipeline Statistics

| Metric                    | Value      |
|---------------------------|------------|
| Input poison groups       | 3,145      |
| Kept                      | 2,982 (94.8%) |
| Dropped                   | 163 (5.2%)    |
| — Verifier negative filter| 160        |
| — Semantic disjoint       | 2          |
| — Both                    | 1          |

### Per-Dataset Breakdown
| Dataset   | Kept  | Dropped | Keep Rate |
|-----------|-------|---------|-----------|
| HotpotQA  | 1,876 | 124     | 93.8%     |
| NQ        | 1,106 | 39      | 96.6%     |

### Lexical Diversity (tau_lex = 0.8)
| Statistic              | Value |
|------------------------|-------|
| Mean pairwise Jaccard  | 0.323 |
| Max pairwise Jaccard   | 0.600 |
| Groups rejected (lex)  | 0     |

All 3,145 input groups passed the lexical diversity gate (tau_lex=0.8), since they had already passed a stricter tau_lex=0.6 filter in the upstream v2 QC stage.

## A.4 Integrity Hashes

| Artifact                           | SHA-256 (first 16 chars)        |
|------------------------------------|---------------------------------|
| Final manifest (2,982 Q)           | `b875d9062b78a499`              |
| Frozen poison JSONL                | (computed at release)           |
| System prompt                      | `bfda716de5b9f040`              |
| Full prompt template               | `f7e85a5b63b34a94`              |
| Verifier prompt                    | `0a4a02c92506d992`              |

Manifest hash computed with:
```python
canon = json.dumps(questions, separators=(",", ":"), sort_keys=True)
sha256 = hashlib.sha256(canon.encode()).hexdigest()
```

## A.5 Manifest Lineage

```
Raw poison JSONL (11,015 groups)
  |-- v2 QC: tau_lex=0.6, semantic filter --> 4,022 (36.5%)
      |-- Near-balanced sample: seed=42, 2000 Hotpot + 1145 NQ --> 3,145 (28.5%)
          |-- Main QC: tau_lex=0.8, verifier, disjoint --> 2,982 (27.1%)
```

## A.6 Human Audit Protocol (Planned)

To validate verifier accuracy, we plan a stratified audit:
- **100 accepted groups**: Random sample from the 2,982 kept groups
- **50 rejected groups**: Random sample from the 163 dropped groups

Each group is independently annotated by 2 annotators for:
1. Does each sybil document support the gold answer?
2. Does each sybil document support the target answer?
3. Should this group be accepted or rejected?

Inter-annotator agreement will be reported with Cohen's kappa.
