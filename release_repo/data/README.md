# Data Schema

Each line in `main_hotpot_nq.jsonl` is a JSON object (one per retained sybil group):

```json
{
  "ds": "hotpot",
  "id": "5a8b57f25542995d1e6f1371",
  "gold_answers": ["yes"],
  "poison_targets": ["no", "no", "no", "no", "no", "no"],
  "poison_docs": ["passage_1...", "passage_2...", ..., "passage_6"]
}
```

| Field | Type | Description |
|---|---|---|
| `ds` | str | Source dataset (`hotpot`, `nq`, `trivia`, `wiki2`) |
| `id` | str | Question identifier |
| `gold_answers` | list[str] | Canonical gold answers |
| `poison_targets` | list[str] | Attacker-chosen target (one per sybil) |
| `poison_docs` | list[str] | S=6 polymorphic sybil passages |

## Files

| File | Groups | Source | τ_lex |
|---|---:|---|---|
| `main_hotpot_nq.jsonl` | 2,982 | NQ + HotpotQA | 0.8 |
| `cross_triviaqa.jsonl` | 1,398 | TriviaQA | 0.8 |
| `cross_2wiki.jsonl` | 2,696 | 2WikiMultiHopQA | 0.8 |
| `ablation_500q.jsonl` | 500 | Monomorphic baseline | N/A (diversity disabled) |
