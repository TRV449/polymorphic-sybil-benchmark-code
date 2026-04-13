# Leaderboard Structure

## Standardized Prompt Track

The main leaderboard should compare model scale under a fixed setup:

- same `QUESTION_MANIFEST`
- same retrieval backend
- same reranker
- same prompt family
- same decoding defaults

Recommended model axis:

- `small_open`
- `medium_open`
- `large_open`

Recommended default open-weight family for this release:

- `Qwen2.5 GGUF + llama.cpp`

## Track separation

Do not mix these into a single undifferentiated table:

- `clean_strong`
- `attack_defense_fairness`
- `oracle_control`

## Required leaderboard columns

- `System`
- `Model Family`
- `Model Scale`
- `Backend`
- `Track`
- `Dataset`
- `ACC`
- `ASR`
- `CACC`
- `Abstain`
- `Third-Answer Drift`
- `current-op coverage (ASR<=0.10 check)`
- optional `frontier coverage@ASR<=0.10`

## Provenance columns

Every released row should be traceable to:

- `manifest_name`
- `manifest_selection_sha256`
- `source_hotpot_sha256`
- `source_nq_sha256`
- `index_hash`
- `attack_index_hash`
- `git_commit`
- `prompt_hash`
- `generator_model_id`
- `llm_backend`

## Appendix recommendations

Appendix tables can add:

- `Qwen vs Llama`
- `tuned track`
- `Hotpot pair-aware clean reader`
- `pseudo-oracle answer-match`
- `pseudo-oracle answer-match + poison`
