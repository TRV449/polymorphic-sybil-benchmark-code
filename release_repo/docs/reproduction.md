# Reproduction Guide

## Prerequisites

- 2× NVIDIA A100 80GB (or equivalent)
- ≥256 GB RAM
- ~200 GPU-hours for full grid

## Setup

1. Clone this repo
2. `pip install -r requirements.txt`
3. Download retrieval indexes from HuggingFace:
   ```
   huggingface-cli download anon-neurips-ed-2026/polymorphic-sybil-benchmark-data --repo-type=dataset
   ```

## Running

### Quick verification
```bash
python3 -c "from evaluator.benchmark_eval_utils import classify_answer; print('Evaluator OK')"
head -1 data/main_hotpot_nq.jsonl | python3 -m json.tool
```

### Full main grid (5 readers × 2 retrievers × 3 tracks)
```bash
bash scripts/reproduce_main_grid.sh --reader qwen72b --port 8004 --backend llama_cpp_http
```

Repeat for each reader. See `configs/readers.yaml` for settings.

## Source Datasets

| Dataset | Split | Pool | Sampled | Seed |
|---|---|---:|---:|---|
| NQ-open | validation | 3,610 | 1,145 | 42 |
| HotpotQA distractor | dev | 7,405 | 2,000 | 42 |
| TriviaQA unfiltered.nocontext | validation | 11,313 | 2,000 | 42 |
| 2WikiMultiHopQA | dev | 12,576 | 3,000 | 42 |
