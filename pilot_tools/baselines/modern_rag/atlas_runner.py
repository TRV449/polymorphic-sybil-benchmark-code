#!/usr/bin/env python3
"""
atlas_runner.py — Atlas (retriever + encoder-decoder LM) baseline.

Tracks: clean | attack   (forced: deferred to after clean/attack verified)

Atlas repo: https://github.com/facebookresearch/atlas
Install:
  git clone https://github.com/facebookresearch/atlas
  pip install -r atlas/requirements.txt
  # download pre-built index & checkpoint from Atlas model card

Usage example:
  python3 atlas_runner.py \
    --manifest   ../../benchmark_package/fixed_splits/public_all_500_balanced_seed42.json \
    --frozen_poison_jsonl $MEMBER_RUNTIME/poison_docs_llm.frozen_qc.jsonl \
    --track attack \
    --output_csv $MEMBER_RUNTIME/atlas_attack_500.csv \
    --atlas_model_path $MEMBER_RUNTIME/atlas_large \
    --atlas_index_path $WIKI_INDEXES/wikipedia-atlas

Output schema (per row):
  ds, id, question, gold_answer, poison_target,
  atlas_answer_raw, atlas_answer_eval, atlas_answer_final,
  atlas_abstain, atlas_conflict_flag
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List

from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "benchmark_core"))
sys.path.insert(0, _ROOT)

from benchmark_eval_utils import (
    eval_answer_from_raw,
    abstain_from_eval,
    conflict_flag_from_text,
    load_poison_maps_from_jsonl,
)
from manifest_utils import load_manifest, apply_manifest, validate_selected_questions
from official_eval import print_evaluation_report
from run_metadata import build_run_metadata, write_run_metadata

PREFIX = "atlas"


# ---------------------------------------------------------------------------
# Atlas inference stub
# ---------------------------------------------------------------------------

def atlas_predict(question: str, model, index, poison_prepend: List[str] | None = None) -> str:
    """Run Atlas retrieval + generation for one question.

    TODO: Implement using Atlas inference API.
    poison_prepend: if not None, prepend these docs to the retrieved candidate pool
                    before ranking (for attack track).

    Reference implementation in atlas/src/atlas.py:
      passages = index.search(question, k=100)
      if poison_prepend:
          passages = poison_prepend + passages
      answer = model.generate(question, passages[:10])
      return answer
    """
    raise NotImplementedError(
        "Atlas inference not yet implemented. "
        "Clone the Atlas repo and replace this stub."
    )


def load_atlas(model_path: str, index_path: str):
    """Load Atlas model and index.  Replace stub with real Atlas loader."""
    raise NotImplementedError("Atlas model loading not yet implemented.")


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_questions(hotpot_path: str, nq_path: str) -> List[dict]:
    questions = []
    if hotpot_path:
        with open(hotpot_path) as f:
            for line in f:
                d = json.loads(line)
                questions.append({"id": str(d["id"]), "q": d["question"], "a": [d["answer"]], "ds": "hotpot"})
    if nq_path:
        with open(nq_path) as f:
            for line in f:
                d = json.loads(line)
                ans = d.get("answer", [])
                if isinstance(ans, str):
                    ans = [ans]
                questions.append({"id": str(d["id"]), "q": d["question"], "a": ans, "ds": "nq"})
    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas baseline runner")
    parser.add_argument("--manifest",            required=True)
    parser.add_argument("--frozen_poison_jsonl", required=True)
    parser.add_argument("--track",               required=True, choices=["clean", "attack"])
    parser.add_argument("--output_csv",          required=True)
    parser.add_argument("--output_json",         default="")
    parser.add_argument("--hotpot",              default="")
    parser.add_argument("--nq",                  default="")
    parser.add_argument("--atlas_model_path",    required=True)
    parser.add_argument("--atlas_index_path",    required=True)
    parser.add_argument("--poison_per_query",    type=int, default=6)
    parser.add_argument("--overwrite_output",    action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_csv) and not args.overwrite_output:
        print(f"[atlas] output already exists, skipping: {args.output_csv}")
        return

    manifest  = load_manifest(args.manifest)
    questions = load_questions(args.hotpot, args.nq)
    questions = apply_manifest(questions, manifest)
    validate_selected_questions(questions, manifest)

    poison_map, poison_target_map = load_poison_maps_from_jsonl(args.frozen_poison_jsonl)

    model, index = load_atlas(args.atlas_model_path, args.atlas_index_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    fieldnames = [
        "ds", "id", "question", "gold_answer", "poison_target",
        f"{PREFIX}_answer_raw", f"{PREFIX}_answer_eval", f"{PREFIX}_answer_final",
        f"{PREFIX}_abstain", f"{PREFIX}_conflict_flag",
    ]

    rows = []
    for q in tqdm(questions, desc=f"atlas/{args.track}"):
        ds, qid, question, golds = q["ds"], q["id"], q["q"], q["a"]
        key = (ds, qid)
        poison_docs    = poison_map.get(key, [])[:args.poison_per_query]
        poison_targets = poison_target_map.get(key, [])

        prepend = poison_docs if args.track == "attack" else None

        try:
            answer_raw = atlas_predict(question, model, index, poison_prepend=prepend)
        except Exception as e:
            answer_raw = f"ERROR: {e}"

        answer_eval  = eval_answer_from_raw(answer_raw)
        answer_final = answer_eval
        abstain      = int(abstain_from_eval(answer_eval))
        conflict     = int(conflict_flag_from_text(answer_raw, golds, poison_targets))

        rows.append({
            "ds": ds,
            "id": qid,
            "question": question,
            "gold_answer": " | ".join(golds),
            "poison_target": " | ".join(poison_targets),
            f"{PREFIX}_answer_raw":    answer_raw,
            f"{PREFIX}_answer_eval":   answer_eval,
            f"{PREFIX}_answer_final":  answer_final,
            f"{PREFIX}_abstain":       abstain,
            f"{PREFIX}_conflict_flag": conflict,
        })

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print_evaluation_report(args.output_csv, system_prefixes=[PREFIX], strict_schema=True)

    meta = build_run_metadata(
        workspace_root=_ROOT,
        manifest=manifest,
        base_index=args.atlas_index_path,
        poison_jsonl=args.frozen_poison_jsonl,
        llm_backend="hf_local",
        llm_model_id=args.atlas_model_path,
        extra_fields={
            "atlas_track": args.track,
            "poison_per_query": args.poison_per_query,
            "n_questions": len(rows),
        },
    )
    write_run_metadata(args.output_csv, meta)
    print(f"[atlas] done → {args.output_csv}")


if __name__ == "__main__":
    main()
