#!/usr/bin/env python3
"""Build final question manifest from QC-frozen release JSONL.

Outputs a JSON manifest with question IDs, per-dataset counts,
seed, and content-based SHA-256 for reproducibility tracking.
"""
import argparse
import hashlib
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_release", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = []
    with open(args.frozen_release) as f:
        for line in f:
            d = json.loads(line)
            questions.append({"ds": d["ds"], "id": d["id"]})

    ds_count = dict(Counter(q["ds"] for q in questions))
    content_hash = hashlib.sha256(
        json.dumps(questions, sort_keys=True).encode()
    ).hexdigest()

    manifest = {
        "name": f"final_manifest_qc_{len(questions)}_seed{args.seed}",
        "seed": args.seed,
        "n_questions": len(questions),
        "source_manifest": f"near_balanced_3145_seed{args.seed}",
        "qc_artifact": "frozen_release_full.jsonl",
        "qc_report": "qc_report_full.json",
        "tau_lex": 0.8,
        "questions": questions,
        "sha256": content_hash,
    }

    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Total: {len(questions)}")
    print(f"By dataset: {ds_count}")
    print(f"Content SHA-256: {content_hash}")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
