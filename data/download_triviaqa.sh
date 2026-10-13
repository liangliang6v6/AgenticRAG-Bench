#!/usr/bin/env bash
# Download TriviaQA unfiltered dev split.
# HuggingFace: trivia_qa (unfiltered.nocontext)
set -euo pipefail

OUT="${1:-$(dirname "$0")/triviaqa}"
mkdir -p "$OUT"

python3 - <<'EOF'
import json
from datasets import load_dataset

out_dir = "$OUT"
ds = load_dataset("trivia_qa", "unfiltered.nocontext")
for split in ("validation",):
    records = []
    for r in ds[split]:
        aliases = r["answer"].get("aliases", [])
        norm_aliases = r["answer"].get("normalized_aliases", aliases)
        records.append({
            "id": r["question_id"],
            "question": r["question"],
            "answer": r["answer"]["value"],
            "answers": list(set([r["answer"]["value"]] + aliases + norm_aliases)),
        })
    out_path = f"{out_dir}/{split}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")
EOF

echo "Done. Files in $OUT"
