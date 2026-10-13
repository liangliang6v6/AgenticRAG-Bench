#!/usr/bin/env bash
# Download Natural Questions (NQ) open dev split.
# HuggingFace: nq_open
set -euo pipefail

OUT="${1:-$(dirname "$0")/nq}"
mkdir -p "$OUT"

python3 - <<'EOF'
import json
from datasets import load_dataset

out_dir = "$OUT"
ds = load_dataset("nq_open")
for split in ("validation",):
    records = []
    for i, r in enumerate(ds[split]):
        records.append({
            "id": str(i),
            "question": r["question"],
            "answer": r["answer"][0] if r["answer"] else "",
            "answers": r["answer"],
        })
    out_path = f"{out_dir}/{split}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")
EOF

echo "Done. Files in $OUT"
