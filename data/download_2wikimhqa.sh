#!/usr/bin/env bash
# Download 2WikiMultiHopQA evaluation splits.
# Source: https://github.com/Alab-NII/2wikimultihop
# HuggingFace: Alab-NII/2wikimultihop
set -euo pipefail

OUT="${1:-$(dirname "$0")/2wikimhqa}"
mkdir -p "$OUT"

python3 - <<'EOF'
import json
from datasets import load_dataset

out_dir = "$OUT"
ds = load_dataset("Alab-NII/2wikimultihop")
for split in ("validation", "test"):
    records = []
    for r in ds[split]:
        records.append({
            "id": r["_id"],
            "question": r["question"],
            "answer": r["answer"],
            "answers": [r["answer"]],
            "supporting_facts": r.get("supporting_facts", []),
            "type": r.get("type", ""),
        })
    out_path = f"{out_dir}/{split}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")
EOF

echo "Done. Files in $OUT"
