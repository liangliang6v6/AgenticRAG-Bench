#!/usr/bin/env bash
# Download Bamboogle evaluation set from AceSearcher HuggingFace dataset.
# Source: AceSearcher/evaluation_datasets on HuggingFace
set -euo pipefail

OUT="${1:-$(dirname "$0")/bamboogle}"
mkdir -p "$OUT"

python3 - <<'EOF'
import json
from datasets import load_dataset

out_dir = "$OUT"
ds = load_dataset("AceSearcher/evaluation_datasets", "bamboogle")
for split in ds.keys():
    records = []
    for r in ds[split]:
        records.append({
            "id": r.get("id", str(len(records))),
            "question": r["question"],
            "answer": r["answer"] if isinstance(r["answer"], str) else r["answer"][0],
            "answers": r["answer"] if isinstance(r["answer"], list) else [r["answer"]],
        })
    out_path = f"{out_dir}/{split}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")
EOF

echo "Done. Files in $OUT"
