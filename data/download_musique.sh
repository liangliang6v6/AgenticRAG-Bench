#!/usr/bin/env bash
# Download MuSiQue answerable dev and test splits.
# Source: https://github.com/StonyBrookNLP/musique
set -euo pipefail

OUT="${1:-$(dirname "$0")/musique}"
mkdir -p "$OUT"

python3 - <<'EOF'
import json
from datasets import load_dataset

out_dir = "$OUT"
ds = load_dataset("StonyBrookNLP/musique")
for split in ("validation", "test"):
    records = []
    for r in ds[split]:
        if not r.get("answerable", True):
            continue
        records.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "answers": r.get("answer_aliases", [r["answer"]]),
            "decomposition": r.get("question_decomposition", []),
        })
    out_path = f"{out_dir}/{split}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")
EOF

echo "Done. Files in $OUT"
