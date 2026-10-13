#!/usr/bin/env bash
# Download PopQA evaluation set.
# Source: https://github.com/AlexTMallen/adaptive-retrieval
# HuggingFace: akariasai/PopQA
set -euo pipefail

OUT="${1:-$(dirname "$0")/popqa}"
mkdir -p "$OUT"

python3 - <<'EOF'
import json
from datasets import load_dataset

out_dir = "$OUT"
ds = load_dataset("akariasai/PopQA")
for split in ("test",):
    records = []
    for r in ds[split]:
        possible = r.get("possible_answers", [r["obj"]])
        if isinstance(possible, str):
            import ast
            try:
                possible = ast.literal_eval(possible)
            except Exception:
                possible = [possible]
        records.append({
            "id": str(r["id"]),
            "question": r["question"],
            "answer": r["obj"],
            "answers": possible,
            "prop": r.get("prop", ""),
        })
    out_path = f"{out_dir}/{split}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")
EOF

echo "Done. Files in $OUT"
