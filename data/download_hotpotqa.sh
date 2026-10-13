#!/usr/bin/env bash
# Download HotpotQA fullwiki dev and test splits.
# Source: https://hotpotqa.github.io
set -euo pipefail

OUT="${1:-$(dirname "$0")/hotpotqa}"
mkdir -p "$OUT"

DEV_URL="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
wget -c -O "$OUT/hotpot_dev_fullwiki_v1.json" "$DEV_URL"

python3 - <<'EOF'
import json, sys

out_dir = "$OUT"
with open(f"{out_dir}/hotpot_dev_fullwiki_v1.json") as f:
    data = json.load(f)

records = []
for r in data:
    records.append({
        "id": r["_id"],
        "question": r["question"],
        "answer": r["answer"],
        "answers": [r["answer"]],
        "supporting_facts": r.get("supporting_facts", []),
        "type": r.get("type", ""),
        "level": r.get("level", ""),
    })

out_path = f"{out_dir}/validation.jsonl"
with open(out_path, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
print(f"Wrote {len(records)} records to {out_path}")
EOF

echo "Done. Files in $OUT"
