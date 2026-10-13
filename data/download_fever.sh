#!/usr/bin/env bash
# Download FEVER and HoverFEVER evaluation splits.
# FEVER source: https://fever.ai/dataset/fever.html
# HoverFEVER source: https://github.com/hover-nlp/hover
set -euo pipefail

OUT="${1:-$(dirname "$0")/fever}"
mkdir -p "$OUT/fever" "$OUT/hoverfever"

# FEVER via HuggingFace
python3 - <<'EOF'
import json
from datasets import load_dataset

# FEVER
out_dir = "$OUT/fever"
ds = load_dataset("fever", "v1.0")
for split in ("validation", "test"):
    records = []
    for r in ds[split]:
        records.append({
            "id": str(r["id"]),
            "question": r["claim"],
            "answer": r["label"],
            "answers": [r["label"]],
        })
    out_path = f"{out_dir}/{split}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"FEVER: wrote {len(records)} to {out_path}")
EOF

# HoverFEVER from AceSearcher evaluation_datasets
python3 - <<'EOF'
import json
from datasets import load_dataset

out_dir = "$OUT/hoverfever"
try:
    ds = load_dataset("AceSearcher/evaluation_datasets", "hover")
    for split in ds.keys():
        records = []
        for r in ds[split]:
            records.append({
                "id": r.get("id", str(len(records))),
                "question": r["claim"],
                "answer": r["label"],
                "answers": [r["label"]],
            })
        out_path = f"{out_dir}/{split}.jsonl"
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"HoverFEVER: wrote {len(records)} to {out_path}")
except Exception as e:
    print(f"HoverFEVER download failed: {e}")
    print("Download manually from https://github.com/hover-nlp/hover")
EOF

echo "Done. Files in $OUT"
