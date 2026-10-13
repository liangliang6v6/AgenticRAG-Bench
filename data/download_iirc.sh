#!/usr/bin/env bash
# Download IIRC (Initial Information Seeking with Reading Comprehension) evaluation set.
# Source: https://iirc-dataset.apps.allenai.org
# Used by: IRCoT
set -euo pipefail

OUT="${1:-$(dirname "$0")/iirc}"
mkdir -p "$OUT"

IIRC_URL="https://iirc-dataset.apps.allenai.org/data/iirc_train_dev.tgz"
TMP="$(mktemp -d)"
wget -c -O "$TMP/iirc.tgz" "$IIRC_URL"
tar -xzf "$TMP/iirc.tgz" -C "$TMP"

python3 - <<'EOF'
import json, glob, os

out_dir = "$OUT"
tmp_dir = "$TMP"

for split_file in glob.glob(f"{tmp_dir}/**/*.json", recursive=True):
    split = "validation" if "dev" in split_file else "train"
    with open(split_file) as f:
        data = json.load(f)
    records = []
    for passage in data:
        for q in passage.get("questions", []):
            answer = q.get("answer", {})
            ans_str = answer.get("answer_text", "")
            records.append({
                "id": q.get("qid", str(len(records))),
                "question": q["question"],
                "answer": ans_str,
                "answers": [ans_str],
                "context_title": passage.get("title", ""),
            })
    out_path = f"{out_dir}/{split}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")
EOF

rm -rf "$TMP"
echo "Done. Files in $OUT"
