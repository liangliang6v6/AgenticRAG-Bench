#!/usr/bin/env bash
# Smoke test: run AceSearcher on 1 example per supported dataset.
# Requires the FAISS index and at least 1 GPU.
# Usage: bash methods/acesearcher/test.sh [--gpus 0]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPUS="${ACESEARCHER_GPUS:-5}"
MODEL="${ACESEARCHER_MODEL:-AceSearcher/AceSearcher-14B}"
INDEX_ROOT="${ACESEARCHER_INDEX_ROOT:-$SCRIPT_DIR/embeddings}"
MAX_MODEL_LEN="${ACESEARCHER_MAX_MODEL_LEN:-8192}"

# bamboogle excluded: no FAISS index available locally
DATASETS=("2wikimhqa" "hotpotqa" "musique")

for DS in "${DATASETS[@]}"; do
    SPLIT="validation"
    FULL_INPUT="$BENCH_ROOT/data/$DS/$SPLIT.jsonl"
    if [[ ! -f "$FULL_INPUT" ]]; then
        echo "SKIP $DS — data not found at $FULL_INPUT"
        continue
    fi

    SMOKE_INPUT="/tmp/ace_smoke_${DS}.jsonl"
    head -1 "$FULL_INPUT" > "$SMOKE_INPUT"

    SMOKE_OUT="/tmp/ace_smoke_out_${DS}"
    mkdir -p "$SMOKE_OUT"

    echo "=== Smoke test: AceSearcher on $DS (1 example) ==="
    bash "$SCRIPT_DIR/run.sh" \
        --dataset       "$DS" \
        --split         "$SPLIT" \
        --input         "$SMOKE_INPUT" \
        --output        "$SMOKE_OUT" \
        --gpus          "$GPUS" \
        --model         "$MODEL" \
        --index-path    "$INDEX_ROOT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager || {
        echo "FAIL: $DS"
        continue
    }

    if [[ -f "$SMOKE_OUT/predictions.jsonl" ]]; then
        echo "PASS: $DS — $(python3 -c "import json; r=json.loads(open('$SMOKE_OUT/predictions.jsonl').readline()); print('prediction:', r['prediction'])")"
    fi
done

echo "Smoke test done."
