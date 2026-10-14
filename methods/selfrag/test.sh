#!/usr/bin/env bash
# Smoke test: run Self-RAG on 1 example per supported dataset.
# Requires: retrieval server running at $RETRIEVAL_URL (default: localhost:8000/retrieve).
# Usage: bash methods/selfrag/test.sh [--gpus 0] [--retrieval-url http://...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPUS="${SELFRAG_GPUS:-0}"
MODEL="${SELFRAG_MODEL:-selfrag/selfrag_llama2_7b}"
RETRIEVAL_URL="${SELFRAG_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)          GPUS="$2";          shift 2 ;;
        --retrieval-url) RETRIEVAL_URL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Check retrieval server
if ! curl -s --max-time 2 "$RETRIEVAL_URL" > /dev/null 2>&1; then
    echo "WARNING: Retrieval server may not be responding at $RETRIEVAL_URL"
fi

DATASETS=("triviaqa" "popqa" "fever")

for DS in "${DATASETS[@]}"; do
    FULL_INPUT="$BENCH_ROOT/data/$DS/test.jsonl"
    [[ -f "$FULL_INPUT" ]] || { echo "SKIP $DS — data not found at $FULL_INPUT"; continue; }

    SMOKE_INPUT="/tmp/selfrag_smoke_${DS}.jsonl"
    head -1 "$FULL_INPUT" > "$SMOKE_INPUT"
    SMOKE_OUT="/tmp/selfrag_smoke_out_${DS}"
    mkdir -p "$SMOKE_OUT"

    echo "=== Smoke test: Self-RAG on $DS (1 example) ==="
    CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/infer.py" \
        --input         "$SMOKE_INPUT" \
        --output        "$SMOKE_OUT/predictions.jsonl" \
        --model         "$MODEL" \
        --retrieval-url "$RETRIEVAL_URL" \
        --gpus          "$GPUS" \
        --topk          5 \
        --batch-size    1 && \
    echo "PASS: $DS" || echo "FAIL: $DS"
done

echo "Smoke test done."
