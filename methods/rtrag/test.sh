#!/usr/bin/env bash
# Smoke test: run RT-RAG on 1 example per supported dataset.
# Requires: retrieval server running at $RETRIEVAL_URL (default: localhost:8000/retrieve).
# Usage: bash methods/rtrag/test.sh [--gpus 0] [--retrieval-url http://...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPUS="${RTRAG_GPUS:-5}"
MODEL="${RTRAG_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
MAX_MODEL_LEN="${RTRAG_MAX_MODEL_LEN:-24576}"
RETRIEVAL_URL="${RTRAG_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)          GPUS="$2";          shift 2 ;;
        --retrieval-url) RETRIEVAL_URL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Check retrieval server is up
if ! curl -s --max-time 2 "$RETRIEVAL_URL" > /dev/null 2>&1; then
    echo "WARNING: Retrieval server may not be responding at $RETRIEVAL_URL"
fi

DATASETS=("hotpotqa" "2wikimultihopqa" "musique")

for DS in "${DATASETS[@]}"; do
    FULL_INPUT="$BENCH_ROOT/data/$DS/validation.jsonl"
    [[ -f "$FULL_INPUT" ]] || { echo "SKIP $DS — data not found at $FULL_INPUT"; continue; }

    SMOKE_INPUT="/tmp/rtrag_smoke_${DS}.jsonl"
    head -1 "$FULL_INPUT" > "$SMOKE_INPUT"
    SMOKE_OUT="/tmp/rtrag_smoke_out_${DS}"
    mkdir -p "$SMOKE_OUT"

    echo "=== Smoke test: RT-RAG on $DS (1 example) ==="
    CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/infer.py" \
        --input         "$SMOKE_INPUT" \
        --output        "$SMOKE_OUT/predictions.jsonl" \
        --model         "$MODEL" \
        --retrieval-url "$RETRIEVAL_URL" \
        --gpus          "$GPUS" \
        --max-depth     4 \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager && \
    echo "PASS: $DS" || echo "FAIL: $DS"
done

echo "Smoke test done."
