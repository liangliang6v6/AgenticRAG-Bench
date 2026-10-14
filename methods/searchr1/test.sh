#!/usr/bin/env bash
# Smoke test: run SearchR1 on 1 example per supported dataset.
# Requires: retrieval server running at $RETRIEVAL_URL (default: localhost:8000).
# Usage: bash methods/searchr1/test.sh [--gpus 0] [--retrieval-url http://...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPUS="${SEARCHR1_GPUS:-0}"
MODEL="${SEARCHR1_MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo}"
RETRIEVAL_URL="${SEARCHR1_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"

# Check retrieval server is up
if ! curl -s --max-time 2 "$RETRIEVAL_URL" > /dev/null 2>&1; then
    echo "WARNING: Retrieval server may not be running at $RETRIEVAL_URL"
    echo "Start it with: bash methods/searchr1/start_retrieval.sh --index-path <path> --corpus-path <path>"
fi

DATASETS=("hotpotqa" "2wikimultihopqa" "musique" "nq")

for DS in "${DATASETS[@]}"; do
    FULL_INPUT="$BENCH_ROOT/data/$DS/test.jsonl"
    [[ -f "$FULL_INPUT" ]] || { echo "SKIP $DS — data not found"; continue; }

    SMOKE_INPUT="/tmp/sr1_smoke_${DS}.jsonl"
    head -1 "$FULL_INPUT" > "$SMOKE_INPUT"
    SMOKE_OUT="/tmp/sr1_smoke_out_${DS}"
    mkdir -p "$SMOKE_OUT"

    echo "=== Smoke test: SearchR1 on $DS (1 example) ==="
    CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/infer.py" \
        --input         "$SMOKE_INPUT" \
        --output        "$SMOKE_OUT/predictions.jsonl" \
        --model         "$MODEL" \
        --retrieval-url "$RETRIEVAL_URL" \
        --gpus          "$GPUS" \
        --max-search-turns 5 && \
    echo "PASS: $DS" || echo "FAIL: $DS"
done

echo "Smoke test done."
