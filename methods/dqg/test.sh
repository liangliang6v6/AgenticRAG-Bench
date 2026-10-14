#!/usr/bin/env bash
# Smoke test: run DQG/FT-PANEL on 1 example per supported dataset.
# Requires: retrieval server running at $RETRIEVAL_URL.
# Usage: bash methods/dqg/test.sh [--gpus 0] [--retrieval-url http://...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPUS="${DQG_GPUS:-0,1}"
READER_MODEL="${DQG_READER_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
RETRIEVAL_URL="${DQG_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
ADAPTER="${DQG_ADAPTER_PATH:-$SCRIPT_DIR/adapter}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)          GPUS="$2";          shift 2 ;;
        --retrieval-url) RETRIEVAL_URL="$2"; shift 2 ;;
        --adapter)       ADAPTER="$2";       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

FIRST_GPU="${GPUS%%,*}"

# Check retrieval server
if ! curl -s --max-time 2 "$RETRIEVAL_URL" > /dev/null 2>&1; then
    echo "WARNING: Retrieval server may not be responding at $RETRIEVAL_URL"
fi

# Check adapter
if [[ ! -d "$ADAPTER" ]]; then
    echo "ERROR: FT-PANEL adapter not found: $ADAPTER"
    echo "Set DQG_ADAPTER_PATH or pass --adapter <path>"
    exit 1
fi

DATASETS=("hotpotqa" "2wikimultihopqa" "musique")

for DS in "${DATASETS[@]}"; do
    FULL_INPUT="$BENCH_ROOT/data/$DS/test.jsonl"
    [[ -f "$FULL_INPUT" ]] || { echo "SKIP $DS — data not found at $FULL_INPUT"; continue; }

    SMOKE_INPUT="/tmp/dqg_smoke_${DS}.jsonl"
    head -1 "$FULL_INPUT" > "$SMOKE_INPUT"
    SMOKE_OUT="/tmp/dqg_smoke_out_${DS}"
    mkdir -p "$SMOKE_OUT"

    echo "=== Smoke test: DQG/FT-PANEL on $DS (1 example) ==="
    SMOKE_DECOMP="$SMOKE_OUT/decomposed.jsonl"
    SMOKE_PRED="$SMOKE_OUT/predictions.jsonl"

    # Stage 1: decompose
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" python "$SCRIPT_DIR/src/decompose.py" \
        --input   "$SMOKE_INPUT" \
        --output  "$SMOKE_DECOMP" \
        --adapter "$ADAPTER" \
        --gpus    "$FIRST_GPU" && \

    # Stage 2: solve
    CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/solve.py" \
        --input         "$SMOKE_DECOMP" \
        --output        "$SMOKE_PRED" \
        --model         "$READER_MODEL" \
        --retrieval-url "$RETRIEVAL_URL" \
        --gpus          "$GPUS" && \
    echo "PASS: $DS" || echo "FAIL: $DS"
done

echo "Smoke test done."
