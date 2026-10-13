#!/usr/bin/env bash
# Smoke test: run IRCoT on 1 example per supported dataset.
# Requires: retrieval server running at $RETRIEVAL_URL (default: localhost:9200).
# Usage: bash methods/ircot/test.sh [--gpus 0] [--retrieval-url http://...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPUS="${IRCOT_GPUS:-5}"
MODEL="${IRCOT_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
MAX_MODEL_LEN="${IRCOT_MAX_MODEL_LEN:-24576}"
RETRIEVAL_URL="${IRCOT_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
PROMPTS_DIR="${IRCOT_PROMPTS_DIR:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)          GPUS="$2";          shift 2 ;;
        --retrieval-url) RETRIEVAL_URL="$2"; shift 2 ;;
        --prompts-dir)   PROMPTS_DIR="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Check retrieval server is up
if ! curl -s --max-time 2 "$RETRIEVAL_URL" > /dev/null 2>&1; then
    echo "WARNING: Retrieval server may not be responding at $RETRIEVAL_URL"
fi

DATASETS=("hotpotqa" "2wikimultihopqa" "musique" "iirc")

EXTRA_ARGS=()
[[ -n "$PROMPTS_DIR" ]] && EXTRA_ARGS+=(--prompts-dir "$PROMPTS_DIR")

for DS in "${DATASETS[@]}"; do
    FULL_INPUT="$BENCH_ROOT/data/$DS/validation.jsonl"
    [[ -f "$FULL_INPUT" ]] || { echo "SKIP $DS — data not found at $FULL_INPUT"; continue; }

    SMOKE_INPUT="/tmp/ircot_smoke_${DS}.jsonl"
    head -1 "$FULL_INPUT" > "$SMOKE_INPUT"
    SMOKE_OUT="/tmp/ircot_smoke_out_${DS}"
    mkdir -p "$SMOKE_OUT"

    echo "=== Smoke test: IRCoT on $DS (1 example) ==="
    CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/infer.py" \
        --input         "$SMOKE_INPUT" \
        --output        "$SMOKE_OUT/predictions.jsonl" \
        --dataset       "$DS" \
        --model         "$MODEL" \
        --retrieval-url "$RETRIEVAL_URL" \
        --gpus          "$GPUS" \
        --max-steps     7 \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager \
        "${EXTRA_ARGS[@]}" && \
    echo "PASS: $DS" || echo "FAIL: $DS"
done

echo "Smoke test done."
