#!/usr/bin/env bash
# Run Self-RAG inference.
#
# Paper: ICLR 2024 oral — Asai et al.
# "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
# Official repo: https://github.com/AkariAsai/self-rag
#
# Usage:
#   bash methods/selfrag/run.sh \
#       --dataset      triviaqa \
#       --split        validation \
#       --output       results/selfrag/ \
#       [--model       selfrag/selfrag_llama2_7b] \
#       [--retrieval-url http://127.0.0.1:8000/retrieve] \
#       [--gpus        0] \
#       [--topk        10]
#
# Models:
#   selfrag/selfrag_llama2_7b   (default)
#   selfrag/selfrag_llama2_13b
#
# Retrieval server:
#   Any server matching POST /retrieve API.
#   See methods/searchr1/start_retrieval.sh for a compatible server.
#
# Scoring weights (from official Self-RAG short-form protocol):
#   --w-rel 1.0 --w-sup 1.0 --w-use 0.5

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- defaults ----------
DATASET=""
SPLIT="test"
OUTPUT_DIR=""
MODEL="${SELFRAG_MODEL:-selfrag/selfrag_llama2_7b}"
RETRIEVAL_URL="${SELFRAG_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
GPUS="${SELFRAG_GPUS:-0}"
TOPK=10
MAX_TOKENS=100
BATCH_SIZE=8
W_REL=1.0
W_SUP=1.0
W_USE=0.5
MAX_MODEL_LEN=""

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";       shift 2 ;;
        --split)         SPLIT="$2";         shift 2 ;;
        --output)        OUTPUT_DIR="$2";    shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --retrieval-url) RETRIEVAL_URL="$2"; shift 2 ;;
        --gpus)          GPUS="$2";          shift 2 ;;
        --topk)          TOPK="$2";          shift 2 ;;
        --max-tokens)    MAX_TOKENS="$2";    shift 2 ;;
        --batch-size)    BATCH_SIZE="$2";    shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash run.sh --dataset <name> --output <dir> [options]"
    echo "Datasets: triviaqa, popqa, fever, hotpotqa"
    exit 1
fi

# ---------- dataset name normalisation ----------
case "$DATASET" in
    triviaqa)         DS_INTERNAL="triviaqa" ;;
    popqa)            DS_INTERNAL="popqa" ;;
    fever|hoverfever) DS_INTERNAL="fever" ;;
    hotpotqa)         DS_INTERNAL="hotpotqa" ;;
    *) echo "ERROR: Unknown dataset '$DATASET'. Choose from: triviaqa, popqa, fever, hotpotqa"; exit 1 ;;
esac

INPUT_FILE="$BENCH_ROOT/data/$DS_INTERNAL/$SPLIT.jsonl"
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Download it with: bash data/download.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/$DATASET"
PREDICTIONS="$OUTPUT_DIR/$DATASET/predictions.jsonl"

# ---------- build args ----------
EXTRA_ARGS=()
[[ -n "$MAX_MODEL_LEN" ]] && EXTRA_ARGS+=(--max-model-len "$MAX_MODEL_LEN")

echo "=== Self-RAG: $DATASET ($SPLIT) ==="
echo "  Model:       $MODEL"
echo "  Retrieval:   $RETRIEVAL_URL"
echo "  Input:       $INPUT_FILE"
echo "  Output:      $PREDICTIONS"
echo "  GPUs:        $GPUS"
echo ""

CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/infer.py" \
    --input         "$INPUT_FILE" \
    --output        "$PREDICTIONS" \
    --model         "$MODEL" \
    --retrieval-url "$RETRIEVAL_URL" \
    --gpus          "$GPUS" \
    --topk          "$TOPK" \
    --max-tokens    "$MAX_TOKENS" \
    --batch-size    "$BATCH_SIZE" \
    --w-rel         "$W_REL" \
    --w-sup         "$W_SUP" \
    --w-use         "$W_USE" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Evaluating ==="
python "$BENCH_ROOT/eval/eval.py" \
    --predictions "$PREDICTIONS" \
    --output      "$OUTPUT_DIR/$DATASET/metrics.json"
