#!/usr/bin/env bash
# Run RT-RAG (Reasoning in Trees for Retrieval-Augmented Generation).
#
# Paper: "Reasoning in Trees: Improving Retrieval-Augmented Generation
#         for Multi-Hop Question Answering" (arxiv 2601.11255)
# Official repo: https://github.com/sakura20221/RT-RAG
#
# Usage:
#   bash methods/rtrag/run.sh \
#       --dataset      hotpotqa \
#       --split        validation \
#       --output       results/rtrag/ \
#       [--retrieval-url http://127.0.0.1:8000/retrieve] \
#       [--model       Qwen/Qwen2.5-14B-Instruct] \
#       [--gpus        0,1] \
#       [--topk        5] \
#       [--max-depth   4]
#
# Retrieval server:
#   RT-RAG uses FAISS dense retrieval (official) or any compatible API.
#   Any server matching POST /retrieve API can be used.
#   See methods/searchr1/start_retrieval.sh for a compatible server.
#
# Notes:
#   - The official implementation uses GPT-4o-mini via OpenAI API.
#   - This adaptation uses vLLM with any instruction-tuned model.
#   - Official uses TREES_PER_QUESTION=5 with consensus; this uses single tree.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- defaults ----------
DATASET=""
SPLIT="test"
OUTPUT_DIR=""
RETRIEVAL_URL="${RTRAG_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
MODEL="${RTRAG_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
GPUS="0,1"
TOPK=5
MAX_DEPTH=4
MAX_MODEL_LEN=""

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";       shift 2 ;;
        --split)         SPLIT="$2";         shift 2 ;;
        --output)        OUTPUT_DIR="$2";    shift 2 ;;
        --retrieval-url) RETRIEVAL_URL="$2"; shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --gpus)          GPUS="$2";          shift 2 ;;
        --topk)          TOPK="$2";          shift 2 ;;
        --max-depth)     MAX_DEPTH="$2";     shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash run.sh --dataset <name> --output <dir> [options]"
    echo "Datasets: hotpotqa, 2wikimultihopqa, musique"
    exit 1
fi

# ---------- dataset name normalisation ----------
case "$DATASET" in
    hotpotqa)       DS_INTERNAL="hotpotqa" ;;
    2wikimultihopqa|2wikimultihopqa) DS_INTERNAL="2wikimultihopqa" ;;
    musique)        DS_INTERNAL="musique" ;;
    *) echo "ERROR: Unknown dataset '$DATASET'. Choose from: hotpotqa, 2wikimultihopqa, musique"; exit 1 ;;
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

echo "=== RT-RAG: $DATASET ($SPLIT) ==="
echo "  Model:       $MODEL"
echo "  Retrieval:   $RETRIEVAL_URL"
echo "  Input:       $INPUT_FILE"
echo "  Output:      $PREDICTIONS"
echo "  GPUs:        $GPUS"
echo "  Max depth:   $MAX_DEPTH"
echo ""

CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/infer.py" \
    --input         "$INPUT_FILE" \
    --output        "$PREDICTIONS" \
    --model         "$MODEL" \
    --retrieval-url "$RETRIEVAL_URL" \
    --gpus          "$GPUS" \
    --topk          "$TOPK" \
    --max-depth     "$MAX_DEPTH" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Evaluating ==="
python "$BENCH_ROOT/eval/eval.py" \
    --predictions "$PREDICTIONS" \
    --output      "$OUTPUT_DIR/$DATASET/metrics.json"
