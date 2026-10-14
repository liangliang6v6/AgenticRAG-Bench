#!/usr/bin/env bash
# Run DQG/FT-PANEL (two-stage: decompose → retrieve + answer).
#
# Paper: NAACL 2025 — Han & Gardent
# "Generating Complex Question Decompositions in the Face of Distribution Shifts"
# Official repo: https://github.com/hankelvin/complex_question_decomposition
#
# Stage 1: Decompose questions with FT-PANEL LoRA adapter (Qwen2.5-7B-Instruct)
# Stage 2: Retrieve passages for each sub-question and answer with a reader model
#
# Usage:
#   bash methods/dqg/run.sh \
#       --dataset      hotpotqa \
#       --split        validation \
#       --output       results/dqg/ \
#       --adapter      /path/to/ft_panel_lora \
#       [--retrieval-url http://127.0.0.1:8000/retrieve] \
#       [--reader-model Qwen/Qwen2.5-14B-Instruct] \
#       [--gpus        0,1] \
#       [--topk        5]
#
# FT-PANEL LoRA adapter:
#   Download from: https://github.com/hankelvin/complex_question_decomposition
#   or use local path: /data2/liang/data3/code/baseline/models/results/dqg_dqa/
#
# Retrieval server:
#   Any server matching POST /retrieve API.
#   See methods/searchr1/start_retrieval.sh for a compatible server.
#
# Environment variables:
#   DQG_ADAPTER_PATH   — path to FT-PANEL LoRA adapter
#   DQG_RETRIEVAL_URL  — retrieval server URL
#   DQG_READER_MODEL   — reader model HF ID
#   DQG_GPUS           — GPU IDs

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- defaults ----------
DATASET=""
SPLIT="test"
OUTPUT_DIR=""
ADAPTER="${DQG_ADAPTER_PATH:-}"
RETRIEVAL_URL="${DQG_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
READER_MODEL="${DQG_READER_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
GPUS="${DQG_GPUS:-0,1}"
TOPK=5
MAX_MODEL_LEN=""

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";       shift 2 ;;
        --split)         SPLIT="$2";         shift 2 ;;
        --output)        OUTPUT_DIR="$2";    shift 2 ;;
        --adapter)       ADAPTER="$2";       shift 2 ;;
        --retrieval-url) RETRIEVAL_URL="$2"; shift 2 ;;
        --reader-model)  READER_MODEL="$2";  shift 2 ;;
        --gpus)          GPUS="$2";          shift 2 ;;
        --topk)          TOPK="$2";          shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash run.sh --dataset <name> --output <dir> [--adapter <path>] [options]"
    echo "Datasets: hotpotqa, 2wikimultihopqa, musique"
    exit 1
fi

# Default adapter path: methods/dqg/adapter/ (gitignored — download separately)
if [[ -z "$ADAPTER" ]]; then
    ADAPTER="$SCRIPT_DIR/adapter"
fi

if [[ ! -d "$ADAPTER" ]]; then
    echo "ERROR: FT-PANEL adapter not found: $ADAPTER"
    echo ""
    echo "Download the FT-PANEL LoRA adapter:"
    echo "  1. Clone the official repo:"
    echo "     git clone https://github.com/hankelvin/complex_question_decomposition"
    echo "  2. Follow their README to download the model checkpoint."
    echo "  3. Place the adapter at: $ADAPTER"
    echo ""
    echo "Or set DQG_ADAPTER_PATH to an existing adapter directory."
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
DECOMPOSED="$OUTPUT_DIR/$DATASET/decomposed.jsonl"
PREDICTIONS="$OUTPUT_DIR/$DATASET/predictions.jsonl"

EXTRA_ARGS=()
[[ -n "$MAX_MODEL_LEN" ]] && EXTRA_ARGS+=(--max-model-len "$MAX_MODEL_LEN")

echo "=== DQG/FT-PANEL: $DATASET ($SPLIT) ==="
echo "  Adapter:     $ADAPTER"
echo "  Reader:      $READER_MODEL"
echo "  Retrieval:   $RETRIEVAL_URL"
echo ""

# ---------- Stage 1: Decompose ----------
echo "--- Stage 1: Question decomposition (FT-PANEL) ---"
# Use first GPU for the smaller decomposer model
FIRST_GPU="${GPUS%%,*}"
CUDA_VISIBLE_DEVICES="$FIRST_GPU" python "$SCRIPT_DIR/src/decompose.py" \
    --input    "$INPUT_FILE" \
    --output   "$DECOMPOSED" \
    --adapter  "$ADAPTER" \
    --gpus     "$FIRST_GPU"

# ---------- Stage 2: Retrieve + Answer ----------
echo ""
echo "--- Stage 2: Retrieve and answer sub-questions ---"
CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/solve.py" \
    --input         "$DECOMPOSED" \
    --output        "$PREDICTIONS" \
    --model         "$READER_MODEL" \
    --retrieval-url "$RETRIEVAL_URL" \
    --gpus          "$GPUS" \
    --topk          "$TOPK" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Evaluating ==="
python "$BENCH_ROOT/eval/eval.py" \
    --predictions "$PREDICTIONS" \
    --output      "$OUTPUT_DIR/$DATASET/metrics.json"
