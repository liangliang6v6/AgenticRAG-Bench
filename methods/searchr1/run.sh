#!/usr/bin/env bash
# Run SearchR1 on a QA dataset.
#
# Usage:
#   bash methods/searchr1/run.sh \
#       --dataset      hotpotqa \
#       --split        validation \
#       --output       results/searchr1/ \
#       --retrieval-url http://127.0.0.1:8000/retrieve \
#       --gpus         0
#
# Required: a retrieval server must be running before this script.
# See: methods/searchr1/start_retrieval.sh
#
# Optional:
#   --model    PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo (default)
#   --size     3b_base|3b_instruct|7b_base|7b_instruct (shorthand for --model)
#   --topk     3            (passages per search call)
#   --max-turns 10          (maximum search turns)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
DATASET="hotpotqa"
SPLIT="test"
OUTPUT=""
GPUS="0"
MODEL=""
SIZE="7b_base"
TOPK=3
MAX_TURNS=10
RETRIEVAL_URL="${SEARCHR1_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";       shift 2 ;;
        --split)         SPLIT="$2";         shift 2 ;;
        --output)        OUTPUT="$2";        shift 2 ;;
        --gpus)          GPUS="$2";          shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --size)          SIZE="$2";          shift 2 ;;
        --topk)          TOPK="$2";          shift 2 ;;
        --max-turns)     MAX_TURNS="$2";     shift 2 ;;
        --retrieval-url) RETRIEVAL_URL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── model selection ───────────────────────────────────────────────────────────
BASE_PREFIX="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5"
if [[ -z "$MODEL" ]]; then
    case "${SIZE,,}" in
        3b_base)      MODEL="${BASE_PREFIX}-3b-em-ppo" ;;
        3b_instruct)  MODEL="${BASE_PREFIX}-3b-it-ppo" ;;
        7b_base)      MODEL="${BASE_PREFIX}-7b-em-ppo" ;;
        7b_instruct)  MODEL="${BASE_PREFIX}-7b-it-ppo" ;;
        *) echo "Unknown size: $SIZE. Use 3b_base, 3b_instruct, 7b_base, or 7b_instruct."; exit 1 ;;
    esac
fi

# ── dataset normalisation ─────────────────────────────────────────────────────
case "${DATASET,,}" in
    2wikimultihopqa) DS_DIR="2wikimultihopqa" ;;
    hotpotqa)                  DS_DIR="hotpotqa"  ;;
    musique)                   DS_DIR="musique"   ;;
    nq)                        DS_DIR="nq"        ;;
    triviaqa)                  DS_DIR="triviaqa"  ;;
    popqa)                     DS_DIR="popqa"     ;;
    *) echo "Unsupported dataset: $DATASET"; exit 1 ;;
esac

INPUT="$BENCH_ROOT/data/$DS_DIR/$SPLIT.jsonl"
if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: Input file not found: $INPUT"
    echo "Run: bash data/download.sh"
    exit 1
fi

OUTPUT="${OUTPUT:-$BENCH_ROOT/results/searchr1/${DS_DIR}_${SPLIT}}"
mkdir -p "$OUTPUT"

echo "=== SearchR1 =============================================="
echo "Model:         $MODEL"
echo "Dataset:       $DS_DIR  Split: $SPLIT  (n=$(wc -l < "$INPUT"))"
echo "Retrieval URL: $RETRIEVAL_URL"
echo "GPUs:          $GPUS  TopK: $TOPK  MaxTurns: $MAX_TURNS"
echo "Output:        $OUTPUT"
echo "==========================================================="

CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/infer.py" \
    --input         "$INPUT" \
    --output        "$OUTPUT/predictions.jsonl" \
    --model         "$MODEL" \
    --retrieval-url "$RETRIEVAL_URL" \
    --gpus          "$GPUS" \
    --topk          "$TOPK" \
    --max-search-turns "$MAX_TURNS"

# Evaluate
python "$BENCH_ROOT/eval/eval.py" \
    --predictions "$OUTPUT/predictions.jsonl" \
    --output      "$OUTPUT/metrics.json"

echo "Done. Results: $OUTPUT/metrics.json"
