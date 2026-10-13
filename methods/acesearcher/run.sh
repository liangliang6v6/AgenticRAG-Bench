#!/usr/bin/env bash
# Run AceSearcher on a dataset split.
#
# Usage:
#   bash methods/acesearcher/run.sh \
#       --dataset  2wikimhqa \
#       --split    validation \
#       --output   results/acesearcher/ \
#       --gpus     0,1
#
# Optional:
#   --model    AceSearcher/AceSearcher-14B   (default)
#   --size     1.5b|8b|14b|32b               (shorthand for --model)
#   --topk     10                             (number of retrieved passages)
#   --index-path  /path/to/embeddings/       (FAISS index directory)
#
# Environment:
#   HF_HOME  — HuggingFace cache directory (default: ~/.cache/huggingface)
#   ACESEARCHER_INDEX_ROOT — root for embeddings/<dataset>/ (default: ./embeddings)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
DATASET="2wikimhqa"
SPLIT="validation"
INPUT_OVERRIDE=""
OUTPUT=""
GPUS="0,1"
SIZE="14b"
MODEL=""
TOPK=10
INDEX_ROOT="${ACESEARCHER_INDEX_ROOT:-$SCRIPT_DIR/embeddings}"
MAX_MODEL_LEN=""
ENFORCE_EAGER=""

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET="$2";  shift 2 ;;
        --split)    SPLIT="$2";    shift 2 ;;
        --input)    INPUT_OVERRIDE="$2"; shift 2 ;;
        --output)   OUTPUT="$2";   shift 2 ;;
        --gpus)     GPUS="$2";     shift 2 ;;
        --model)    MODEL="$2";    shift 2 ;;
        --size)     SIZE="$2";     shift 2 ;;
        --topk)          TOPK="$2";          shift 2 ;;
        --index-path)    INDEX_ROOT="$2";    shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --enforce-eager) ENFORCE_EAGER="1";  shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── model selection ───────────────────────────────────────────────────────────
if [[ -z "$MODEL" ]]; then
    case "${SIZE,,}" in
        1.5b) MODEL="AceSearcher/AceSearcher-1.5B" ;;
        8b)   MODEL="AceSearcher/AceSearcher-8B" ;;
        14b)  MODEL="AceSearcher/AceSearcher-14B" ;;
        32b)  MODEL="AceSearcher/AceSearcher-32B" ;;
        *) echo "Unknown size: $SIZE. Use 1.5b, 8b, 14b, or 32b."; exit 1 ;;
    esac
fi

# ── dataset → dataset_key mapping ────────────────────────────────────────────
# Normalise user-friendly names to AceSearcher internal dataset keys
case "${DATASET,,}" in
    2wikimhqa|2wikimultihopqa) DATASET_KEY="2wikimultihopqa" ;;
    hotpotqa)                  DATASET_KEY="hotpotqa" ;;
    musique)                   DATASET_KEY="musique" ;;
    bamboogle)                 DATASET_KEY="bamboogle" ;;
    fever)                     DATASET_KEY="exfever" ;;
    hoverfever|hover)          DATASET_KEY="hover" ;;
    *) echo "Unsupported dataset: $DATASET"; exit 1 ;;
esac

INPUT="${INPUT_OVERRIDE:-$BENCH_ROOT/data/$DATASET/$SPLIT.jsonl}"
if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: Input file not found: $INPUT"
    echo "Run: bash data/download_${DATASET,,}.sh"
    exit 1
fi

OUTPUT="${OUTPUT:-$BENCH_ROOT/results/acesearcher/${DATASET_KEY}_${SPLIT}}"
mkdir -p "$OUTPUT"
INDEX_PATH="$INDEX_ROOT/$DATASET_KEY"

# embed-model-name: subfolder inside the index dir (may differ from HF model name)
case "$DATASET_KEY" in
    2wikimultihopqa|musique) EMBED_MODEL_NAME="e5-large" ;;
    hotpotqa)                EMBED_MODEL_NAME="e5-large-v2" ;;
    *)                       EMBED_MODEL_NAME="e5-large-v2" ;;
esac

DECOMPOSE_OUT="$OUTPUT/decompose.jsonl"
PREDICTIONS_OUT="$OUTPUT/predictions.jsonl"

N_GPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)

echo "=== AceSearcher ============================================"
echo "Model:     $MODEL"
echo "Dataset:   $DATASET_KEY  Split: $SPLIT  (n=$(wc -l < "$INPUT"))"
echo "Index:     $INDEX_PATH"
echo "GPUs:      $GPUS  (tensor_parallel=$N_GPU)"
echo "Output:    $OUTPUT"
echo "============================================================"

# Stage 1: Decompose
DECOMPOSE_EXTRA=()
[[ -n "$MAX_MODEL_LEN" ]] && DECOMPOSE_EXTRA+=(--max_model_len "$MAX_MODEL_LEN")
[[ -n "$ENFORCE_EAGER" ]] && DECOMPOSE_EXTRA+=(--enforce-eager)

echo "[1/2] Decomposing questions..."
CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/decompose.py" \
    --input    "$INPUT" \
    --output   "$DECOMPOSE_OUT" \
    --model    "$MODEL" \
    --dataset  "$DATASET_KEY" \
    --gpus     "$GPUS" \
    "${DECOMPOSE_EXTRA[@]}"

# Stage 2: Retrieve + Answer
SOLVE_EXTRA=()
[[ -n "$MAX_MODEL_LEN" ]] && SOLVE_EXTRA+=(--max-model-len "$MAX_MODEL_LEN")
[[ -n "$ENFORCE_EAGER" ]] && SOLVE_EXTRA+=(--enforce-eager)

echo "[2/2] Retrieving and answering..."
CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/solve.py" \
    --decomposed       "$DECOMPOSE_OUT" \
    --output           "$PREDICTIONS_OUT" \
    --index-path       "$INDEX_PATH" \
    --embed-model      "intfloat/e5-large-v2" \
    --embed-model-name "$EMBED_MODEL_NAME" \
    --model            "$MODEL" \
    --dataset          "$DATASET_KEY" \
    --gpus             "$GPUS" \
    --topk             "$TOPK" \
    --tensor-parallel-size "$N_GPU" \
    "${SOLVE_EXTRA[@]}"

# Evaluate
echo "[eval] Computing metrics..."
python "$BENCH_ROOT/eval/eval.py" \
    --predictions "$PREDICTIONS_OUT" \
    --output      "$OUTPUT/metrics.json"

echo "Done. Results: $OUTPUT/metrics.json"
