#!/usr/bin/env bash
# Run DecomP-ODQA (Decomposed Prompting for Open-Domain QA).
#
# Paper: ICLR 2023 — Khot, Trivedi, et al.
# Official repo: https://github.com/HarshTrivedi/DecomP-ODQA
#
# Usage:
#   bash methods/decomp/run.sh \
#       --dataset      hotpotqa \
#       --split        validation \
#       --output       results/decomp/ \
#       [--retrieval-url http://127.0.0.1:9200/retrieve] \
#       [--model       Qwen/Qwen2.5-14B-Instruct] \
#       [--prompts-dir /path/to/DecomP-ODQA/prompts] \
#       [--gpus        0,1] \
#       [--topk        5]
#
# Retrieval server:
#   DecomP-ODQA uses Elasticsearch BM25 (official) or any compatible API.
#   To start the official Elasticsearch-based server:
#     git clone https://github.com/HarshTrivedi/DecomP-ODQA /tmp/DecomP-ODQA
#     cd /tmp/DecomP-ODQA && pip install -r requirements.txt
#     # Start Elasticsearch 7.10, then:
#     python retriever_server/serve.py --index_name <dataset>_index --port 9200
#   API: POST {"queries": ["..."], "topk": 5, "return_scores": false}
#        → {"result": [[{"document": {"contents": "title\nbody"}, ...}]]}
#
# Prompts:
#   Official few-shot prompts from HarshTrivedi/DecomP-ODQA/prompts/<dataset>/
#   Clone the repo and pass --prompts-dir <repo>/prompts/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- defaults ----------
DATASET=""
SPLIT="validation"
OUTPUT_DIR=""
RETRIEVAL_URL="${DECOMP_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
MODEL="${DECOMP_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
PROMPTS_DIR="${DECOMP_PROMPTS_DIR:-}"
GPUS="0,1"
TOPK=5
MAX_SUB_QUESTIONS=4
MAX_MODEL_LEN=""

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)           DATASET="$2";            shift 2 ;;
        --split)             SPLIT="$2";              shift 2 ;;
        --output)            OUTPUT_DIR="$2";         shift 2 ;;
        --retrieval-url)     RETRIEVAL_URL="$2";      shift 2 ;;
        --model)             MODEL="$2";              shift 2 ;;
        --prompts-dir)       PROMPTS_DIR="$2";        shift 2 ;;
        --gpus)              GPUS="$2";               shift 2 ;;
        --topk)              TOPK="$2";               shift 2 ;;
        --max-sub-questions) MAX_SUB_QUESTIONS="$2";  shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash run.sh --dataset <name> --output <dir> [options]"
    echo "Datasets: hotpotqa, 2wikimhqa, musique"
    exit 1
fi

# ---------- dataset name normalisation ----------
case "$DATASET" in
    hotpotqa)       DS_INTERNAL="hotpotqa" ;;
    2wikimhqa|2wikimultihopqa) DS_INTERNAL="2wikimultihopqa" ;;
    musique)        DS_INTERNAL="musique" ;;
    *) echo "ERROR: Unknown dataset '$DATASET'. Choose from: hotpotqa, 2wikimhqa, musique"; exit 1 ;;
esac

INPUT_FILE="$BENCH_ROOT/data/$DS_INTERNAL/$SPLIT.jsonl"
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Download it with: bash data/download_${DS_INTERNAL}.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/$DATASET"
PREDICTIONS="$OUTPUT_DIR/$DATASET/predictions.jsonl"

# ---------- build args ----------
EXTRA_ARGS=()
[[ -n "$PROMPTS_DIR" ]]  && EXTRA_ARGS+=(--prompts-dir "$PROMPTS_DIR")
[[ -n "$MAX_MODEL_LEN" ]] && EXTRA_ARGS+=(--max-model-len "$MAX_MODEL_LEN")

echo "=== DecomP-ODQA: $DATASET ($SPLIT) ==="
echo "  Model:        $MODEL"
echo "  Retrieval:    $RETRIEVAL_URL"
echo "  Input:        $INPUT_FILE"
echo "  Output:       $PREDICTIONS"
echo "  GPUs:         $GPUS"
echo ""

CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/infer.py" \
    --input              "$INPUT_FILE" \
    --output             "$PREDICTIONS" \
    --dataset            "$DS_INTERNAL" \
    --model              "$MODEL" \
    --retrieval-url      "$RETRIEVAL_URL" \
    --gpus               "$GPUS" \
    --topk               "$TOPK" \
    --max-sub-questions  "$MAX_SUB_QUESTIONS" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Evaluating ==="
python "$BENCH_ROOT/eval/eval.py" \
    --predictions "$PREDICTIONS" \
    --output      "$OUTPUT_DIR/$DATASET/metrics.json"
