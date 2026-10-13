#!/usr/bin/env bash
# Run IRCoT (Interleaving Retrieval with Chain-of-Thought Reasoning).
#
# Paper: ACL 2023 — Trivedi et al.
# Official repo: https://github.com/StonyBrookNLP/ircot
#
# Usage:
#   bash methods/ircot/run.sh \
#       --dataset      hotpotqa \
#       --split        validation \
#       --output       results/ircot/ \
#       [--retrieval-url http://127.0.0.1:9200] \
#       [--model       Qwen/Qwen2.5-14B-Instruct] \
#       [--prompts-dir /path/to/ircot/prompts] \
#       [--gpus        0,1] \
#       [--topk        6] \
#       [--max-steps   7] \
#       [--max-model-len 4096]
#
# Retrieval server:
#   IRCoT was originally designed with Elasticsearch BM25.
#   Any server matching the POST /retrieve API can be used.
#   To start the official Elasticsearch-based server:
#     git clone https://github.com/StonyBrookNLP/ircot
#     cd ircot && pip install -r requirements.txt
#     # Start Elasticsearch (v7.10), then:
#     python retriever_server/serve.py \
#         --corpus_path <wiki_corpus.jsonl> \
#         --index_name  <dataset>_index
#   API: POST {"queries": ["..."], "topk": 6, "return_scores": false}
#        → {"result": [[{"document": {"contents": "..."}, ...}]]}
#
# Prompts:
#   Official few-shot CoT prompts from StonyBrookNLP/ircot/prompts/
#   Clone the repo and pass --prompts-dir ircot/prompts/
#
# Environment variables (override defaults):
#   IRCOT_RETRIEVAL_URL  — retrieval server URL
#   IRCOT_PROMPTS_DIR    — path to ircot/prompts/ directory
#   IRCOT_MODEL          — HuggingFace model ID

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- defaults ----------
DATASET=""
SPLIT="validation"
OUTPUT_DIR=""
RETRIEVAL_URL="${IRCOT_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
MODEL="${IRCOT_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
PROMPTS_DIR="${IRCOT_PROMPTS_DIR:-}"
GPUS="0,1"
TOPK=6
MAX_STEPS=7
MAX_MODEL_LEN=""

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";       shift 2 ;;
        --split)         SPLIT="$2";         shift 2 ;;
        --output)        OUTPUT_DIR="$2";    shift 2 ;;
        --retrieval-url) RETRIEVAL_URL="$2"; shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --prompts-dir)   PROMPTS_DIR="$2";   shift 2 ;;
        --gpus)          GPUS="$2";          shift 2 ;;
        --topk)          TOPK="$2";          shift 2 ;;
        --max-steps)     MAX_STEPS="$2";     shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash run.sh --dataset <name> --output <dir> [options]"
    echo "Datasets: hotpotqa, 2wikimhqa, musique, iirc"
    exit 1
fi

# ---------- dataset name normalisation ----------
case "$DATASET" in
    hotpotqa)       DS_INTERNAL="hotpotqa" ;;
    2wikimhqa|2wikimultihopqa) DS_INTERNAL="2wikimultihopqa" ;;
    musique)        DS_INTERNAL="musique" ;;
    iirc)           DS_INTERNAL="iirc" ;;
    *) echo "ERROR: Unknown dataset '$DATASET'. Choose from: hotpotqa, 2wikimhqa, musique, iirc"; exit 1 ;;
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
[[ -n "$PROMPTS_DIR" ]] && EXTRA_ARGS+=(--prompts-dir "$PROMPTS_DIR")
[[ -n "$MAX_MODEL_LEN" ]] && EXTRA_ARGS+=(--max-model-len "$MAX_MODEL_LEN")

echo "=== IRCoT: $DATASET ($SPLIT) ==="
echo "  Model:       $MODEL"
echo "  Retrieval:   $RETRIEVAL_URL"
echo "  Input:       $INPUT_FILE"
echo "  Output:      $PREDICTIONS"
echo "  GPUs:        $GPUS"
echo ""

CUDA_VISIBLE_DEVICES="$GPUS" python "$SCRIPT_DIR/src/infer.py" \
    --input         "$INPUT_FILE" \
    --output        "$PREDICTIONS" \
    --dataset       "$DS_INTERNAL" \
    --model         "$MODEL" \
    --retrieval-url "$RETRIEVAL_URL" \
    --gpus          "$GPUS" \
    --topk          "$TOPK" \
    --max-steps     "$MAX_STEPS" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Evaluating ==="
python "$BENCH_ROOT/eval/eval.py" \
    --predictions "$PREDICTIONS" \
    --output      "$OUTPUT_DIR/$DATASET/metrics.json"
