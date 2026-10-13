#!/usr/bin/env bash
# Start the SearchR1 dense retrieval server.
# Source: https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py
#
# Usage:
#   bash methods/searchr1/start_retrieval.sh \
#       --index-path  /path/to/e5_HNSW64.index \
#       --corpus-path /path/to/wiki-18.jsonl \
#       [--port 8000] \
#       [--topk 3]
#
# The retrieval server exposes:
#   POST /retrieve
#   Body: {"queries": ["..."], "topk": 3, "return_scores": true}
#   Returns: {"result": [[{"document": {"contents": "title\nbody"}, ...}]]}
#
# Default wiki-18 corpus and HNSW index can be built via:
#   git clone https://github.com/PeterGriffinJin/Search-R1
#   cd Search-R1 && bash download.sh  # downloads wiki-18 corpus
#   python search_r1/search/build_index.py ...  # builds HNSW index

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to index/corpus built by build_index.sh
INDEX_PATH="${SEARCHR1_INDEX_PATH:-$SCRIPT_DIR/retriever/e5_Flat.index}"
CORPUS_PATH="${SEARCHR1_CORPUS_PATH:-$SCRIPT_DIR/retriever/wiki-18.jsonl}"
TOPK=3
RETRIEVER="e5"
RETRIEVER_MODEL="intfloat/e5-base-v2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --index-path)    INDEX_PATH="$2";    shift 2 ;;
        --corpus-path)   CORPUS_PATH="$2";   shift 2 ;;
        --topk)          TOPK="$2";          shift 2 ;;
        --retriever)     RETRIEVER="$2";     shift 2 ;;
        --retriever-model) RETRIEVER_MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$INDEX_PATH" || ! -f "$CORPUS_PATH" ]]; then
    echo "ERROR: Index or corpus not found."
    echo "  INDEX_PATH:  $INDEX_PATH"
    echo "  CORPUS_PATH: $CORPUS_PATH"
    echo ""
    echo "Build them first with:"
    echo "  bash methods/searchr1/build_index.sh"
    exit 1
fi

echo "Starting SearchR1 retrieval server on port 8000..."
echo "Index:   $INDEX_PATH"
echo "Corpus:  $CORPUS_PATH"

python "$SCRIPT_DIR/src/retrieval_server.py" \
    --index_path    "$INDEX_PATH" \
    --corpus_path   "$CORPUS_PATH" \
    --topk          "$TOPK" \
    --retriever_name  "$RETRIEVER" \
    --retriever_model "$RETRIEVER_MODEL"
