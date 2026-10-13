#!/usr/bin/env bash
# Start a retrieval server for IRCoT.
#
# IRCoT was originally designed for Elasticsearch BM25.  This script starts
# a compatible dense-retrieval server (same POST /retrieve API) backed by
# the wiki-18 corpus and a FAISS Flat index.  Build the index first:
#
#   bash methods/searchr1/build_index.sh
#
# Then start the server:
#   bash methods/ircot/start_retrieval.sh
#
# Usage (override paths):
#   bash methods/ircot/start_retrieval.sh \
#       --index-path  /path/to/e5_Flat.index \
#       --corpus-path /path/to/wiki-18.jsonl \
#       [--port 8000]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SR1_DIR="$SCRIPT_DIR/../searchr1"

# Default: share index/corpus with SearchR1
INDEX_PATH="${IRCOT_INDEX_PATH:-$SR1_DIR/retriever/e5_Flat.index}"
CORPUS_PATH="${IRCOT_CORPUS_PATH:-$SR1_DIR/retriever/wiki-18.jsonl}"
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --index-path)  INDEX_PATH="$2";  shift 2 ;;
        --corpus-path) CORPUS_PATH="$2"; shift 2 ;;
        --port)        PORT="$2";        shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$INDEX_PATH" || ! -f "$CORPUS_PATH" ]]; then
    echo "ERROR: Index or corpus not found."
    echo "  INDEX_PATH:  $INDEX_PATH"
    echo "  CORPUS_PATH: $CORPUS_PATH"
    echo ""
    echo "Build the index first:"
    echo "  bash methods/searchr1/build_index.sh"
    exit 1
fi

echo "Starting IRCoT retrieval server on port $PORT ..."
python "$SR1_DIR/src/retrieval_server.py" \
    --index_path    "$INDEX_PATH" \
    --corpus_path   "$CORPUS_PATH" \
    --topk          6 \
    --retriever_name  "e5" \
    --retriever_model "intfloat/e5-base-v2" \
    --port          "$PORT"
