#!/usr/bin/env bash
# Build FAISS retrieval indexes for AceSearcher datasets.
#
# Downloads each dataset's Wikipedia corpus from HuggingFace
# (AceSearcher/evaluation_datasets) and encodes it with the appropriate E5
# model, saving 4-shard embeddings-{0..3}-of-4.pkl files into:
#   embeddings/<dataset>/<embed_model_name>/
#
# These files are gitignored (see .gitignore) — run this once before using
# run.sh or test.sh.
#
# Usage:
#   bash methods/acesearcher/build_index.sh [--dataset 2wikimultihopqa] [--gpu 0]
#
# Options:
#   --dataset  <name>   Build only this dataset (default: all three)
#   --gpu      <id>     CUDA device to use (default: 0)
#   --batch-size <n>    Passages per batch (default: 256)
#
# Environment:
#   HF_HOME — HuggingFace cache directory (default: ~/.cache/huggingface)
#
# Approximate disk usage after build:
#   2WikiMHQA  ~1.8 GB   (416K passages × 1024-d float32 × 4 bytes × 4 shards)
#   HotpotQA   ~2.4 GB
#   MuSiQue    ~0.7 GB

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
DATASETS=("2wikimultihopqa" "hotpotqa" "musique")
GPU="0"
BATCH_SIZE="256"
INDEX_DIR="$SCRIPT_DIR/embeddings"

# ── args ──────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            case "${2,,}" in
                2wikimultihopqa|2wikimultihopqa) DATASETS=("2wikimultihopqa") ;;
                hotpotqa)                  DATASETS=("hotpotqa") ;;
                musique)                   DATASETS=("musique") ;;
                *) echo "Unknown dataset: $2"; exit 1 ;;
            esac
            shift 2 ;;
        --gpu)        GPU="$2";        shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$INDEX_DIR"

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "=== Building index: $DS ==="
    CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT_DIR/src/build_index.py" \
        --dataset    "$DS" \
        --index-dir  "$INDEX_DIR" \
        --batch-size "$BATCH_SIZE" \
        --device     cuda
done

echo ""
echo "All indexes built. Run the smoke test with:"
echo "  bash methods/acesearcher/test.sh"
