#!/usr/bin/env bash
# Build the wiki-18 retrieval index for SearchR1.
#
# Downloads the Wikipedia-2018 corpus (~14 GB) from HuggingFace
# (PeterJinGo/wiki-18) and builds a FAISS Flat index (~61 GB) using
# E5-base-v2.  Results are stored at:
#   methods/searchr1/retriever/wiki-18.jsonl
#   methods/searchr1/retriever/e5_Flat.index
#
# Both paths are gitignored — run this once before start_retrieval.sh.
#
# Usage:
#   bash methods/searchr1/build_index.sh [--gpu 0] [--batch-size 512]
#
# Environment:
#   HF_HOME — HuggingFace cache directory (default: ~/.cache/huggingface)
#
# Approximate times (single A100 80 GB):
#   Download corpus  : ~30 min
#   Build index      : ~4-6 h

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="0"
BATCH_SIZE="512"
OUT_DIR="$SCRIPT_DIR/retriever"
CORPUS_HF_REPO="PeterJinGo/wiki-18"          # HuggingFace dataset repo
RETRIEVER_MODEL="intfloat/e5-base-v2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU="$2";        shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"
CORPUS_PATH="$OUT_DIR/wiki-18.jsonl"
INDEX_PATH="$OUT_DIR/e5_Flat.index"

# ── Step 1: Download corpus ────────────────────────────────────────────────────
if [[ -f "$CORPUS_PATH" ]]; then
    echo "Corpus already exists: $CORPUS_PATH ($(wc -l < "$CORPUS_PATH") docs)"
else
    echo "[1/2] Downloading wiki-18 corpus from $CORPUS_HF_REPO ..."
    python3 - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
from pathlib import Path

out_dir = sys.argv[1] if len(sys.argv) > 1 else "retriever"
hf_home = os.environ.get("HF_HOME")
kwargs = {"cache_dir": hf_home} if hf_home else {}

snapshot_download(
    repo_id="PeterJinGo/wiki-18",
    repo_type="dataset",
    local_dir=out_dir,
    **kwargs,
)
PYEOF
    # Combine any split files into a single jsonl
    if [[ ! -f "$CORPUS_PATH" ]]; then
        echo "Merging split files ..."
        cat "$OUT_DIR"/part_* > "$CORPUS_PATH" 2>/dev/null || true
    fi
    echo "Corpus saved: $CORPUS_PATH"
fi

# ── Step 2: Build FAISS index ─────────────────────────────────────────────────
if [[ -f "$INDEX_PATH" ]]; then
    echo "Index already exists: $INDEX_PATH"
else
    echo "[2/2] Building FAISS Flat index (this takes several hours) ..."
    CUDA_VISIBLE_DEVICES="$GPU" python3 - <<PYEOF
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

corpus_path = "$CORPUS_PATH"
index_path  = "$INDEX_PATH"
model_name  = "$RETRIEVER_MODEL"
batch_size  = int("$BATCH_SIZE")
device      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading corpus from {corpus_path} ...")
corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
print(f"  {len(corpus)} documents loaded.")

print(f"Loading encoder {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(model_name).half().to(device).eval()

all_embs = []
texts = corpus["contents"]
for start in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
    batch = [f"passage: {t}" for t in texts[start:start+batch_size]]
    enc = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        last_hidden = out.last_hidden_state
        mask = enc["attention_mask"]
        emb = last_hidden.masked_fill(~mask[..., None].bool(), 0.0).sum(1) / mask.sum(1)[..., None]
        emb = torch.nn.functional.normalize(emb, dim=-1)
    all_embs.append(emb.cpu().float().numpy())

embeddings = np.concatenate(all_embs, axis=0).astype(np.float32, order="C")
print(f"Embedding matrix: {embeddings.shape}")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, index_path)
print(f"Index saved: {index_path}  ({index.ntotal} vectors)")
PYEOF
fi

echo ""
echo "Done. Start the retrieval server with:"
echo "  bash methods/searchr1/start_retrieval.sh"
