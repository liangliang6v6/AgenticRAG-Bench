#!/usr/bin/env python3
"""Build FAISS embeddings for AceSearcher datasets.

Downloads the Wikipedia corpus for a dataset from HuggingFace
(AceSearcher/evaluation_datasets) and encodes it with E5-large or E5-large-v2,
saving 4-shard .pkl files that solve.py expects.

Usage:
    python src/build_index.py \
        --dataset 2wikimultihopqa \
        --index-dir embeddings/ \
        [--embed-model intfloat/e5-large] \
        [--batch-size 256] \
        [--device cuda]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Dataset → embed model used by official AceSearcher checkpoints
DATASET_EMBED_MODEL = {
    "2wikimultihopqa": "intfloat/e5-large",
    "musique":         "intfloat/e5-large",
    "hotpotqa":        "intfloat/e5-large-v2",
}
DATASET_EMBED_NAME = {
    "2wikimultihopqa": "e5-large",
    "musique":         "e5-large",
    "hotpotqa":        "e5-large-v2",
}
# HuggingFace repo that hosts the corpus TSV files
HF_CORPUS_REPO = "AceSearcher/evaluation_datasets"


def download_corpus(dataset: str, out_dir: Path, hf_home: str | None = None) -> Path:
    """Download corpus.tsv from AceSearcher/evaluation_datasets if not present."""
    corpus_path = out_dir / "corpus.tsv"
    if corpus_path.exists():
        print(f"  corpus.tsv already exists at {corpus_path}, skipping download.")
        return corpus_path

    print(f"  Downloading corpus for {dataset} from {HF_CORPUS_REPO} ...")
    from huggingface_hub import hf_hub_download
    kwargs = {}
    if hf_home:
        kwargs["cache_dir"] = hf_home
    downloaded = hf_hub_download(
        repo_id=HF_CORPUS_REPO,
        filename=f"{dataset}/corpus.tsv",
        repo_type="dataset",
        local_dir=str(out_dir.parent),
        **kwargs,
    )
    # hf_hub_download writes to local_dir/<filename>
    src = Path(downloaded)
    if src != corpus_path:
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        src.rename(corpus_path)
    print(f"  Corpus saved to {corpus_path}")
    return corpus_path


def load_corpus(corpus_path: Path) -> list[str]:
    passages: list[str] = []
    with corpus_path.open() as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] == "id":
                continue
            passages.append(row[1])
    return passages


def embed_passages(
    passages: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    print(f"  Loading embed model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device).eval()

    all_embs: list[np.ndarray] = []
    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        # E5 models expect "passage: " prefix for corpus encoding
        prefixed = [f"passage: {p}" for p in batch]
        enc = tokenizer(
            prefixed,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
            emb = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
        all_embs.append(emb.cpu().float().numpy())
        if (start // batch_size) % 20 == 0:
            pct = min(start + batch_size, len(passages))
            print(f"    Embedded {pct}/{len(passages)} passages ...")

    return np.concatenate(all_embs, axis=0)


def save_shards(embeddings: np.ndarray, out_dir: Path, n_shards: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(embeddings)
    shard_size = math.ceil(total / n_shards)
    for i in range(n_shards):
        shard = embeddings[i * shard_size : (i + 1) * shard_size]
        shard_path = out_dir / f"embeddings-{i}-of-{n_shards}.pkl"
        with shard_path.open("wb") as f:
            pickle.dump(shard, f, protocol=4)
        print(f"  Saved shard {i}: {shard.shape} → {shard_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASET_EMBED_MODEL.keys()))
    parser.add_argument("--index-dir", default="embeddings",
                        help="Root directory; embeddings go into <index-dir>/<dataset>/")
    parser.add_argument("--embed-model", default=None,
                        help="Override HF embed model (default: per-dataset)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = args.dataset
    embed_model = args.embed_model or DATASET_EMBED_MODEL[dataset]
    embed_name  = DATASET_EMBED_NAME.get(dataset, embed_model.split("/")[-1])
    index_dir   = Path(args.index_dir) / dataset

    index_dir.mkdir(parents=True, exist_ok=True)
    hf_home = os.environ.get("HF_HOME")

    print(f"=== Building index for {dataset} ===")
    print(f"  Embed model : {embed_model}  ({embed_name})")
    print(f"  Index dir   : {index_dir}")
    print(f"  Device      : {args.device}")

    # Step 1: corpus
    corpus_path = download_corpus(dataset, index_dir, hf_home)
    print(f"  Loading corpus ...")
    passages = load_corpus(corpus_path)
    print(f"  Loaded {len(passages)} passages.")

    # Step 2: embed
    shard_dir = index_dir / embed_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    first_shard = shard_dir / "embeddings-0-of-4.pkl"
    if first_shard.exists():
        print(f"  Embeddings already exist at {shard_dir}, skipping.")
    else:
        embeddings = embed_passages(passages, embed_model, args.batch_size, args.device)
        print(f"  Embedding matrix: {embeddings.shape}")
        save_shards(embeddings, shard_dir)

    print(f"=== Done. Index at {index_dir} ===")


if __name__ == "__main__":
    main()
