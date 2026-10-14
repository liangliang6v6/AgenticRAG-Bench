# RT-RAG

**Paper**: [Reasoning in Trees: Improving Retrieval-Augmented Generation for Multi-Hop Question Answering](https://arxiv.org/abs/2601.11255)  
**Venue**: arxiv 2025  
**Official Repository**: https://github.com/sakura20221/RT-RAG

---

## Method Overview

RT-RAG systematically decomposes complex multi-hop questions into **binary reasoning trees** and solves them via bottom-up traversal:

1. **Decompose**: Classify the question as Sequential, Parallel, or None.
   - **Sequential**: The second sub-question depends on the answer to the first (uses `[answer_subquestion1]` placeholder).
   - **Parallel**: Both sub-questions are independent.
   - **None**: The question is already atomic; retrieve and answer directly.
2. **Recurse**: Build left and right subtrees up to `max_depth`.
3. **Bottom-up solve**: Solve leaf nodes with retrieval, then compose answers up the tree.

### Prompt Source

System prompts adapted from [`sakura20221/RT-RAG/main/tree_decompose.py`](https://github.com/sakura20221/RT-RAG/blob/main/main/tree_decompose.py):

| Prompt | Purpose |
|---|---|
| `DECOMPOSE_SYSTEM` | Classify question type and decompose into sub-questions |
| `ANSWER_SYSTEM` | Answer a leaf node question with retrieved passages |
| `COMPOSE_SYSTEM` | Compose final answer from sub-question answers |

**Decomposition output format** (from the official code):
```json
{
  "type": "Sequential",
  "left": "Who directed Inception?",
  "right": "When did [answer_subquestion1] die?"
}
```

**Answer format**: ends with `so the Final answer is: [answer]`

---

## Models

| Model | Notes |
|---|---|
| `Qwen/Qwen2.5-14B-Instruct` | Default in this repo |
| `Qwen/Qwen2.5-7B-Instruct` | Smaller option |
| GPT-4o-mini | Original paper model (via OpenAI API) |

The official implementation uses **GPT-4o-mini** via OpenAI-compatible API.  
This adaptation replaces it with a local vLLM model.

---

## Supported Datasets

| Dataset | Official split (size) | Download |
|---|---|---|
| 2WikiMultiHopQA | dev, 12 576 q | `bash data/download.sh 2wikimultihopqa` |
| HotpotQA | fullwiki-dev, 7 405 q | `bash data/download.sh hotpotqa` |
| MuSiQue | answerable-dev, 2 417 q | `bash data/download.sh musique` |

---

## Retrieval Setup

RT-RAG uses a **FAISS dense retrieval** server (official).  
Any server matching the standard POST `/retrieve` API can substitute.

**Recommended**: Use the same retrieval server as SearchR1 (E5-base-v2 + HNSW64 FAISS):
```bash
bash methods/searchr1/start_retrieval.sh \
    --index-path  /path/to/e5_HNSW64.index \
    --corpus-path /path/to/wiki-18.jsonl \
    --port 8000
```

**Official setup** (from the repo):
```bash
git clone https://github.com/sakura20221/RT-RAG /tmp/RT-RAG
# Build FAISS index with their build_dense_index/ scripts
# Requires an embedding API endpoint (OpenAI-compatible)
```

---

## Differences from Official Implementation

| Feature | Official | This repo |
|---|---|---|
| LLM backend | GPT-4o-mini (OpenAI API) | vLLM (local) |
| Trees per question | 5 (consensus) | 1 (deterministic) |
| Retrieval | FAISS + BGE reranker | Standard `/retrieve` API |
| Async execution | asyncio | Sequential |

---

## Installation

```bash
pip install -r methods/rtrag/requirements.txt
# vllm>=0.4.0, transformers>=4.40.0, torch>=2.0, requests
```

---

## Quick Start

```bash
# 1. Start retrieval server
bash methods/searchr1/start_retrieval.sh \
    --index-path  /path/to/e5_HNSW64.index \
    --corpus-path /path/to/wiki-18.jsonl

# 2. Download dataset
bash data/download.sh hotpotqa

# 3. Run RT-RAG on HotpotQA
bash methods/rtrag/run.sh \
    --dataset      hotpotqa \
    --split        test \
    --output       results/rtrag/ \
    --retrieval-url http://127.0.0.1:8000/retrieve \
    --model        Qwen/Qwen2.5-14B-Instruct \
    --gpus         0,1
```

---

## Smoke Test

```bash
RTRAG_RETRIEVAL_URL=http://127.0.0.1:8000/retrieve \
bash methods/rtrag/test.sh
```

---

## Verified Results (from paper)

| Dataset | Model | EM |
|---|---|---|
| 2WikiMHQA | Qwen2.5-14B-Instruct | see paper |
| HotpotQA | Qwen2.5-14B-Instruct | see paper |
| MuSiQue | Qwen2.5-14B-Instruct | see paper |

See [arxiv 2601.11255](https://arxiv.org/abs/2601.11255) for full results.

---

## Citation

```bibtex
@article{rtrag2025,
    title   = {Reasoning in Trees: Improving Retrieval-Augmented Generation for Multi-Hop Question Answering},
    year    = {2025},
    url     = {https://arxiv.org/abs/2601.11255},
}
```
