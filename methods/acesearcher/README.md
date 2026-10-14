# AceSearcher

**Paper**: [AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](https://arxiv.org/abs/2509.24193)  
**Venue**: NeurIPS 2025 Spotlight  
**Authors**: Ran Xu, Yuchen Zhuang, Zihan Dong, Ruiyu Wang, Yue Yu, Joyce C. Ho, Linjun Zhang, Haoyu Wang, Wenqi Shi, Carl Yang  
**Official Repository**: https://github.com/ritaranx/AceSearcher  
**Official Repo Commit**: tracked at time of integration

---

## Method Overview

AceSearcher jointly trains a **decomposer** and **solver** via reinforced self-play:

1. **Decompose** (Stage 1): given a multi-hop question, a planning LLM breaks it into ordered atomic sub-questions (`### Q1:`, `### Q2:`, …), from most-selective to least-selective.
2. **Retrieve + Answer** (Stage 2): for each sub-question, top-k passages are retrieved from a FAISS dense index (E5-large-v2 embeddings). The same LLM answers each sub-question given its passages, propagating answers via `#1`, `#2` placeholders. A final synthesis call combines all sub-answers.

Both stages use the **same checkpoint** — no separate planner and reader model.

### Prompt Sources

| Prompt | Source |
|---|---|
| Decompose (QA) | [`inference/decompose_vllm.py`](https://github.com/ritaranx/AceSearcher/blob/main/inference/decompose_vllm.py) lines 22–41 |
| Sub-question answer | [`inference/main_qa.py`](https://github.com/ritaranx/AceSearcher/blob/main/inference/main_qa.py) lines 144–157 |
| Final synthesis | [`inference/main_qa.py`](https://github.com/ritaranx/AceSearcher/blob/main/inference/main_qa.py) lines 205–238 |

---

## Models

| Size | HuggingFace ID | Base | Tensor parallel |
|---|---|---|---|
| 1.5B | `AceSearcher/AceSearcher-1.5B` | Qwen2-1.5B | 1 GPU |
| 8B | `AceSearcher/AceSearcher-8B` | Llama-3.1-8B-Instruct | 1–2 GPUs |
| 14B | `AceSearcher/AceSearcher-14B` | Qwen2.5-14B | 2 GPUs |
| 32B | `AceSearcher/AceSearcher-32B` | Qwen2.5-32B | 4 GPUs |

---

## Supported Datasets

| Dataset | Official split (size) | Download |
|---|---|---|
| 2WikiMultiHopQA | dev, 12 576 q | `bash data/download.sh 2wikimultihopqa` |
| HotpotQA | fullwiki-dev, 7 405 q | `bash data/download.sh hotpotqa` |
| MuSiQue | answerable-dev, 2 417 q | `bash data/download.sh musique` |
| Bamboogle | test, 125 q | `bash data/download.sh bamboogle` |
| FEVER | v1.0 dev, 19 998 q | `bash data/download.sh fever` |
| HoverFEVER | dev | `bash data/download.sh hoverfever` |

---

## Retrieval Setup

- **Encoders**: `intfloat/e5-large` (2WikiMHQA, MuSiQue) / `intfloat/e5-large-v2` (HotpotQA) — 1024-dim
- **Query prefix**: `"query: <q>"`; corpus prefix: `"passage: <p>"`
- **Index**: 4 FAISS `IndexFlatIP` shards per dataset, stored under `embeddings/<dataset>/<embed_name>/`
- **Top-k**: 10 passages per sub-question

The corpus files (`corpus.tsv`) and shards are **not committed** to git (too large). Build them locally with:

```bash
bash methods/acesearcher/build_index.sh
```

This downloads each dataset's Wikipedia corpus from
`AceSearcher/evaluation_datasets` on HuggingFace and encodes it with the
appropriate E5 model. Approximate disk usage:

| Dataset | Passages | Disk |
|---|---|---|
| 2WikiMultiHopQA | 416 K | ~1.8 GB |
| HotpotQA | ~535 K | ~2.2 GB |
| MuSiQue | ~140 K | ~0.6 GB |

To build a single dataset (e.g., only 2WikiMHQA) or use a different GPU:

```bash
bash methods/acesearcher/build_index.sh --dataset 2wikimultihopqa --gpu 0
```

The index is stored at `methods/acesearcher/embeddings/` and resolved
automatically by `run.sh`. Override with `ACESEARCHER_INDEX_ROOT`.

---

## Installation

```bash
pip install -r methods/acesearcher/requirements.txt
# vllm==0.8.2, transformers==4.51.0, faiss-cpu, huggingface_hub
```

---

## Deployment (Step by Step)

```bash
# 1. Install dependencies
pip install -r methods/acesearcher/requirements.txt

# 2. Download question sets
bash data/download.sh 2wikimultihopqa
bash data/download.sh hotpotqa
bash data/download.sh musique

# 3. Build retrieval indexes (one time, ~30–90 min per dataset on 1 GPU)
bash methods/acesearcher/build_index.sh

# 4. Run inference
bash methods/acesearcher/run.sh \
    --dataset  2wikimultihopqa \
    --split    test \
    --output   results/acesearcher_14b/ \
    --gpus     0,1 \
    --size     14b
# Output: results/acesearcher_14b/predictions.jsonl + metrics.json
```

---

## Smoke Test

```bash
# After completing steps 1–3 above:
bash methods/acesearcher/test.sh
# Default: GPUs 0,1 for 14B model. Override: GPUS=0 bash methods/acesearcher/test.sh
# Runs 1 example per dataset; prints prediction for each.
```

---

## Verified Results (from paper / our runs)

| Dataset | Size | Acc | EM |
|---|---|---|---|
| 2WikiMHQA | 1.5B | 0.689 | — |
| 2WikiMHQA | 8B | 0.741 | — |
| 2WikiMHQA | 14B | 0.739 | — |
| 2WikiMHQA | 32B | 0.719 | — |
| HotpotQA | 14B | see paper | — |
| Bamboogle | 14B | see paper | — |

*Acc = normalized answer containment; EM = exact match after normalization.*

---

## Citation

```bibtex
@inproceedings{xu2025acesearcher,
  title     = {AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play},
  author    = {Ran Xu and Yuchen Zhuang and Zihan Dong and Ruiyu Wang and Yue Yu and Joyce C. Ho and Linjun Zhang and Haoyu Wang and Wenqi Shi and Carl Yang},
  booktitle = {the 39th Annual Conference on Neural Information Processing Systems},
  year      = {2025},
  url       = {https://openreview.net/forum?id=jSgCM0uZn3},
}
```
