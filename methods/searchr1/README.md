# SearchR1

**Paper**: [Search-R1: Training LLMs to Reason and Search with Reinforcement Learning](https://arxiv.org/abs/2503.09516) / [Search-R1 v2](https://arxiv.org/abs/2505.15117)  
**Venue**: arxiv 2025  
**Authors**: Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Heng Ji, Jiawei Han  
**Official Repository**: https://github.com/PeterGriffinJin/Search-R1

---

## Method Overview

SearchR1 trains LLMs to **interleave reasoning and search** via reinforcement learning (PPO/GRPO):

1. The model generates reasoning steps inside `<think>...</think>` tags.
2. When it needs external knowledge, it emits `<search>query</search>`.
3. The runtime retrieves results and injects them as `<information>...</information>`.
4. This repeats until the model emits `<answer>...</answer>` with the final answer.

The inference script here adapts the official `infer_dataset.py` to accept a unified JSONL input format.

### Prompt Source

Official prompt (verbatim from [`infer_dataset.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/infer_dataset.py#L113-L126)):

```
Answer the given question. You must conduct reasoning inside <think> and </think> first every
time you get new information. After reasoning, if you find you lack some knowledge, you can
call a search engine by <search> query </search> and it will return the top searched results
between <information> and </information>. You can search as many times as your want. If you
find no further external knowledge needed, you can directly provide the answer inside <answer>
and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
Question: {question}
```

---

## Models

| Model ID | Base | Notes |
|---|---|---|
| `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo` | Qwen2.5-3B | EM reward, PPO |
| `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-ppo` | Qwen2.5-3B-Instruct | EM reward, PPO |
| `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo` | Qwen2.5-7B | EM reward, PPO **(default)** |
| `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-ppo` | Qwen2.5-7B-Instruct | EM reward, PPO |

Full model collection: https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5

---

## Supported Datasets

| Dataset | Split | Download |
|---|---|---|
| HotpotQA | fullwiki-dev | `bash data/download_hotpotqa.sh` |
| 2WikiMultiHopQA | dev | `bash data/download_2wikimhqa.sh` |
| MuSiQue | dev | `bash data/download_musique.sh` |
| NQ | dev | `bash data/download_nq.sh` |
| TriviaQA | dev | `bash data/download_triviaqa.sh` |
| PopQA | test | `bash data/download_popqa.sh` |

---

## Retrieval Setup

SearchR1 uses a **FastAPI retrieval server** that the inference code calls at runtime.

- **Default encoder**: `intfloat/e5-base-v2`
- **Default index**: HNSW64 FAISS (built from wiki-18 corpus)
- **API endpoint**: `POST /retrieve` → `{"queries": [...], "topk": 3, "return_scores": true}`

### Starting the retrieval server

```bash
# 1. Build the corpus + index (one-time setup)
git clone https://github.com/PeterGriffinJin/Search-R1 /tmp/SearchR1
cd /tmp/SearchR1
bash download.sh                   # downloads wiki-18.jsonl
python search_r1/search/build_index.py \
    --corpus_path wiki-18.jsonl \
    --save_path   /path/to/index \
    --retriever_model intfloat/e5-base-v2

# 2. Start the server
bash methods/searchr1/start_retrieval.sh \
    --index-path  /path/to/index/e5_HNSW64.index \
    --corpus-path /path/to/wiki-18.jsonl \
    --port 8000
```

---

## Installation

```bash
pip install -r methods/searchr1/requirements.txt
# transformers>=4.40,<4.48, torch>=2.0, requests, datasets
```

---

## Quick Start

```bash
# 1. Start retrieval server (see above)

# 2. Download dataset
bash data/download_hotpotqa.sh

# 3. Run SearchR1-7B on HotpotQA
bash methods/searchr1/run.sh \
    --dataset      hotpotqa \
    --split        validation \
    --output       results/searchr1/ \
    --retrieval-url http://127.0.0.1:8000/retrieve \
    --gpus         0 \
    --size         7b_base
```

---

## Smoke Test

```bash
SEARCHR1_RETRIEVAL_URL=http://127.0.0.1:8000/retrieve \
bash methods/searchr1/test.sh
```

---

## Verified Results (from paper / our runs)

| Dataset | Model | EM | F1 |
|---|---|---|---|
| 2WikiMHQA | SearchR1-3B-base | 0.318 | — |
| 2WikiMHQA | SearchR1-7B-base | 0.709 | — |
| HotpotQA | SearchR1-7B-base | see paper | — |
| NQ | SearchR1-7B-base | see paper | — |

---

## Citation

```bibtex
@article{jin2025searchr1,
  title   = {Search-R1: Training LLMs to Reason and Search with Reinforcement Learning},
  author  = {Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Wang, Dong and Ji, Heng and Han, Jiawei},
  journal = {arXiv preprint arXiv:2503.09516},
  year    = {2025},
}
```
