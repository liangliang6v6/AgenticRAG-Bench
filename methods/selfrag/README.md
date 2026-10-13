# Self-RAG

**Paper**: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)  
**Venue**: ICLR 2024 (oral)  
**Authors**: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi  
**Official Repository**: https://github.com/AkariAsai/self-rag

---

## Method Overview

Self-RAG trains a language model with **reflection tokens** embedded in the vocabulary. At inference time:

1. **Retrieve**: Retrieve top-k passages for the question.
2. **Generate with each passage**: Build prompt `[Retrieval]<paragraph>{passage}</paragraph>` and generate.
3. **Score with reflection tokens**: Use logprob distributions over special reflection tokens at specific positions to compute a composite score.
4. **Select best**: Pick the highest-scoring generation.

### Reflection Token Scoring

| Token group | Meaning | Position in output |
|---|---|---|
| `[Relevant]` / `[Irrelevant]` | Passage relevance | First token |
| `[Fully supported]` / `[Partially supported]` / `[No support / Contradictory]` | Factual grounding | At the support position |
| `[Utility:1..5]` | Answer utility | At the utility position |

**Score formula** (from official repo):
```
total = exp(mean_logprob) + w_rel * P([Relevant]) + w_sup * (P([Fully]) + 0.5 * P([Partially])) + w_use * weighted_utility
```
Default weights: `w_rel=1.0, w_sup=1.0, w_use=0.5`

### Prompt Source

Official prompt format (verbatim from [`AkariAsai/self-rag/retrieval_lm/run_short_form.py`](https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py)):

```
### Instruction:
{question}

### Response:
[Retrieval]<paragraph>{passage}</paragraph>
```

---

## Models

| Model | Notes |
|---|---|
| `selfrag/selfrag_llama2_7b` | Default — official 7B model |
| `selfrag/selfrag_llama2_13b` | Official 13B model |

Both models have Self-RAG reflection tokens in their vocabulary and require **no chat template** — they use the `### Instruction:` / `### Response:` format directly.

---

## Supported Datasets

| Dataset | Split | Download |
|---|---|---|
| TriviaQA | validation | `bash data/download_triviaqa.sh` |
| PopQA | test | `bash data/download_popqa.sh` |
| FEVER | dev | `bash data/download_fever.sh` |
| HotpotQA | dev | `bash data/download_hotpotqa.sh` |

---

## Retrieval Setup

Self-RAG uses any dense retrieval server matching the standard API.

**Recommended**: Use the SearchR1 retrieval server:
```bash
bash methods/searchr1/start_retrieval.sh \
    --index-path  /path/to/e5_HNSW64.index \
    --corpus-path /path/to/wiki-18.jsonl \
    --port 8000
```

**API format**:
```
POST /retrieve
Body: {"queries": ["..."], "topk": 10, "return_scores": false}
Response: {"result": [[{"document": {"contents": "..."}, ...}]]}
```

---

## Installation

```bash
pip install -r methods/selfrag/requirements.txt
# vllm>=0.4.0, transformers>=4.40.0, torch>=2.0, requests
```

---

## Quick Start

```bash
# 1. Download Self-RAG model (from HuggingFace)
#    selfrag/selfrag_llama2_7b

# 2. Start retrieval server (see above)

# 3. Download dataset
bash data/download_triviaqa.sh

# 4. Run Self-RAG on TriviaQA
bash methods/selfrag/run.sh \
    --dataset      triviaqa \
    --split        validation \
    --output       results/selfrag/ \
    --retrieval-url http://127.0.0.1:8000/retrieve \
    --model        selfrag/selfrag_llama2_7b \
    --gpus         0
```

---

## Smoke Test

```bash
SELFRAG_RETRIEVAL_URL=http://127.0.0.1:8000/retrieve \
bash methods/selfrag/test.sh
```

---

## Verified Results (from paper)

| Dataset | Model | EM / Accuracy |
|---|---|---|
| TriviaQA | selfrag_llama2_7b | 66.4 |
| PopQA | selfrag_llama2_7b | 57.2 |
| TriviaQA | selfrag_llama2_13b | 69.3 |
| PopQA | selfrag_llama2_13b | 61.2 |

Source: Table 1 in [Asai et al., ICLR 2024](https://arxiv.org/abs/2310.11511).

---

## Citation

```bibtex
@inproceedings{asai2024selfrag,
    title     = {{Self-RAG}: Learning to Retrieve, Generate, and Critique through Self-Reflection},
    author    = {Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
    booktitle = {The Twelfth International Conference on Learning Representations},
    year      = {2024},
    url       = {https://arxiv.org/abs/2310.11511},
}
```
