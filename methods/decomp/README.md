# DecomP-ODQA

**Paper**: [Decomposed Prompting: A Modular Approach for Solving Complex Tasks](https://arxiv.org/abs/2210.02406)  
**Venue**: ICLR 2023  
**Authors**: Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, Ashish Sabharwal  
**Official Repository**: https://github.com/HarshTrivedi/DecomP-ODQA  
**Main DecomP Repo**: https://github.com/allenai/DecomP

---

## Method Overview

DecomP-ODQA decomposes complex multi-hop questions into atomic sub-questions, answers each sub-question separately with retrieved context, then composes the final answer:

1. **Decompose**: Given a complex question, use few-shot prompting to generate atomic sub-questions.
2. **Retrieve**: For each sub-question, retrieve top-k passages from the retrieval server.
3. **Answer sub-questions**: Answer each sub-question using few-shot prompting with retrieved passages.
4. **Compose**: Generate the final answer using all retrieved passages and sub-answers via chain-of-thought prompting.

### Prompt Source

Official few-shot prompts from [`HarshTrivedi/DecomP-ODQA/prompts/`](https://github.com/HarshTrivedi/DecomP-ODQA/tree/main/prompts):

| Prompt file | Purpose |
|---|---|
| `question_decomposition.txt` | Decompose complex question into atomic sub-questions |
| `singlehop_titleqa_with_all_gold_paras_and_1_distractor.txt` | Answer each sub-question given retrieved passages |
| `multihop_cot_titleqa_with_all_gold_paras_and_1_distractor.txt` | Compose final answer with CoT reasoning |

**Decomposition prompt format** (from `question_decomposition.txt`):
```
QC: <complex question>
QS: (select) [retrieve_odqa] <sub-question-1>
A: {"titles": [...], "answer": ["<sub-answer-1>"]}
QS: (select) [retrieve_odqa] <sub-question-2>
A: {"titles": [...], "answer": ["<sub-answer-2>"]}
QS: [EOQ]
```

**Singlehop QA prompt format** (from `singlehop_titleqa_*.txt`):
```
Wikipedia Title: <title>
<passage text>

Q: <sub-question>
A: ["<answer>"]
```

**Multihop CoT prompt format** (from `multihop_cot_titleqa_*.txt`):
```
Wikipedia Title: <title>
<passage text>

Q: <complex question>
<chain-of-thought reasoning>
A: <final answer>.
```

---

## Models

Original paper uses Codex (`code-davinci-002`).  
This implementation supports any instruction-tuned LLM via vLLM.

| Model | Notes |
|---|---|
| `Qwen/Qwen2.5-14B-Instruct` | Default in this repo |
| `Qwen/Qwen2.5-7B-Instruct` | Smaller option |

---

## Supported Datasets

| Dataset | Split | Download |
|---|---|---|
| HotpotQA (fullwiki) | dev | `bash data/download_hotpotqa.sh` |
| 2WikiMultiHopQA | dev | `bash data/download_2wikimhqa.sh` |
| MuSiQue | dev | `bash data/download_musique.sh` |

---

## Retrieval Setup

DecomP-ODQA uses **Elasticsearch BM25** (official).

### Starting the Elasticsearch retrieval server

```bash
# 1. Clone DecomP-ODQA
git clone https://github.com/HarshTrivedi/DecomP-ODQA /tmp/DecomP-ODQA
cd /tmp/DecomP-ODQA && pip install -r requirements.txt

# 2. Start Elasticsearch 7.10
#    https://www.elastic.co/downloads/past-releases/elasticsearch-7-10-2

# 3. Build the index (one-time per dataset)
python retriever_server/build_index.py \
    --corpus_path <wiki_corpus.jsonl> \
    --index_name  hotpotqa_index

# 4. Start the retrieval server
python retriever_server/serve.py \
    --index_name hotpotqa_index \
    --port 9200
```

**API format**:
```
POST /retrieve
Body: {"queries": ["..."], "topk": 5, "return_scores": false}
Response: {"result": [[{"document": {"contents": "title\nbody"}, ...}]]}
```

---

## Official Prompts Setup

```bash
# Clone DecomP-ODQA to get the official few-shot prompts
git clone https://github.com/HarshTrivedi/DecomP-ODQA /tmp/DecomP-ODQA
# Pass --prompts-dir to run.sh:
# --prompts-dir /tmp/DecomP-ODQA/prompts
```

Prompts are auto-discovered if cloned to `/data2/liang/data3/code/DecomP-ODQA` or placed in `methods/decomp/prompts/`.

---

## Installation

```bash
pip install -r methods/decomp/requirements.txt
# vllm>=0.4.0, transformers>=4.40.0, torch>=2.0, requests
```

---

## Quick Start

```bash
# 1. Clone DecomP-ODQA to get official prompts
git clone https://github.com/HarshTrivedi/DecomP-ODQA /tmp/DecomP-ODQA

# 2. Start retrieval server (see Retrieval Setup above)

# 3. Download dataset
bash data/download_hotpotqa.sh

# 4. Run DecomP-ODQA on HotpotQA
bash methods/decomp/run.sh \
    --dataset      hotpotqa \
    --split        validation \
    --output       results/decomp/ \
    --retrieval-url http://127.0.0.1:9200/retrieve \
    --prompts-dir  /tmp/DecomP-ODQA/prompts \
    --model        Qwen/Qwen2.5-14B-Instruct \
    --gpus         0,1
```

---

## Smoke Test

```bash
DECOMP_RETRIEVAL_URL=http://127.0.0.1:9200/retrieve \
DECOMP_PROMPTS_DIR=/tmp/DecomP-ODQA/prompts \
bash methods/decomp/test.sh
```

---

## Verified Results (from paper)

Results with Codex (`code-davinci-002`), `decomp_context` variant:

| Dataset | EM (DecomP) | EM (No Decomp) |
|---|---|---|
| HotpotQA | 62.4 | 56.4 |
| 2WikiMHQA | 68.5 | 44.8 |
| MuSiQue | 35.7 | 20.5 |

Source: Table 1 in [Khot et al., ICLR 2023](https://arxiv.org/abs/2210.02406).

---

## Citation

```bibtex
@inproceedings{khot2023decomposed,
    title     = {Decomposed Prompting: A Modular Approach for Solving Complex Tasks},
    author    = {Khot, Tushar and Trivedi, Harsh and Finlayson, Matthew and Fu, Yao and Richardson, Kyle and Clark, Peter and Sabharwal, Ashish},
    booktitle = {The Eleventh International Conference on Learning Representations},
    year      = {2023},
    url       = {https://arxiv.org/abs/2210.02406},
}
```
