# IRCoT

**Paper**: [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://aclanthology.org/2023.acl-long.557)  
**Venue**: ACL 2023  
**Authors**: Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal  
**Official Repository**: https://github.com/StonyBrookNLP/ircot

---

## Method Overview

IRCoT interleaves retrieval with chain-of-thought (CoT) reasoning for multi-hop QA:

1. **Initial retrieval**: retrieve top-k passages for the original question.
2. **CoT generation**: prompt an LLM with few-shot CoT examples + retrieved passages; generate the next reasoning sentence.
3. **Query formation**: use the last generated sentence as the next retrieval query.
4. **Iterate**: retrieve new passages, de-duplicate, and continue.
5. **Termination**: stop when the model generates `"So the answer is:"` or `max_steps` is reached.
6. **Answer extraction**: parse the final answer from the `"So the answer is: X."` pattern.

### Prompt Source

Official few-shot CoT prompts are in the IRCoT repository under `prompts/<dataset>/`:

```
prompts/
├── hotpotqa/
│   └── gold_with_2_distractors_context_cot_qa_codex.txt
├── 2wikimultihopqa/
│   └── gold_with_2_distractors_context_cot_qa_codex.txt
├── musique/
│   └── gold_with_2_distractors_context_cot_qa_codex.txt
└── iirc/
    └── gold_with_2_distractors_context_cot_qa_codex.txt
```

Source: [StonyBrookNLP/ircot/prompts/](https://github.com/StonyBrookNLP/ircot/tree/main/prompts)

These prompts contain ~20 few-shot CoT examples with interleaved retrieval context.

---

## Models

Original paper uses Codex (`code-davinci-002`) and Flan-T5-xl/xxl.  
This implementation supports any instruction-tuned LLM via vLLM.

| Model | Notes |
|---|---|
| `Qwen/Qwen2.5-14B-Instruct` | Default in this repo (strong open-source substitute) |
| `Qwen/Qwen2.5-7B-Instruct` | Smaller option |
| `google/flan-t5-xxl` | Original paper model (requires seq2seq adaptation) |

---

## Supported Datasets

| Dataset | Split | Download |
|---|---|---|
| HotpotQA (fullwiki) | dev | `bash data/download_hotpotqa.sh` |
| 2WikiMultiHopQA | dev | `bash data/download_2wikimhqa.sh` |
| MuSiQue | dev | `bash data/download_musique.sh` |
| IIRC | dev | `bash data/download_iirc.sh` |

---

## Retrieval Setup

IRCoT was designed with **Elasticsearch BM25** as the retriever.

### Starting the official Elasticsearch retrieval server

```bash
# 1. Clone IRCoT and install deps
git clone https://github.com/StonyBrookNLP/ircot /tmp/ircot
cd /tmp/ircot && pip install -r requirements.txt

# 2. Start Elasticsearch 7.10 (one-time)
#    https://www.elastic.co/downloads/past-releases/elasticsearch-7-10-2

# 3. Index the Wikipedia corpus (one-time per dataset)
python retriever_server/build_index.py \
    --corpus_path <wiki_corpus.jsonl> \
    --index_name  hotpotqa_index

# 4. Start the retrieval server
python retriever_server/serve.py \
    --index_name hotpotqa_index \
    --port 9200
```

**API format** (compatible with `--retrieval-url`):
```
POST /retrieve
Body: {"queries": ["..."], "topk": 6, "return_scores": false}
Response: {"result": [[{"document": {"contents": "title\nbody"}, ...}]]}
```

Any server matching this API can substitute Elasticsearch.

---

## Official Prompts Setup

```bash
# Clone IRCoT to get the official few-shot prompts
git clone https://github.com/StonyBrookNLP/ircot /tmp/ircot
# Pass --prompts-dir to run.sh:
# --prompts-dir /tmp/ircot/prompts
```

The prompts are auto-discovered if cloned to `/data2/liang/data3/code/ircot` or placed in `methods/ircot/prompts/`.

---

## Installation

```bash
pip install -r methods/ircot/requirements.txt
# vllm>=0.4.0, transformers>=4.40.0, torch>=2.0, requests
```

---

## Quick Start

```bash
# 1. Clone IRCoT to get official prompts
git clone https://github.com/StonyBrookNLP/ircot /tmp/ircot

# 2. Start retrieval server (see Retrieval Setup above)

# 3. Download dataset
bash data/download_hotpotqa.sh

# 4. Run IRCoT on HotpotQA
bash methods/ircot/run.sh \
    --dataset      hotpotqa \
    --split        validation \
    --output       results/ircot/ \
    --retrieval-url http://127.0.0.1:9200 \
    --prompts-dir  /tmp/ircot/prompts \
    --model        Qwen/Qwen2.5-14B-Instruct \
    --gpus         0,1
```

---

## Smoke Test

```bash
IRCOT_RETRIEVAL_URL=http://127.0.0.1:9200 \
IRCOT_PROMPTS_DIR=/tmp/ircot/prompts \
bash methods/ircot/test.sh
```

---

## Verified Results (from paper)

Results with Codex (`code-davinci-002`), fullwiki setting:

| Dataset | EM (IRCoT) | EM (No Retrieval CoT) |
|---|---|---|
| HotpotQA | 59.9 | 48.0 |
| 2WikiMHQA | 50.5 | 27.2 |
| MuSiQue | 23.7 | 10.5 |
| IIRC | 38.1 | 30.5 |

Source: Table 1 in [Trivedi et al., ACL 2023](https://aclanthology.org/2023.acl-long.557).

---

## Citation

```bibtex
@inproceedings{trivedi2023interleaving,
    title     = {Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions},
    author    = {Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
    booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    year      = {2023},
    publisher = {Association for Computational Linguistics},
    url       = {https://aclanthology.org/2023.acl-long.557},
}
```
