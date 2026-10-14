# DQG / FT-PANEL

**Paper**: [Generating Complex Question Decompositions in the Face of Distribution Shifts](https://aclanthology.org/2025.naacl-long.432)  
**Venue**: NAACL 2025  
**Authors**: Han Liang, Claire Gardent  
**Official Repository**: https://github.com/hankelvin/complex_question_decomposition

---

## Method Overview

DQG (Decomposition Question Generation) uses a **fine-tuned LoRA adapter (FT-PANEL)** to decompose complex multi-hop questions into ordered atomic sub-questions, then answers each sub-question with retrieval:

1. **Decompose** (Stage 1): FT-PANEL LoRA on Qwen2.5-7B-Instruct generates sub-questions with `[SQ1]...[SQN]` markers.
2. **Resolve references**: `#N` placeholders in later sub-questions are substituted with answers to earlier sub-questions.
3. **Retrieve** (Stage 2): For each sub-question, retrieve top-k passages.
4. **Answer**: Answer each sub-question with the retrieved passages using a reader LLM.
5. **Compose**: Compose the final answer from all sub-question answers.

### Prompt Source

Official prompt-v2 zero-shot instruction from [`hankelvin/complex_question_decomposition`](https://github.com/hankelvin/complex_question_decomposition):

```
I will provide you with one complex multi-hop question. Decompose it into a set of simpler
sub-questions so that it will be easier to identify and retrieve the information for answering
them instead of the complex question. It is important that the sub-questions (i) are not
ambiguously worded, (ii) are logically connected, and (iii) that they cover all the necessary
information and steps to answer the original complex question. ...

It is very important that you prefix the start of each sub-question with a "[SQX]" marker
that indicates the position number (X) of the sub-question, i.e. "[SQ1] ... [SQ2] ... [SQ3] ..."
```

**Output format**:
```
[SQ1] What is the nationality of the director of <film>?
[SQ2] What is the capital of the country identified in #1?
```

---

## Models

| Model | Role | Notes |
|---|---|---|
| FT-PANEL LoRA on `Qwen/Qwen2.5-7B-Instruct` | Decomposer | Fine-tuned on PANEL synthetic data |
| FT-PANEL LoRA on `meta-llama/Llama-3.1-8B-Instruct` | Decomposer (alt) | Also available |
| `Qwen/Qwen2.5-14B-Instruct` | Reader | Default in this repo |

**FT-PANEL** = Fine-Tuned on PANEL synthetic decomposition data.

---

## Supported Datasets

| Dataset | Official split (size) | Download |
|---|---|---|
| 2WikiMultiHopQA | dev, 12 576 q | `bash data/download.sh 2wikimultihopqa` |
| HotpotQA | fullwiki-dev, 7 405 q | `bash data/download.sh hotpotqa` |
| MuSiQue | answerable-dev, 2 417 q | `bash data/download.sh musique` |

---

## FT-PANEL Adapter Setup

Download the LoRA adapter from the official repository:

```bash
# Option 1: Download from official repo (Google Drive link in their README)
git clone https://github.com/hankelvin/complex_question_decomposition
# Follow README instructions to download models.zip

# Option 2: Use local path (if already available)
export DQG_ADAPTER_PATH=/path/to/ft_panel_lora/test_model_lora.ckpt
```

Local adapters available at:
```
/data2/liang/data3/code/baseline/models/results/dqg_dqa/dqg_breakhigh/
  dqg_breakhigh_qwen_prmptv2_...accelerate/test_model_lora.ckpt   # Qwen2.5-7B
  dqg_breakhigh_llama_prmptv2_...accelerate/test_model_lora.ckpt  # LLaMA-3.1-8B
```

---

## Retrieval Setup

Requires any server matching the standard POST `/retrieve` API.

**Recommended**: Use the same retrieval server as SearchR1:
```bash
bash methods/searchr1/start_retrieval.sh \
    --index-path  /path/to/e5_HNSW64.index \
    --corpus-path /path/to/wiki-18.jsonl \
    --port 8000
```

---

## Installation

```bash
pip install -r methods/dqg/requirements.txt
# vllm>=0.4.0, transformers>=4.40.0, peft>=0.10.0, torch>=2.0, requests
```

---

## Quick Start

```bash
# 1. Start retrieval server (see above)

# 2. Download dataset
bash data/download.sh hotpotqa

# 3. Run DQG/FT-PANEL on HotpotQA
bash methods/dqg/run.sh \
    --dataset      hotpotqa \
    --split        test \
    --output       results/dqg/ \
    --adapter      /path/to/ft_panel_lora/test_model_lora.ckpt \
    --retrieval-url http://127.0.0.1:8000/retrieve \
    --reader-model Qwen/Qwen2.5-14B-Instruct \
    --gpus         0,1
```

---

## Smoke Test

```bash
DQG_RETRIEVAL_URL=http://127.0.0.1:8000/retrieve \
DQG_ADAPTER_PATH=/path/to/ft_panel_lora/test_model_lora.ckpt \
bash methods/dqg/test.sh
```

---

## Verified Results (from paper)

| Dataset | Model | EM | F1 |
|---|---|---|---|
| 2WikiMHQA | FT-PANEL Qwen2.5-7B | see paper | see paper |
| HotpotQA | FT-PANEL Qwen2.5-7B | see paper | see paper |
| MuSiQue | FT-PANEL Qwen2.5-7B | see paper | see paper |

See [NAACL 2025 paper](https://aclanthology.org/2025.naacl-long.432) for full results.

---

## Citation

```bibtex
@inproceedings{han2025dqg,
    title     = {Generating Complex Question Decompositions in the Face of Distribution Shifts},
    author    = {Han, Liang and Gardent, Claire},
    booktitle = {Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
    year      = {2025},
    url       = {https://aclanthology.org/2025.naacl-long.432},
}
```
