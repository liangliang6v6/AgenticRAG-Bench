# AgenticRAG-Bench

**A collection of reproducible implementations of Agentic Retrieval-Augmented Generation (RAG) methods.**

Agentic RAG methods are often difficult to reproduce because they rely on different model checkpoints, retrieval systems, datasets, prompts, and inference settings. **AgenticRAG-Bench** brings together representative methods in one repository, making it easier to reproduce existing baselines, evaluate their performance, and build upon prior work.

The repository currently includes **8 methods** covering reinforcement learning, iterative retrieval, question decomposition, tree-based reasoning, self-reflection, and retrieval planning.

**Key features:**
- **Multiple baselines:** Representative Agentic RAG methods with individual implementations and setup instructions.
- **Reproduction-oriented:** Model checkpoints, prompts, retrieval configurations, and dependencies documented for each method.
- **Dataset support:** Common multi-hop QA and knowledge-intensive benchmarks, including HotpotQA, 2WikiMultiHopQA, and MuSiQue.
- **Shared evaluation:** A common interface for computing answer-level Exact Match (EM), F1, and accuracy.
- **Extensible structure:** Independent method directories for integrating additional baselines.

> **Note:** This repository focuses on reproducing individual methods under their respective configurations. It is not yet a fully controlled comparison framework. Some implementations adapt the original models or retrieval environments; these differences are documented in the method-specific READMEs.

---

## Supported Methods

| Method | Paper | Venue | Approach | Models |
|---|---|---|---|---|
| [AceSearcher](methods/acesearcher/) | [Bootstrapping Reasoning and Search via Reinforced Self-Play](https://arxiv.org/abs/2509.24193) | NeurIPS 2025 Spotlight | RL-based search | 1.5B, 8B, 14B, 32B |
| [Search-R1](methods/searchr1/) | [Training LLMs to Reason and Search with Reinforcement Learning](https://arxiv.org/abs/2503.09516) | 2025 | RL-based search | Qwen2.5 3B/7B |
| [IRCoT](methods/ircot/) | [Interleaving Retrieval with Chain-of-Thought Reasoning](https://aclanthology.org/2023.acl-long.557/) | ACL 2023 | Iterative retrieval | Configurable LLM |
| [DecomP-ODQA](methods/decomp/) | [Decomposed Prompting: A Modular Approach](https://arxiv.org/abs/2210.02406) | ICLR 2023 | Question decomposition | Configurable LLM |
| [RT-RAG](methods/rtrag/) | [Reasoning in Trees: Improving RAG for Multi-Hop QA](https://arxiv.org/abs/2601.11255) | 2026 | Tree-based reasoning | Configurable LLM |
| [DQG / FT-PANEL](methods/dqg/) | [Generating Complex Question Decompositions in the Face of Distribution Shifts](https://aclanthology.org/2025.naacl-long.55/) | NAACL 2025 | Supervised decomposition | Qwen2.5-7B + LoRA |
| [Self-RAG](methods/selfrag/) | [Learning to Retrieve, Generate, and Critique](https://openreview.net/forum?id=hSyW5go0v8) | ICLR 2024 | Self-reflective retrieval | Llama2 7B/13B |
| [PlanRAG](methods/planrag/) | [A Plan-then-Retrieval Augmented Generation](https://aclanthology.org/2024.naacl-long.364/) | NAACL 2024 | Planning and retrieval | OpenAI-compatible LLM |

Each method has its own directory containing implementation details, installation requirements, retrieval setup, and inference scripts.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/liangliang6v6/AgenticRAG-Bench.git
cd AgenticRAG-Bench
```

### 2. Install dependencies

Each method maintains its own dependencies to accommodate different models and retrieval frameworks.

For example, to run AceSearcher:

```bash
python -m venv .venv-acesearcher
source .venv-acesearcher/bin/activate

pip install datasets huggingface_hub
pip install -r methods/acesearcher/requirements.txt
```

For other methods, install the corresponding `methods/<name>/requirements.txt`.

Most local-model implementations require NVIDIA GPUs and a compatible CUDA environment. Hardware requirements depend on the selected model and retrieval configuration.

### 3. Download datasets

Use the unified dataset download script:

```bash
# Download selected datasets
bash data/download.sh hotpotqa 2wikimultihopqa musique

# Download all supported datasets
bash data/download.sh
```

Datasets are saved under:

```text
data/
├── hotpotqa/test.jsonl
├── 2wikimultihopqa/test.jsonl
├── musique/test.jsonl
└── ...
```

The local `test.jsonl` filenames are used for evaluation inputs and do not necessarily correspond to the original datasets' held-out test splits. For example, HotpotQA and 2WikiMultiHopQA use publicly available development examples.

See [Dataset Documentation](data/README.md) for dataset sources and preprocessing details.

### 4. Prepare retrieval resources

Different methods require different retrieval configurations. Evaluation questions alone are insufficient to run retrieval-based methods.

| Method | Retrieval Setup |
|---|---|
| AceSearcher | E5 embeddings + local FAISS index |
| Search-R1 | Wikipedia corpus + dense retrieval server |
| IRCoT | BM25 / compatible retrieval server |
| DecomP-ODQA | BM25 / compatible retrieval server |
| RT-RAG | Dense retrieval server |
| DQG / FT-PANEL | Compatible retrieval server |
| Self-RAG | Compatible retrieval server |
| PlanRAG | Neo4j / MySQL database |

For example, build the AceSearcher retrieval index:

```bash
bash methods/acesearcher/build_index.sh \
    --dataset hotpotqa \
    --gpu 0
```

This prepares the passage corpus and embeddings required by AceSearcher.

For methods that use a retrieval server, follow the corresponding method README to build the index and start the service before inference.

### 5. Run a baseline

**Example 1: AceSearcher-8B on HotpotQA**

```bash
bash methods/acesearcher/run.sh \
    --dataset hotpotqa \
    --split test \
    --size 8b \
    --gpus 0 \
    --output results/acesearcher_hotpotqa
```

**Example 2: Search-R1-7B on HotpotQA**

After starting the retrieval server:

```bash
bash methods/searchr1/run.sh \
    --dataset hotpotqa \
    --split test \
    --size 7b_base \
    --gpus 0 \
    --retrieval-url http://127.0.0.1:8000/retrieve \
    --output results/searchr1_hotpotqa
```

**Example 3: IRCoT on HotpotQA**

After preparing a compatible retrieval service:

```bash
bash methods/ircot/run.sh \
    --dataset hotpotqa \
    --split test \
    --model Qwen/Qwen2.5-14B-Instruct \
    --retrieval-url http://127.0.0.1:8000/retrieve \
    --gpus 0,1 \
    --output results/ircot
```

The IRCoT implementation also requires its dataset-specific few-shot prompts. See [IRCoT Setup](methods/ircot/README.md).

Additional commands and configuration options are provided in each method's README.

### 6. Smoke test

Each method includes a `test.sh` script for checking its inference pipeline on a small number of examples.

```bash
bash methods/acesearcher/test.sh
bash methods/searchr1/test.sh
bash methods/ircot/test.sh
```

Run these tests after downloading the required datasets, checkpoints, and retrieval resources.

Some smoke-test scripts use specific default GPU IDs or input paths. Check the script or method README if the default configuration does not match your environment.

### 7. Evaluate predictions

The repository provides a shared evaluation interface:

```bash
python eval/eval.py \
    --predictions results/acesearcher_hotpotqa/predictions.jsonl
```

The input is a JSONL file containing at least the following fields:

```json
{
  "id": "example_id",
  "question": "What is the capital of France?",
  "prediction": "Paris",
  "answer": "Paris"
}
```

The evaluator computes:

| Metric | Description |
|---|---|
| EM | Exact Match after answer normalization |
| F1 | Token-level precision and recall |
| Acc | Normalized substring-based answer matching |

Example output:

```json
{
  "n": 1,
  "em": 1.0,
  "f1": 1.0,
  "acc": 1.0
}
```

Results are also saved as `metrics.json`.

**Note:** The repository's current accuracy implementation considers substring matches in either direction. It may differ from the evaluation protocols used by the original papers. Consult [eval/metrics.py](eval/metrics.py) before comparing reported scores.

---

## Dataset Coverage

The following table summarizes the datasets supported by the current method implementations.

| Dataset | AceSearcher | Search-R1 | IRCoT | DecomP | RT-RAG | DQG | Self-RAG | PlanRAG |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2WikiMultiHopQA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| HotpotQA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| MuSiQue | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| Natural Questions | — | ✓ | — | — | — | — | — | — |
| TriviaQA | — | ✓ | — | — | — | — | ✓ | — |
| PopQA | — | ✓ | — | — | — | — | ✓ | — |
| Bamboogle | ✓ | — | — | — | — | — | — | — |
| FEVER | ✓ | — | — | — | — | — | ✓ | — |
| HoverFEVER | ✓ | — | — | — | — | — | — | — |
| IIRC | — | — | ✓ | — | — | — | — | — |
| DQA (LocatE / ControlE) | — | — | — | — | — | — | — | ✓ |

A checkmark indicates that a method has an implementation path for the dataset. It does not guarantee that every combination has been independently verified or reproduces the paper's reported results.

PlanRAG is evaluated on structured database decision-making tasks rather than conventional open-domain text QA.

---

## Repository Structure

```text
AgenticRAG-Bench/
│
├── README.md
│
├── data/
│   ├── README.md
│   └── download.sh
│
├── eval/
│   ├── eval.py
│   └── metrics.py
│
└── methods/
    ├── acesearcher/
    ├── searchr1/
    ├── ircot/
    ├── decomp/
    ├── rtrag/
    ├── dqg/
    ├── selfrag/
    └── planrag/
```

Each method directory includes:

- `README.md`: Method overview, paper reference, models, datasets, retrieval setup, and usage.
- `requirements.txt`: Method-specific dependencies.
- `run.sh`: Main inference launcher.
- `test.sh`: Small-scale inference test.
- `src/`: Implementation and inference code.

Some methods additionally include retrieval-server scripts, indexing utilities, or prompt templates.

Model checkpoints, retrieval corpora, and large indexes are not included in the repository.

---

## Reproducibility Notes

Agentic RAG methods differ in multiple aspects that can substantially affect their performance.

**Model backbones.** Some methods use pretrained or RL-trained model checkpoints, while others use prompted general-purpose LLMs. Replacing the backbone can change both reasoning behavior and retrieval decisions.

**Retrieval environments.** Methods may rely on different Wikipedia corpora, embedding models, retrieval indexes, and passage-ranking strategies.

**Inference configurations.** The number of retrieval calls, retrieved passages, reasoning steps, and generated tokens can vary across methods.

**Evaluation protocols.** Dataset subsets, answer normalization, and metric implementations may differ from those reported in the original papers.

For these reasons, results from different methods should not be interpreted as controlled comparisons without accounting for these factors.

Some implementations also intentionally adapt the original method. For example, the RT-RAG implementation uses a configurable local LLM and a single reasoning tree rather than the original multi-tree consensus procedure.

The repository is intended to make these implementations accessible while preserving visibility into their experimental configurations.

### Known Limitations

- Some method-specific documentation still refers to older dataset download scripts or split names.
- Several methods require substantial retrieval-index preparation before inference.
- Model dependencies and hardware requirements vary, so a single Python environment may not support every baseline.
- Some configurations are adaptations rather than exact reproductions of the original paper.
- Published results and independently reproduced results are not yet maintained in a unified, fully validated results table.

Contributions addressing these limitations are welcome.

---

## Adding New Methods

We welcome contributions that improve the coverage, reproducibility, and usability of AgenticRAG-Bench.

To add a new baseline:

1. Create a directory under `methods/<name>/`.
2. Include `README.md`, `requirements.txt`, `run.sh`, `test.sh`, and implementation code under `src/`.
3. Document the original paper, official repository, model checkpoints, datasets, retrieval configuration, and any implementation differences.
4. Follow the existing inference conventions where applicable.
5. Export predictions in JSONL format with `id`, `question`, `prediction`, and `answer`.
6. Update the Supported Methods and Dataset Coverage tables.

Bug fixes, dependency updates, corrected configurations, and verified reproduction results are also welcome.

Please open an issue or pull request for contributions.


## License

A repository-level license has not yet been specified. Please consult the original projects' licenses and terms before reusing or redistributing adapted code, model checkpoints, prompts, or datasets.
