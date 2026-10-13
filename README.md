# AgenticRAG-Bench

A collection of reproducible **Agentic RAG** baselines, each following its official implementation with pinned model IDs, official prompt templates, and verified results.

> **Design principle**: Every method lives in its own folder with its own retrieval setup, datasets, and dependencies — matching the original paper exactly. This is not a controlled comparison suite; it is a one-stop deployment reference.

---

## Methods

| Method | Paper | Venue | Type | Models | Datasets |
|---|---|---|---|---|---|
| [AceSearcher](methods/acesearcher/) | [Bootstrapping Reasoning and Search via Reinforced Self-Play](https://arxiv.org/abs/2509.24193) | NeurIPS 2025 Spotlight | RL-trained | `AceSearcher/AceSearcher-{1.5B,8B,14B,32B}` | 2WikiMHQA, HotpotQA, MuSiQue, Bamboogle, FEVER |
| [SearchR1](methods/searchr1/) | [Search-R1: Train LLMs to Reason and Search](https://arxiv.org/abs/2503.09516) | arxiv 2025 | RL PPO | `PeterJinGo/SearchR1-qwen2.5-{3b,7b}-*-ppo` | NQ, HotpotQA, TriviaQA, 2WikiMHQA, MuSiQue |
| [IRCoT](methods/ircot/) | [Interleaving Retrieval with Chain-of-Thought](https://aclanthology.org/2023.acl-long.557/) | ACL 2023 | Prompting | Codex / Flan-T5 / LLM-adaptable | HotpotQA, 2WikiMHQA, MuSiQue, IIRC |
| [DecomP-ODQA](methods/decomp/) | [Decomposed Prompting: A Modular Approach](https://openreview.net/forum?id=_nGgzQjzaRy) | ICLR 2023 | Prompting | Codex / LLM-adaptable | HotpotQA, 2WikiMHQA, MuSiQue |
| [RT-RAG](methods/rtrag/) | [Reasoning in Trees: Improving RAG for Multi-Hop QA](https://arxiv.org/abs/2601.11255) | arxiv 2026 | Prompting | LLM-configurable | 2WikiMHQA, HotpotQA, MuSiQue |
| [DQG / FT-PANEL](methods/dqg/) | [Generating Complex Question Decompositions in the Face of Distribution Shifts](https://aclanthology.org/2025.naacl-long.55/) | NAACL 2025 | Supervised (LoRA) | `Qwen2.5-7B-Instruct` + FT-PANEL LoRA | BREAKHigh, 2WikiMHQA, HotpotQA |
| [Self-RAG](methods/selfrag/) | [Self-RAG: Learning to Retrieve, Generate, and Critique](https://openreview.net/forum?id=hSyW5go0v8) | ICLR 2024 oral | Trained | `selfrag/selfrag_llama2_{7b,13b}` | TriviaQA, PopQA, PubHealth, ARC-C, ASQA |
| [PlanRAG](methods/planrag/) | [PlanRAG: A Plan-then-Retrieval Augmented Generation](https://aclanthology.org/2024.naacl-long.364/) | NAACL 2024 | Prompting | GPT-4 / LLM-adaptable | DQA (LocatE + ControlE) |

---

## Quick Start

### 1. Clone this repo
```bash
git clone <this-repo>
cd AgenticRAG-Bench
```

### 2. Download datasets
```bash
# Download all datasets needed for a specific method:
bash data/download_2wikimhqa.sh    # for AceSearcher, SearchR1, IRCoT, DecomP, RT-RAG, DQG
bash data/download_hotpotqa.sh     # for AceSearcher, SearchR1, IRCoT, DecomP, RT-RAG, DQG
# ... see data/README.md for full list
```

### 3. Run a method
```bash
# AceSearcher (14B) on 2WikiMHQA dev
bash methods/acesearcher/run.sh --dataset 2wikimhqa --split validation --output results/acesearcher/ --gpus 0,1

# SearchR1 (7B) on HotpotQA
bash methods/searchr1/run.sh --dataset hotpotqa --split validation --output results/searchr1/ --gpus 0

# IRCoT on 2WikiMHQA
bash methods/ircot/run.sh --dataset 2wikimhqa --split validation --output results/ircot/
```

### 4. Smoke test (1 example per dataset)
```bash
bash methods/acesearcher/test.sh
bash methods/searchr1/test.sh
# ...
```

### 5. Evaluate
```bash
python eval/eval.py --predictions results/acesearcher/predictions.jsonl
# → prints {"n": ..., "em": ..., "f1": ..., "acc": ...}
```

---

## Dataset Coverage

| Dataset | AceSearcher | SearchR1 | IRCoT | DecomP | RT-RAG | DQG | Self-RAG | PlanRAG |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2WikiMHQA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| HotpotQA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| MuSiQue | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — |
| NQ | — | ✓ | — | — | — | — | — | — |
| TriviaQA | — | ✓ | — | — | — | — | ✓ | — |
| Bamboogle | ✓ | — | — | — | — | — | — | — |
| FEVER / HoverFEVER | ✓ | — | — | — | — | — | — | — |
| PopQA | — | ✓ | — | — | — | — | ✓ | — |
| IIRC | — | — | ✓ | — | — | — | — | — |
| DQA (LocatE/ControlE) | — | — | — | — | — | — | — | ✓ |

---

## Repository Layout

```
AgenticRAG-Bench/
├── README.md               # This file
├── eval/
│   ├── metrics.py          # EM, F1, Accuracy
│   └── eval.py             # predictions.jsonl → metrics.json
├── data/
│   ├── README.md           # Per-dataset download instructions
│   └── download_*.sh       # Dataset download scripts
└── methods/
    ├── acesearcher/        # NeurIPS 2025 Spotlight
    ├── searchr1/           # arxiv 2025
    ├── ircot/              # ACL 2023
    ├── decomp/             # ICLR 2023
    ├── rtrag/              # arxiv 2026
    ├── dqg/                # NAACL 2025
    ├── selfrag/            # ICLR 2024
    └── planrag/            # NAACL 2024
```

Each `methods/<name>/` folder contains:
- `README.md` — paper, models, datasets, retrieval setup, run command, results
- `requirements.txt` — pinned dependencies
- `run.sh` — `--dataset`, `--split`, `--output`, `--gpus`
- `test.sh` — smoke test: 1 example per supported dataset
- `src/` — inference code adapted from the official repository

---

## Adding New Methods

1. Create `methods/<name>/src/`, `README.md`, `requirements.txt`, `run.sh`, `test.sh`
2. Follow the run.sh convention: `--dataset`, `--split`, `--output`
3. Output format: `predictions.jsonl` with `{"id", "question", "prediction", "answer"}` per line
4. Add a row to the Methods table above and the Dataset Coverage table

---

## Citation

If you use this benchmark collection, please cite the original papers for each method you use.
See each `methods/<name>/README.md` for the appropriate BibTeX.
