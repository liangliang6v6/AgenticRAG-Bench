# PlanRAG

**Paper**: [PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers](https://arxiv.org/abs/2406.12430)  
**Venue**: NAACL 2024  
**Authors**: Myeonghwa Lee, Seonho An, Min-Chul Yang, Hyeon Hwang, Seungpil Kim  
**Official Repository**: https://github.com/myeon9h/PlanRAG

> **Note**: PlanRAG is fundamentally different from all other methods in AgenticRAG-Bench.
> It is a **structured database decision-making** system, not a free-text open-domain QA system.
> It requires running Neo4j or MySQL databases with specific video game datasets loaded.

---

## Method Overview

PlanRAG solves **Decision QA** — given a decision question, business rules, and a structured database, determine the best action:

1. **Plan**: Generate an analysis plan based on the question, rules, and database schema.
2. **Thought**: Determine what specific query to run next.
3. **Action**: Generate and execute a database query (Cypher for Neo4j, SQL for MySQL).
4. **Observation**: Get results from the database.
5. Repeat the Plan-Thought-Action-Observation loop until enough data is gathered.
6. **Decision**: Make the final decision based on all gathered evidence.

### Prompt Source

'Plan-Thought-Action-Observation' loop structure from [`myeon9h/PlanRAG`](https://github.com/myeon9h/PlanRAG).  
Reference: [arxiv.org/abs/2406.12430](https://arxiv.org/abs/2406.12430), Figure 2, Section 3.2.

---

## Datasets (DQA Benchmark)

PlanRAG uses the **DQA (Decision QA) benchmark** with two scenarios:

| Scenario | Game | Database | Task |
|---|---|---|---|
| **LocatE** | Europa Universalis IV (EU4) | Neo4j (labeled property graph) | Find the best location |
| **ControlE** | Victoria 3 | MySQL (relational) | Find the best control strategy |

Download both scenarios with:
```bash
bash data/download_dqa.sh
```
This clones `myeon9h/PlanRAG` and extracts the DQA JSONL files.

---

## Models

| Model | Notes |
|---|---|
| `gpt-4o-mini` | Default — good balance of cost and performance |
| `gpt-4o` | Original paper model (GPT-4) |
| Any OpenAI-compatible model | Set `--base-url` for local LLM server |

---

## Database Setup

### LocatE scenario (Neo4j)

```bash
# 1. Install and start Neo4j 5.x
#    https://neo4j.com/download/

# 2. Clone PlanRAG and load the EU4 graph data
git clone https://github.com/myeon9h/PlanRAG /tmp/PlanRAG
# Follow /tmp/PlanRAG/README.md to load the Neo4j database

# 3. Verify connection
python -c "from neo4j import GraphDatabase; d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','password')); d.verify_connectivity(); print('OK')"
```

### ControlE scenario (MySQL)

```bash
# 1. Install and start MySQL 8.x

# 2. Load the Victoria 3 data
# Follow /tmp/PlanRAG/README.md to create and populate the MySQL database

# 3. Verify connection
python -c "import pymysql; c = pymysql.connect(host='localhost',user='root',password='password',database='victoria3'); print('OK')"
```

---

## Installation

```bash
pip install -r methods/planrag/requirements.txt
# openai>=1.0.0, neo4j>=5.0.0, pymysql>=1.1.0
```

---

## Quick Start

```bash
# Download DQA dataset (clones PlanRAG repo)
bash data/download_dqa.sh

# Run PlanRAG on LocatE (Neo4j)
OPENAI_API_KEY=sk-... bash methods/planrag/run.sh \
    --scenario  locate \
    --output    results/planrag/ \
    --model     gpt-4o-mini

# Run PlanRAG on ControlE (MySQL)
OPENAI_API_KEY=sk-... bash methods/planrag/run.sh \
    --scenario  controle \
    --output    results/planrag/ \
    --model     gpt-4o-mini

# With local LLM (vLLM OpenAI-compatible server)
PLANRAG_BASE_URL=http://localhost:8080/v1 \
bash methods/planrag/run.sh \
    --scenario  locate \
    --output    results/planrag/ \
    --model     Qwen/Qwen2.5-14B-Instruct \
    --api-key   dummy
```

---

## Smoke Test

```bash
OPENAI_API_KEY=sk-... bash methods/planrag/test.sh
# or test a specific scenario:
OPENAI_API_KEY=sk-... bash methods/planrag/test.sh locate
```

---

## Verified Results (from paper)

| Scenario | Method | Accuracy |
|---|---|---|
| LocatE | PlanRAG (GPT-4) | 66.7 |
| LocatE | IterRAG (GPT-4) | 50.9 |
| ControlE | PlanRAG (GPT-4) | 77.0 |
| ControlE | IterRAG (GPT-4) | 69.6 |

Source: Table 2 in [Lee et al., NAACL 2024](https://arxiv.org/abs/2406.12430).

---

## Citation

```bibtex
@inproceedings{lee2024planrag,
    title     = {{PlanRAG}: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers},
    author    = {Lee, Myeonghwa and An, Seonho and Yang, Min-Chul and Hwang, Hyeon and Kim, Seungpil},
    booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
    year      = {2024},
    url       = {https://arxiv.org/abs/2406.12430},
}
```
