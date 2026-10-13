# Datasets

Each method in AgenticRAG-Bench uses its own official evaluation datasets.
Download only the datasets needed for the methods you want to run.

## Dataset ↔ Method Coverage

| Dataset | AceSearcher | SearchR1 | IRCoT | DecomP | RT-RAG | DQG | Self-RAG | PlanRAG |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2WikiMHQA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| HotpotQA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| MuSiQue | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — |
| NQ | — | ✓ | — | — | — | — | — | — |
| TriviaQA | — | ✓ | — | — | — | — | ✓ | — |
| Bamboogle | ✓ | — | — | — | — | — | — | — |
| FEVER | ✓ | — | — | — | — | — | — | — |
| HoverFEVER | ✓ | — | — | — | — | — | — | — |
| PopQA | — | ✓ | — | — | — | — | ✓ | — |
| IIRC | — | — | ✓ | — | — | — | — | — |
| DQA (LocatE+ControlE) | — | — | — | — | — | — | — | ✓ |

---

## Download Instructions

### 2WikiMultiHopQA
```bash
bash data/download_2wikimhqa.sh
```
- **Source**: https://github.com/Alab-NII/2wikimultihop
- **HuggingFace**: `Alab-NII/2wikimultihop`
- **Splits used**: dev / test
- **Format**: `{"_id": ..., "question": ..., "answer": ..., "supporting_facts": ...}`

### HotpotQA
```bash
bash data/download_hotpotqa.sh
```
- **Source**: https://hotpotqa.github.io
- **Splits used**: fullwiki dev / test
- **Format**: `{"_id": ..., "question": ..., "answer": ..., "context": [...]}`

### MuSiQue
```bash
bash data/download_musique.sh
```
- **Source**: https://github.com/StonyBrookNLP/musique
- **Splits used**: dev (answerable) / test
- **Format**: `{"id": ..., "question": ..., "answer": ..., "question_decomposition": [...]}`

### Natural Questions (NQ)
```bash
bash data/download_nq.sh
```
- **Source**: https://ai.google.com/research/NaturalQuestions
- **Format**: `{"id": ..., "question": ..., "answer": [...]}`

### TriviaQA
```bash
bash data/download_triviaqa.sh
```
- **Source**: https://nlp.cs.washington.edu/triviaqa/
- **Splits used**: unfiltered dev / test

### Bamboogle
```bash
bash data/download_bamboogle.sh
```
- **Source**: Included in `AceSearcher/evaluation_datasets` on HuggingFace
- **HuggingFace**: `AceSearcher/evaluation_datasets`

### FEVER / HoverFEVER
```bash
bash data/download_fever.sh
```
- **FEVER source**: https://fever.ai/dataset/fever.html
- **HoverFEVER source**: https://hover-nlp.github.io

### PopQA
```bash
bash data/download_popqa.sh
```
- **Source**: https://github.com/AlexTMallen/adaptive-retrieval
- **Format**: `{"id": ..., "question": ..., "answer": ..., "prop": ...}`

### IIRC
```bash
bash data/download_iirc.sh
```
- **Source**: https://iirc-dataset.apps.allenai.org
- **Splits used**: dev / test

### DQA (PlanRAG — LocatE + ControlE)
```bash
bash data/download_dqa.sh
```
- **Source**: https://github.com/myeon9h/PlanRAG (data/ directory)
- **Scenarios**: LocatE (Europa Universalis IV), ControlE (Victoria 3)
- **Format**: `{"question": ..., "answer": ..., "db_id": ..., "db_type": ...}`

---

## Corpus Downloads (for retrieval index building)

Methods using dense/sparse retrieval need a Wikipedia passage corpus.
The IRCoT repo provides the canonical corpus-building scripts:

```bash
# Clone IRCoT corpus scripts
git clone https://github.com/StonyBrookNLP/ircot /tmp/ircot_corpus
cd /tmp/ircot_corpus && bash download/download_and_preprocess_all_datasets.sh
```

AceSearcher also documents corpus setup at:
https://github.com/ritaranx/AceSearcher#data-directories

---

## Unified Record Format

All downloaded datasets are normalized to this format by each method's `src/load_data.py`:

```json
{
  "id": "unique_question_id",
  "question": "What is the capital of the country ...",
  "answer": "Paris",
  "answers": ["Paris", "paris"],
  "metadata": {}
}
```
