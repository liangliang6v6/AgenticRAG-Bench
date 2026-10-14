#!/usr/bin/env bash
# Download evaluation datasets for AgenticRAG-Bench.
#
# Each dataset is written to  data/<name>/test.jsonl
# in a unified format: {"id":str, "question":str, "answer":str, "answers":[str,...]}
#
# Usage:
#   bash data/download.sh                            # all datasets
#   bash data/download.sh hotpotqa 2wikimultihopqa   # one or more
#
# ──────────────────────────────────────────────────────────────────────────────
# Dataset         Split used          Size     Used by
# ──────────────────────────────────────────────────────────────────────────────
# hotpotqa        fullwiki dev        7405     AceSearcher, SearchR1, IRCoT,
#                                              DecomP, RT-RAG, DQG
# 2wikimultihopqa dev                 12576    AceSearcher, SearchR1, IRCoT,
#                                              DecomP, RT-RAG, DQG
# musique         answerable dev      2417     AceSearcher, SearchR1, IRCoT,
#                                              DecomP, RT-RAG
# iirc            dev                 954      IRCoT
# nq              open dev            3610     SearchR1
# triviaqa        unfiltered dev      7993     SearchR1, Self-RAG
# bamboogle       test (full)         125      AceSearcher
# fever           FEVER v1.0 dev      19998    AceSearcher, Self-RAG
# hoverfever      HoverFEVER dev      ?        AceSearcher
# popqa           test (full)         14267    SearchR1, Self-RAG
# dqa             LocatE + ControlE   ~60      PlanRAG
# ──────────────────────────────────────────────────────────────────────────────
#
# Note on splits:
#   IRCoT (ACL 2023) evaluates on a 500-example subsample of the dev set
#   (seed 13370).  The full dev sets are downloaded here so each method can
#   apply its own subsampling if desired.  The IRCoT 500-example splits are
#   available in methods/ircot/processed_data/ after running IRCoT's own
#   processing scripts (see methods/ircot/README.md).
#
#   SearchR1 evaluates on subsets stored in their own dataset/ directory.
#   The full dev sets downloaded here are the correct source for reproducing
#   their evaluation.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL_DATASETS=(hotpotqa 2wikimultihopqa musique iirc nq triviaqa bamboogle fever hoverfever popqa dqa)

# Which datasets to download (default: all)
TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("${ALL_DATASETS[@]}")
fi

want() { for t in "${TARGETS[@]}"; do [[ "$t" == "$1" ]] && return 0; done; return 1; }

# ──────────────────────────────────────────────────────────────────────────────
# HotpotQA  —  fullwiki dev split
# Source : https://hotpotqa.github.io
# Paper  : Yang et al., EMNLP 2018
# ──────────────────────────────────────────────────────────────────────────────
if want hotpotqa; then
    OUT="$ROOT/hotpotqa"; mkdir -p "$OUT"
    echo "[hotpotqa] downloading fullwiki dev …"
    wget -q -c -O "$OUT/hotpot_dev_fullwiki_v1.json" \
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
    python3 - "$OUT" << 'EOF'
import json, sys
out_dir = sys.argv[1]
with open(f"{out_dir}/hotpot_dev_fullwiki_v1.json") as f:
    data = json.load(f)
records = [{"id": r["_id"], "question": r["question"],
            "answer": r["answer"], "answers": [r["answer"]],
            "type": r.get("type", ""), "level": r.get("level", "")}
           for r in data]
with open(f"{out_dir}/test.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
print(f"  wrote {len(records)} examples → {out_dir}/test.jsonl")
EOF
fi

# ──────────────────────────────────────────────────────────────────────────────
# 2WikiMultiHopQA  —  dev split (no public test labels)
# Source : https://github.com/Alab-NII/2wikimultihop
#          HuggingFace: Alab-NII/2wikimultihop
# Paper  : Ho et al., COLING 2020
# ──────────────────────────────────────────────────────────────────────────────
if want 2wikimultihopqa; then
    OUT="$ROOT/2wikimultihopqa"; mkdir -p "$OUT"
    echo "[2wikimultihopqa] downloading dev split via HuggingFace …"
    python3 - "$OUT" << 'EOF'
import json, sys
from datasets import load_dataset
out_dir = sys.argv[1]
ds = load_dataset("Alab-NII/2wikimultihop", trust_remote_code=True)
records = [{"id": r["_id"], "question": r["question"],
            "answer": r["answer"], "answers": [r["answer"]],
            "type": r.get("type", "")}
           for r in ds["validation"]]
with open(f"{out_dir}/test.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
print(f"  wrote {len(records)} examples → {out_dir}/test.jsonl")
EOF
fi

# ──────────────────────────────────────────────────────────────────────────────
# MuSiQue  —  answerable dev split
# Source : https://github.com/StonyBrookNLP/musique
#          HuggingFace: StonyBrookNLP/musique
# Paper  : Trivedi et al., TACL 2022
# ──────────────────────────────────────────────────────────────────────────────
if want musique; then
    OUT="$ROOT/musique"; mkdir -p "$OUT"
    echo "[musique] downloading answerable dev split via HuggingFace …"
    python3 - "$OUT" << 'EOF'
import json, sys
from datasets import load_dataset
out_dir = sys.argv[1]
ds = load_dataset("StonyBrookNLP/musique")
records = [{"id": r["id"], "question": r["question"],
            "answer": r["answer"],
            "answers": r.get("answer_aliases", [r["answer"]])}
           for r in ds["validation"] if r.get("answerable", True)]
with open(f"{out_dir}/test.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
print(f"  wrote {len(records)} examples → {out_dir}/test.jsonl")
EOF
fi

# ──────────────────────────────────────────────────────────────────────────────
# IIRC  —  dev split
# Source : https://iirc-dataset.apps.allenai.org
# Paper  : Ferguson et al., EMNLP 2020
# Used by: IRCoT
# ──────────────────────────────────────────────────────────────────────────────
if want iirc; then
    OUT="$ROOT/iirc"; mkdir -p "$OUT"
    echo "[iirc] downloading dev split …"
    TMP="$(mktemp -d)"
    wget -q -c -O "$TMP/iirc.tgz" \
        "https://iirc-dataset.apps.allenai.org/data/iirc_train_dev.tgz"
    tar -xzf "$TMP/iirc.tgz" -C "$TMP"
    python3 - "$OUT" "$TMP" << 'EOF'
import json, glob, sys
out_dir, tmp_dir = sys.argv[1], sys.argv[2]
records = []
for path in glob.glob(f"{tmp_dir}/**/*.json", recursive=True):
    if "dev" not in path:
        continue
    for passage in json.load(open(path)):
        for q in passage.get("questions", []):
            ans = q.get("answer", {}).get("answer_text", "")
            records.append({"id": q.get("qid", str(len(records))),
                             "question": q["question"],
                             "answer": ans, "answers": [ans]})
with open(f"{out_dir}/test.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
print(f"  wrote {len(records)} examples → {out_dir}/test.jsonl")
EOF
    rm -rf "$TMP"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Natural Questions (NQ)  —  open dev split
# Source : HuggingFace: nq_open
# Paper  : Kwiatkowski et al., TACL 2019
# Used by: SearchR1
# ──────────────────────────────────────────────────────────────────────────────
if want nq; then
    OUT="$ROOT/nq"; mkdir -p "$OUT"
    echo "[nq] downloading open dev split via HuggingFace …"
    python3 - "$OUT" << 'EOF'
import json, sys
from datasets import load_dataset
out_dir = sys.argv[1]
ds = load_dataset("nq_open")
records = [{"id": str(i), "question": r["question"],
            "answer": r["answer"][0] if r["answer"] else "",
            "answers": r["answer"]}
           for i, r in enumerate(ds["validation"])]
with open(f"{out_dir}/test.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
print(f"  wrote {len(records)} examples → {out_dir}/test.jsonl")
EOF
fi

# ──────────────────────────────────────────────────────────────────────────────
# TriviaQA  —  unfiltered dev split (no context)
# Source : HuggingFace: trivia_qa (unfiltered.nocontext)
# Paper  : Joshi et al., ACL 2017
# Used by: SearchR1, Self-RAG
# ──────────────────────────────────────────────────────────────────────────────
if want triviaqa; then
    OUT="$ROOT/triviaqa"; mkdir -p "$OUT"
    echo "[triviaqa] downloading unfiltered dev split via HuggingFace …"
    python3 - "$OUT" << 'EOF'
import json, sys
from datasets import load_dataset
out_dir = sys.argv[1]
ds = load_dataset("trivia_qa", "unfiltered.nocontext")
records = []
for r in ds["validation"]:
    aliases = r["answer"].get("aliases", [])
    norm    = r["answer"].get("normalized_aliases", aliases)
    records.append({"id": r["question_id"], "question": r["question"],
                    "answer": r["answer"]["value"],
                    "answers": list(set([r["answer"]["value"]] + aliases + norm))})
with open(f"{out_dir}/test.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
print(f"  wrote {len(records)} examples → {out_dir}/test.jsonl")
EOF
fi

# ──────────────────────────────────────────────────────────────────────────────
# Bamboogle  —  full test set
# Source : HuggingFace: AceSearcher/evaluation_datasets (bamboogle config)
#          Original: google/BIG-bench (task: question_answer_creation)
# Paper  : Press et al., ICLR 2023
# Used by: AceSearcher
# ──────────────────────────────────────────────────────────────────────────────
if want bamboogle; then
    OUT="$ROOT/bamboogle"; mkdir -p "$OUT"
    echo "[bamboogle] downloading test set via HuggingFace …"
    python3 - "$OUT" << 'EOF'
import json, sys
from datasets import load_dataset
out_dir = sys.argv[1]
# AceSearcher hosts the pre-processed version on HuggingFace
ds = load_dataset("AceSearcher/evaluation_datasets", "bamboogle")
split = list(ds.keys())[0]
records = []
for r in ds[split]:
    ans = r["answer"]
    answers = ans if isinstance(ans, list) else [ans]
    records.append({"id": str(r.get("id", len(records))),
                    "question": r["question"],
                    "answer": answers[0], "answers": answers})
with open(f"{out_dir}/test.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
print(f"  wrote {len(records)} examples → {out_dir}/test.jsonl")
EOF
fi

# ──────────────────────────────────────────────────────────────────────────────
# FEVER  —  v1.0 dev split (fact verification)
# Source : https://fever.ai/dataset/fever.html
#          HuggingFace: fever/fever (v1.0)
# Paper  : Thorne et al., NAACL 2018
# Used by: AceSearcher, Self-RAG
# ──────────────────────────────────────────────────────────────────────────────
if want fever; then
    OUT="$ROOT/fever"; mkdir -p "$OUT"
    echo "[fever] downloading v1.0 dev split via HuggingFace …"
    python3 - "$OUT" << 'EOF'
import json, sys
from datasets import load_dataset
out_dir = sys.argv[1]
ds = load_dataset("fever", "v1.0")
records = [{"id": str(r["id"]), "question": r["claim"],
            "answer": r["label"], "answers": [r["label"]]}
           for r in ds["validation"]]
with open(f"{out_dir}/test.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
print(f"  wrote {len(records)} examples → {out_dir}/test.jsonl")
EOF
fi

# ──────────────────────────────────────────────────────────────────────────────
# HoverFEVER  —  dev split (multi-hop fact verification)
# Source : https://github.com/hover-nlp/hover
#          HuggingFace: AceSearcher/evaluation_datasets (hover config)
# Paper  : Jiang et al., EMNLP 2020
# Used by: AceSearcher
# ──────────────────────────────────────────────────────────────────────────────
if want hoverfever; then
    OUT="$ROOT/hoverfever"; mkdir -p "$OUT"
    echo "[hoverfever] downloading dev split via HuggingFace …"
    python3 - "$OUT" << 'EOF'
import json, sys
from datasets import load_dataset
out_dir = sys.argv[1]
ds = load_dataset("AceSearcher/evaluation_datasets", "hover")
split = list(ds.keys())[0]
records = [{"id": str(r.get("id", i)), "question": r["claim"],
            "answer": r["label"], "answers": [r["label"]]}
           for i, r in enumerate(ds[split])]
with open(f"{out_dir}/test.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
print(f"  wrote {len(records)} examples → {out_dir}/test.jsonl")
EOF
fi

# ──────────────────────────────────────────────────────────────────────────────
# PopQA  —  full test set
# Source : https://github.com/AlexTMallen/adaptive-retrieval
#          HuggingFace: akariasai/PopQA
# Paper  : Mallen et al., ACL 2023
# Used by: SearchR1, Self-RAG
# ──────────────────────────────────────────────────────────────────────────────
if want popqa; then
    OUT="$ROOT/popqa"; mkdir -p "$OUT"
    echo "[popqa] downloading test set via HuggingFace …"
    python3 - "$OUT" << 'EOF'
import json, sys, ast
from datasets import load_dataset
out_dir = sys.argv[1]
ds = load_dataset("akariasai/PopQA")
records = []
for r in ds["test"]:
    possible = r.get("possible_answers", [r["obj"]])
    if isinstance(possible, str):
        try: possible = ast.literal_eval(possible)
        except Exception: possible = [possible]
    records.append({"id": str(r["id"]), "question": r["question"],
                    "answer": r["obj"], "answers": possible})
with open(f"{out_dir}/test.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
print(f"  wrote {len(records)} examples → {out_dir}/test.jsonl")
EOF
fi

# ──────────────────────────────────────────────────────────────────────────────
# DQA  —  LocatE (EU4) + ControlE (Victoria 3) scenarios
# Source : https://github.com/myeon9h/PlanRAG  (data/ directory)
# Paper  : Lee et al., NAACL 2024
# Used by: PlanRAG
# Note   : Requires Neo4j (LocatE) and MySQL (ControlE) — see
#          methods/planrag/README.md for full setup.
# ──────────────────────────────────────────────────────────────────────────────
if want dqa; then
    OUT="$ROOT/dqa"; mkdir -p "$OUT"
    echo "[dqa] cloning PlanRAG repo to extract DQA data …"
    TMP="$(mktemp -d)"
    git clone --depth 1 https://github.com/myeon9h/PlanRAG "$TMP/PlanRAG" 2>&1 | tail -1
    if [[ -d "$TMP/PlanRAG/data" ]]; then
        cp -r "$TMP/PlanRAG/data/." "$OUT/"
        echo "  copied DQA data → $OUT"
    else
        echo "  ERROR: data/ not found in PlanRAG repo; check $TMP/PlanRAG"
    fi
    rm -rf "$TMP"
fi

echo ""
echo "Done. Downloaded: ${TARGETS[*]}"
