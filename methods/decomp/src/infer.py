#!/usr/bin/env python3
"""DecomP-ODQA: Decomposed Prompting for Open-Domain Question Answering.

Official source:
  https://github.com/HarshTrivedi/DecomP-ODQA
  Paper: ICLR 2023 — Khot, Trivedi, et al. "Decomposed Prompting"

Algorithm:
  1. Decompose the complex question into atomic sub-questions using the
     official question_decomposition.txt few-shot prompt.
  2. For each sub-question: retrieve top-k passages from the retrieval server.
  3. Answer each sub-question using the official singlehop_titleqa prompt +
     retrieved passages.
  4. Compose the final answer using the official multihop_cot_titleqa prompt
     with all passages and sub-answers gathered so far.

Prompt source:
  Official few-shot prompts from HarshTrivedi/DecomP-ODQA/prompts/<dataset>/:
    - question_decomposition.txt
    - singlehop_titleqa_with_all_gold_paras_and_1_distractor.txt
    - multihop_cot_titleqa_with_all_gold_paras_and_1_distractor.txt

Usage:
    python src/infer.py \\
        --input         ../../data/hotpotqa/validation.jsonl \\
        --output        /tmp/decomp_predictions.jsonl \\
        --dataset       hotpotqa \\
        --model         Qwen/Qwen2.5-14B-Instruct \\
        --retrieval-url http://127.0.0.1:9200/retrieve \\
        --prompts-dir   ./prompts \\
        --gpus          0,1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Official DecomP-ODQA prompts loader
# Source: HarshTrivedi/DecomP-ODQA/prompts/<dataset>/
# ---------------------------------------------------------------------------

DATASET_PROMPT_FILES = {
    "hotpotqa": {
        "decompose": "question_decomposition.txt",
        "singlehop": "singlehop_titleqa_with_all_gold_paras_and_1_distractor.txt",
        "multihop":  "multihop_cot_titleqa_with_all_gold_paras_and_1_distractor.txt",
    },
    "2wikimultihopqa": {
        "decompose": "question_decomposition.txt",
        "singlehop": "singlehop_titleqa_with_all_gold_paras_and_1_distractor.txt",
        "multihop":  "multihop_cot_titleqa_with_all_gold_paras_and_1_distractor.txt",
    },
    "musique": {
        "decompose": "question_decomposition.txt",
        "singlehop": "singlehop_titleqa_with_all_gold_paras_and_1_distractor.txt",
        "multihop":  "multihop_cot_titleqa_with_all_gold_paras_and_1_distractor.txt",
    },
}


def load_prompts(prompts_dir: str, dataset: str) -> dict[str, str]:
    """Load the official DecomP-ODQA few-shot prompts for a dataset."""
    files = DATASET_PROMPT_FILES.get(dataset)
    if files is None:
        raise ValueError(f"No prompts known for dataset: {dataset}")
    base = Path(prompts_dir) / dataset
    prompts = {}
    for key, fname in files.items():
        path = base / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Official DecomP prompt not found: {path}\n"
                f"Clone the repo: git clone https://github.com/HarshTrivedi/DecomP-ODQA\n"
                f"and set --prompts-dir to its prompts/ directory."
            )
        lines = path.read_text(encoding="utf-8").splitlines()
        cleaned = [l for l in lines if not l.startswith("# METADATA:")]
        prompts[key] = "\n".join(cleaned)
    return prompts


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, retrieval_url: str, topk: int = 5, timeout: float = 30.0) -> list[dict]:
    """Retrieve passages from the retrieval server.

    Returns list of dicts with 'title' and 'text' keys.
    """
    payload = {"queries": [query], "topk": topk, "return_scores": False}
    resp = requests.post(retrieval_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    results = resp.json()["result"][0]
    passages = []
    for r in results:
        if isinstance(r, dict):
            doc = r.get("document", {})
            contents = doc.get("contents", r.get("text", ""))
        else:
            contents = str(r)
        contents = contents.strip()
        # Try to split "Title\nBody" format
        lines = contents.split("\n", 1)
        if len(lines) == 2:
            title, text = lines[0].strip(), lines[1].strip()
        else:
            title, text = "", contents
        passages.append({"title": title, "text": text})
    return passages


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def init_llm(model: str, gpus: str, max_model_len: int | None = None, enforce_eager: bool = False):
    """Initialize vLLM model."""
    n_gpu = len(gpus.split(","))
    from vllm import LLM
    kwargs = dict(
        model=model,
        tensor_parallel_size=n_gpu,
        gpu_memory_utilization=0.88,
        trust_remote_code=True,
    )
    if max_model_len:
        kwargs["max_model_len"] = max_model_len
    if enforce_eager:
        kwargs["enforce_eager"] = True
    llm = LLM(**kwargs)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return llm, tokenizer


def generate(
    prompt: str,
    llm,
    tokenizer,
    max_tokens: int = 256,
    stop: list[str] | None = None,
    temperature: float = 0.0,
) -> str:
    """Generate text given a prompt string."""
    from vllm import SamplingParams
    stop_seqs = stop or ["\n\n"]
    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=stop_seqs)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([text], sampling)
    return outputs[0].outputs[0].text.strip()


# ---------------------------------------------------------------------------
# DecomP pipeline helpers
# ---------------------------------------------------------------------------

def format_passage(p: dict) -> str:
    """Format a passage in the DecomP 'Wikipedia Title: ...\n<text>' style."""
    if p["title"]:
        return f"Wikipedia Title: {p['title']}\n{p['text']}"
    return p["text"]


def parse_sub_questions(decomp_output: str) -> list[str]:
    """Extract sub-questions from model decomposition output.

    Parses lines of the form:
      QS: (select) [retrieve_odqa] <sub-question>
    """
    sub_questions = []
    for line in decomp_output.splitlines():
        line = line.strip()
        # Match: QS: (select) [retrieve_odqa] ...
        m = re.match(r"QS:\s*(?:\(select\)\s*)?\[retrieve_odqa\]\s*(.+)", line, re.IGNORECASE)
        if m:
            sub_questions.append(m.group(1).strip())
    return sub_questions


def extract_answer_from_singlehop(output: str) -> str:
    """Extract answer from singlehop model output.

    Model is expected to produce a JSON list like: ["Walls and Bridges"]
    Falls back to the raw output if parsing fails.
    """
    m = re.search(r'\["([^"]+)"', output)
    if m:
        return m.group(1)
    return output.strip(' ."')


def extract_final_answer(output: str) -> str:
    """Extract final answer from multihop model output.

    Model output ends with 'A: <answer>.' or just raw text.
    """
    m = re.search(r'\bA:\s*(.+?)\.?\s*$', output, re.DOTALL)
    if m:
        return m.group(1).strip(' "[]')
    return output.strip(' ."')


# ---------------------------------------------------------------------------
# DecomP-ODQA core algorithm
# ---------------------------------------------------------------------------

def run_decomp(
    question: str,
    prompts: dict[str, str],
    retrieval_url: str,
    llm,
    tokenizer,
    max_sub_questions: int = 4,
    topk: int = 5,
) -> dict[str, Any]:
    """Run the DecomP-ODQA algorithm for one question.

    Steps:
      1. Decompose question → sub-questions (official question_decomposition prompt)
      2. Retrieve passages for each sub-question
      3. Answer each sub-question (official singlehop_titleqa prompt)
      4. Compose final answer (official multihop_cot_titleqa prompt)
    """
    # ---- Step 1: Decompose ----
    decomp_input = prompts["decompose"] + f"\n\nQC: {question}\nQS:"
    decomp_raw = generate(decomp_input, llm, tokenizer, max_tokens=256, stop=["QS: [EOQ]", "[EOQ]"])
    sub_questions = parse_sub_questions("QS:" + decomp_raw)[:max_sub_questions]

    if not sub_questions:
        sub_questions = [question]

    # ---- Step 2 & 3: Retrieve + answer each sub-question ----
    all_passages: list[dict] = []
    sub_answers: list[dict] = []

    for sq in sub_questions:
        passages = retrieve(sq, retrieval_url, topk=topk)
        context_str = "\n\n".join(format_passage(p) for p in passages[:topk])
        sq_prompt = (
            prompts["singlehop"]
            + f"\n\n{context_str}\n\nQ: {sq}\nA:"
        )
        sq_raw = generate(sq_prompt, llm, tokenizer, max_tokens=64, stop=["\n\n", "\nQ:"])
        sq_answer = extract_answer_from_singlehop(sq_raw)
        sub_answers.append({"question": sq, "answer": sq_answer})
        for p in passages:
            if p not in all_passages:
                all_passages.append(p)

    # ---- Step 4: Compose final answer ----
    context_str = "\n\n".join(format_passage(p) for p in all_passages[:topk])
    final_prompt = (
        prompts["multihop"]
        + f"\n\n{context_str}\n\nQ: {question}\n"
    )
    final_raw = generate(final_prompt, llm, tokenizer, max_tokens=128, stop=["\n\n"])
    final_answer = extract_final_answer(final_raw)

    return {
        "answer": final_answer,
        "sub_questions": sub_answers,
        "reasoning": final_raw,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         required=True, type=Path)
    parser.add_argument("--output",        required=True, type=Path)
    parser.add_argument("--dataset",       required=True, choices=list(DATASET_PROMPT_FILES.keys()))
    parser.add_argument("--model",         default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:9200/retrieve")
    parser.add_argument("--prompts-dir",   default=None, help="Path to DecomP-ODQA prompts/ dir")
    parser.add_argument("--gpus",          default="0,1")
    parser.add_argument("--topk",          type=int, default=5)
    parser.add_argument("--max-sub-questions", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    # Locate prompts
    prompts_dir = args.prompts_dir
    if prompts_dir is None:
        for candidate in [
            Path(__file__).parent.parent / "prompts",
            Path("/data2/liang/data3/code/DecomP-ODQA/prompts"),
        ]:
            if candidate.exists():
                prompts_dir = str(candidate)
                break
    if prompts_dir is None:
        print("ERROR: --prompts-dir not specified and no prompts directory found.")
        print("Clone https://github.com/HarshTrivedi/DecomP-ODQA and pass --prompts-dir <repo>/prompts/")
        sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    prompts = load_prompts(prompts_dir, args.dataset)
    llm, tokenizer = init_llm(args.model, args.gpus, args.max_model_len, args.enforce_eager)

    records = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    done_ids: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                done_ids.add(str(json.loads(line)["id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            if str(rec["id"]) in done_ids:
                continue
            print(f"[{i+1}/{len(records)}] {rec['question'][:80]}", flush=True)
            result = run_decomp(
                question=rec["question"],
                prompts=prompts,
                retrieval_url=args.retrieval_url,
                llm=llm,
                tokenizer=tokenizer,
                topk=args.topk,
                max_sub_questions=args.max_sub_questions,
            )
            gold = rec.get("answers", rec.get("answer", ""))
            gold_str = gold[0] if isinstance(gold, list) and gold else (gold or "")
            out_rec = {
                "id": rec["id"],
                "question": rec["question"],
                "prediction": result["answer"],
                "answer": gold_str,
                "sub_questions": result["sub_questions"],
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
