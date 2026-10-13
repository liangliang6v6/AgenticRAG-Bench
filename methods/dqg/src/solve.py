#!/usr/bin/env python3
"""DQG/FT-PANEL: Stage 2 — Retrieve and Answer Sub-questions.

After DQG decomposes a question into sub-questions (Stage 1, decompose.py),
this script:
  1. Iterates through sub-questions in order.
  2. Substitutes #N references with previous answers.
  3. Retrieves top-k passages for each sub-question.
  4. Answers each sub-question with the reader LLM.
  5. Composes the final answer from all sub-answers.

Usage:
    python src/solve.py \\
        --input         /tmp/dqg_decomposed.jsonl \\
        --output        /tmp/dqg_predictions.jsonl \\
        --model         Qwen/Qwen2.5-14B-Instruct \\
        --retrieval-url http://127.0.0.1:8000/retrieve \\
        --gpus          0,1
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, retrieval_url: str, topk: int = 5, timeout: float = 30.0) -> list[str]:
    """Retrieve passages from the retrieval server."""
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
        passages.append(contents.strip())
    return passages


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def init_llm(model: str, gpus: str, max_model_len: int | None = None):
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
    llm = LLM(**kwargs)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return llm, tokenizer


def generate(prompt: str, llm, tokenizer, max_tokens: int = 128) -> str:
    """Generate a response."""
    from vllm import SamplingParams
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([text], sampling)
    return outputs[0].outputs[0].text.strip()


# ---------------------------------------------------------------------------
# DQG solver
# ---------------------------------------------------------------------------

def resolve_references(text: str, previous_answers: dict[int, str]) -> str:
    """Replace #N references with actual answers from previous sub-questions."""
    def replace_ref(m: re.Match) -> str:
        idx = int(m.group(1))
        return previous_answers.get(idx, m.group(0))
    return re.sub(r"#(\d+)", replace_ref, text)


def answer_subquestion(
    question: str,
    passages: list[str],
    llm,
    tokenizer,
    topk: int = 5,
) -> str:
    """Answer a sub-question given retrieved passages."""
    context = "\n\n".join(passages[:topk])
    prompt = (
        f"Answer the following question based on the provided passages. "
        f"Give a short, direct answer.\n\n"
        f"Passages:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    raw = generate(prompt, llm, tokenizer, max_tokens=64)
    # Extract short answer
    answer = raw.split("\n")[0].strip(' ".')
    return answer


def compose_final_answer(
    original_question: str,
    sub_qa: list[dict],
    llm,
    tokenizer,
) -> str:
    """Compose the final answer using all sub-question answers."""
    sub_context = "\n".join(
        f"Q{i+1}: {item['question']}\nA{i+1}: {item['answer']}"
        for i, item in enumerate(sub_qa)
    )
    prompt = (
        f"Using the sub-question answers below, answer the final question.\n\n"
        f"{sub_context}\n\n"
        f"Final question: {original_question}\n"
        f"Final answer:"
    )
    raw = generate(prompt, llm, tokenizer, max_tokens=64)
    return raw.split("\n")[0].strip(' ".')


def solve_record(
    rec: dict[str, Any],
    retrieval_url: str,
    llm,
    tokenizer,
    topk: int = 5,
) -> dict[str, Any]:
    """Run the full DQG solve pipeline for one record."""
    plan: list[dict] = rec.get("plan", [])
    question: str = rec["question"]

    if not plan:
        # No decomposition — answer directly
        passages = retrieve(question, retrieval_url, topk=topk)
        answer = answer_subquestion(question, passages, llm, tokenizer)
        return {"answer": answer, "sub_qa": []}

    previous_answers: dict[int, str] = {}
    sub_qa: list[dict] = []

    for i, step in enumerate(plan):
        sq_text = resolve_references(step["text"], previous_answers)
        passages = retrieve(sq_text, retrieval_url, topk=topk)
        sq_answer = answer_subquestion(sq_text, passages, llm, tokenizer)
        previous_answers[i + 1] = sq_answer
        sub_qa.append({"question": sq_text, "answer": sq_answer})

    final_answer = compose_final_answer(question, sub_qa, llm, tokenizer)
    return {"answer": final_answer, "sub_qa": sub_qa}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         required=True, type=Path,
                        help="Output of decompose.py (decomposed.jsonl)")
    parser.add_argument("--output",        required=True, type=Path)
    parser.add_argument("--model",         default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--gpus",          default="0,1")
    parser.add_argument("--topk",          type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    llm, tokenizer = init_llm(args.model, args.gpus, args.max_model_len)

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
            result = solve_record(rec, args.retrieval_url, llm, tokenizer, topk=args.topk)
            out_rec = {
                "id": rec["id"],
                "question": rec["question"],
                "prediction": result["answer"],
                "answer": rec.get("answer", ""),
                "sub_qa": result["sub_qa"],
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
