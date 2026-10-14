#!/usr/bin/env python3
"""AceSearcher Stage 1: Decompose questions into sub-questions.

Official prompt source:
  https://github.com/ritaranx/AceSearcher/blob/main/inference/decompose_vllm.py

Adapted: accept unified --input jsonl (id, question, answer fields) instead of
  dataset-specific eval_datasets/ paths. Model loaded by HuggingFace ID.

Usage:
    python src/decompose.py \\
        --input  ../../data/2wikimultihopqa/test.jsonl \\
        --output /tmp/ace_decompose.jsonl \\
        --model  AceSearcher/AceSearcher-14B \\
        --dataset 2wikimultihopqa \\
        --gpus 0,1
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Official AceSearcher decompose prompts (verbatim from ritaranx/AceSearcher)
# Source: inference/decompose_vllm.py, commit main
# ---------------------------------------------------------------------------

PROMPT_QA = """You are a query planner. Your task is to minimize the search space needed to answer a question.

Instructions:
1. If the question is already atomic or cannot benefit from decomposition, output it unchanged as a single step.
2. Otherwise, decompose the question into minimal, atomic sub-questions.
3. Order sub-questions from most selective (specific, high filtering power) to least selective (broad or dependent).
4. Use dependency references when needed: refer to previous answers using #1, #2, etc.
5. Each step must contain only one clear, answerable question.
6. Avoid redundancy and unnecessary steps.

Output Format (strict):
- Use exactly: "### Q1:", "### Q2:", etc.
- Output ONLY the questions. No explanations, no preamble, no extra text.

INPUT QUESTION:
"{question}"

OUTPUT:
"""

PROMPT_CLAIM = """Please break down the claim "{claim}" into multiple smaller sub-claims that each focus on a specific component of the original statement, making it easier for a model to verify.
Begin each sub-claim with ###. If needed, refer to answers from earlier sub-claims using #1, #2, etc.
Decomposed claim:"""

TASK_MAP = {
    "bamboogle": "qa",
    "2wikimultihopqa": "qa",
    "hotpotqa": "qa",
    "musique": "qa",
    "hover": "claim",
    "exfever": "claim",
}


def make_prompt(question: str, dataset: str) -> str:
    task = TASK_MAP.get(dataset, "qa")
    if task == "qa":
        return PROMPT_QA.replace("{question}", question)
    else:
        return PROMPT_CLAIM.replace("{claim}", question)


def parse_decomposition(raw: str) -> list[dict]:
    """Parse ### Q<N>: markers into sub-question list."""
    matches = re.findall(r"###\s*(Q\d+):\s*(.*)", raw)
    if not matches:
        return []
    return [
        {"label": label.strip(), "text": text.strip(), "needs_context": True}
        for label, text in matches
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL (id, question, answer)")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL with sub_questions field")
    parser.add_argument("--model", default="AceSearcher/AceSearcher-14B", help="HuggingFace model ID")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer ID (defaults to --model)")
    parser.add_argument("--dataset", default="2wikimultihopqa", choices=list(TASK_MAP.keys()))
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU indices e.g. 0,1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graph compilation (faster startup, slower throughput)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    n_gpu = len(args.gpus.split(","))
    tokenizer_id = args.tokenizer or args.model

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True, use_fast=True)
    llm_kwargs = dict(
        model=args.model,
        tokenizer=tokenizer_id,
        tensor_parallel_size=n_gpu,
        gpu_memory_utilization=0.88,
        trust_remote_code=True,
    )
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,
        max_tokens=2048,
    )

    records = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]

    # Skip already-done records
    done_ids: set[str] = set()
    results: list[dict] = []
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_ids.add(str(r["id"]))
                results.append(r)

    todo = [r for r in records if str(r["id"]) not in done_ids]
    print(f"Decomposing {len(todo)} / {len(records)} questions (skipping {len(done_ids)} done)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as fout:
        for rec in todo:
            qtext = rec.get("question", rec.get("claim", ""))
            prompt = make_prompt(qtext, args.dataset)
            messages = [{"role": "user", "content": prompt.strip()}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = llm.generate([text], sampling_params)
            raw = outputs[0].outputs[0].text
            sub_questions = parse_decomposition(raw)
            if not sub_questions:
                # fallback: treat full question as single sub-question
                sub_questions = [{"label": "Q1", "text": qtext, "needs_context": True}]

            out_rec = {
                "id": rec["id"],
                "question": qtext,
                "answer": rec.get("answer", rec.get("answers", [])),
                "sub_questions": sub_questions,
                "raw_decompose": raw,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Wrote decompositions to {args.output}")


if __name__ == "__main__":
    main()
