#!/usr/bin/env python3
"""RT-RAG: Reasoning in Trees for Retrieval-Augmented Generation.

Official source:
  https://github.com/sakura20221/RT-RAG
  Paper: "Reasoning in Trees: Improving Retrieval-Augmented Generation
          for Multi-Hop Question Answering" (arxiv 2601.11255)

Algorithm:
  1. Analyze question structure → classify as Sequential / Parallel / None.
  2. If Sequential: decompose into q_left and q_right(placeholder); solve left
     first, substitute answer into q_right, solve right.
  3. If Parallel: decompose into independent q_left and q_right; solve both,
     compose final answer.
  4. If None: retrieve top-k passages, answer directly.
  5. Compose the final answer from sub-tree answers.
  Recursion stops at max_depth or when node type is None.

Prompt source:
  System prompts adapted from sakura20221/RT-RAG/main/tree_decompose.py.
  Prompts are zero-shot instruction-based (no per-dataset few-shot files).

Usage:
    python src/infer.py \\
        --input         ../../data/hotpotqa/validation.jsonl \\
        --output        /tmp/rtrag_predictions.jsonl \\
        --model         Qwen/Qwen2.5-14B-Instruct \\
        --retrieval-url http://127.0.0.1:8000/retrieve \\
        --gpus          0,1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Prompt templates
# Source: sakura20221/RT-RAG/main/tree_decompose.py
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = (
    "You are an expert at analyzing questions and breaking them down into simpler subquestions. "
    "You carefully distinguish between sequential questions (where the second subquestion depends "
    "on the answer to the first), parallel questions (where both subquestions can be answered "
    "independently), and None-type questions (which are already simple)."
)

DECOMPOSE_USER_TMPL = """\
Analyze the following question and determine how to decompose it.

Question: {question}

Decide if the question is:
- "Sequential": the second subquestion depends on the answer to the first. Use the placeholder \
[answer_subquestion1] in the right subquestion.
- "Parallel": both subquestions can be answered independently.
- "None": the question is already atomic and does not need decomposition.

Reply in this exact JSON format:
{{
  "type": "Sequential" | "Parallel" | "None",
  "left": "<first subquestion or the original question if type is None>",
  "right": "<second subquestion (use [answer_subquestion1] if Sequential), or null if type is None>"
}}

Examples:
Q: "When did the director of Inception die?"
{{"type": "Sequential", "left": "Who directed Inception?", "right": "When did [answer_subquestion1] die?"}}

Q: "What language is spoken in both Switzerland and Luxembourg?"
{{"type": "Parallel", "left": "What language is spoken in Switzerland?", "right": "What language is spoken in Luxembourg?"}}

Q: "What is the capital of France?"
{{"type": "None", "left": "What is the capital of France?", "right": null}}

Now analyze:
Question: {question}
"""

ANSWER_SYSTEM = (
    "You are an expert at answering questions based on provided context passages. "
    "Always end your response with: 'so the Final answer is: [answer]'"
)

ANSWER_USER_TMPL = """\
Answer the following question using the provided context.

Context:
{context}

Question: {question}

Think step by step, then end with: so the Final answer is: [answer]
"""

COMPOSE_SYSTEM = (
    "You are an expert at answering questions based on provided subquestions and their answers."
)

COMPOSE_USER_TMPL = """\
Use the following subquestion answers to answer the final question.

{sub_qa}

Final Question: {question}

Think step by step, then end with: so the Final answer is: [answer]
"""


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


def chat(
    system: str,
    user: str,
    llm,
    tokenizer,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate a chat completion."""
    from vllm import SamplingParams
    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([text], sampling)
    return outputs[0].outputs[0].text.strip()


# ---------------------------------------------------------------------------
# Tree decomposition
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    question: str
    node_type: str = "None"      # Sequential | Parallel | None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    answer: str = ""
    passages: list[str] = field(default_factory=list)


def decompose_question(question: str, llm, tokenizer) -> tuple[str, str | None, str | None]:
    """Call LLM to classify + decompose the question.

    Returns (node_type, left_question, right_question).
    """
    user = DECOMPOSE_USER_TMPL.format(question=question)
    raw = chat(DECOMPOSE_SYSTEM, user, llm, tokenizer, max_tokens=256)
    # Parse JSON response
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            node_type = data.get("type", "None")
            left = data.get("left", question)
            right = data.get("right")
            return node_type, left, right
        except json.JSONDecodeError:
            pass
    return "None", question, None


def answer_leaf(question: str, retrieval_url: str, llm, tokenizer, topk: int = 5) -> tuple[str, list[str]]:
    """Answer a leaf node question using retrieved passages."""
    passages = retrieve(question, retrieval_url, topk=topk)
    context = "\n\n".join(passages[:topk])
    user = ANSWER_USER_TMPL.format(context=context, question=question)
    raw = chat(ANSWER_SYSTEM, user, llm, tokenizer, max_tokens=256)
    answer = extract_final_answer(raw)
    return answer, passages


def compose_answer(question: str, sub_qa: list[dict], llm, tokenizer) -> str:
    """Compose the final answer from sub-question answers."""
    sub_qa_str = "\n".join(
        f"Subquestion {i+1}: {item['question']}\nAnswer {i+1}: {item['answer']}"
        for i, item in enumerate(sub_qa)
    )
    user = COMPOSE_USER_TMPL.format(sub_qa=sub_qa_str, question=question)
    raw = chat(COMPOSE_SYSTEM, user, llm, tokenizer, max_tokens=256)
    return extract_final_answer(raw)


def extract_final_answer(text: str) -> str:
    """Extract answer from 'so the Final answer is: X' pattern."""
    m = re.search(r'(?:so the )?[Ff]inal answer is:?\s*(.*?)(?:\.|$)', text, re.DOTALL)
    if m:
        return m.group(1).strip(' "[].')
    # Fallback: last sentence
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return sentences[-1] if sentences else text.strip()


def build_and_solve_tree(
    question: str,
    retrieval_url: str,
    llm,
    tokenizer,
    depth: int = 0,
    max_depth: int = 4,
    topk: int = 5,
) -> TreeNode:
    """Recursively build and solve the reasoning tree.

    Implements the RT-RAG bottom-up traversal strategy.
    Source: sakura20221/RT-RAG/main/tree_decompose.py — build_question_tree()
    """
    node = TreeNode(question=question)

    if depth >= max_depth:
        node.node_type = "None"
    else:
        node_type, left_q, right_q = decompose_question(question, llm, tokenizer)
        node.node_type = node_type

        if node_type == "Sequential" and left_q and right_q:
            node.left = build_and_solve_tree(left_q, retrieval_url, llm, tokenizer, depth + 1, max_depth, topk)
            # Substitute left answer into right question
            right_q_resolved = right_q.replace("[answer_subquestion1]", node.left.answer)
            node.right = build_and_solve_tree(right_q_resolved, retrieval_url, llm, tokenizer, depth + 1, max_depth, topk)
            node.answer = compose_answer(
                question,
                [{"question": node.left.question, "answer": node.left.answer},
                 {"question": node.right.question, "answer": node.right.answer}],
                llm, tokenizer,
            )
            return node

        elif node_type == "Parallel" and left_q and right_q:
            node.left  = build_and_solve_tree(left_q,  retrieval_url, llm, tokenizer, depth + 1, max_depth, topk)
            node.right = build_and_solve_tree(right_q, retrieval_url, llm, tokenizer, depth + 1, max_depth, topk)
            node.answer = compose_answer(
                question,
                [{"question": node.left.question,  "answer": node.left.answer},
                 {"question": node.right.question, "answer": node.right.answer}],
                llm, tokenizer,
            )
            return node

    # Leaf node
    node.answer, node.passages = answer_leaf(question, retrieval_url, llm, tokenizer, topk=topk)
    return node


def tree_to_dict(node: TreeNode) -> dict:
    """Serialize tree to dict for logging."""
    d: dict[str, Any] = {"question": node.question, "type": node.node_type, "answer": node.answer}
    if node.left:
        d["left"] = tree_to_dict(node.left)
    if node.right:
        d["right"] = tree_to_dict(node.right)
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         required=True, type=Path)
    parser.add_argument("--output",        required=True, type=Path)
    parser.add_argument("--model",         default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--gpus",          default="0,1")
    parser.add_argument("--topk",          type=int, default=5)
    parser.add_argument("--max-depth",     type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
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
            root = build_and_solve_tree(
                question=rec["question"],
                retrieval_url=args.retrieval_url,
                llm=llm,
                tokenizer=tokenizer,
                max_depth=args.max_depth,
                topk=args.topk,
            )
            gold = rec.get("answers", rec.get("answer", ""))
            gold_str = gold[0] if isinstance(gold, list) and gold else (gold or "")
            out_rec = {
                "id": rec["id"],
                "question": rec["question"],
                "prediction": root.answer,
                "answer": gold_str,
                "tree": tree_to_dict(root),
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
