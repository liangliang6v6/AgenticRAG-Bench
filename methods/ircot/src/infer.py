#!/usr/bin/env python3
"""IRCoT: Interleaving Retrieval with Chain-of-Thought Reasoning.

Official source:
  https://github.com/StonyBrookNLP/ircot
  Paper: ACL 2023 - Trivedi et al.

Algorithm:
  1. Retrieve top-k passages for the question (initial retrieval).
  2. Build a CoT prompt from the official few-shot template + retrieved passages.
  3. Generate the next reasoning sentence.
  4. Use the latest sentence as a new retrieval query.
  5. Retrieve more passages, de-duplicate, and continue.
  6. Stop when model generates "So the answer is:" or max_steps reached.
  7. Extract the final answer.

Prompt source:
  Official few-shot CoT prompts are in the IRCoT repo under prompts/<dataset>/.
  This script loads them from IRCOT_PROMPTS_DIR (default: ./prompts/).

Usage:
    python src/infer.py \\
        --input         ../../data/hotpotqa/test.jsonl \\
        --output        /tmp/ircot_predictions.jsonl \\
        --dataset       hotpotqa \\
        --model         Qwen/Qwen2.5-14B-Instruct \\
        --retrieval-url http://127.0.0.1:8000/retrieve \\
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
# Official IRCoT prompts loader
# Source: prompts/<dataset>/ in https://github.com/StonyBrookNLP/ircot
# ---------------------------------------------------------------------------

DATASET_PROMPT_FILE = {
    "hotpotqa": "gold_with_2_distractors_context_cot_qa_codex.txt",
    "2wikimultihopqa": "gold_with_2_distractors_context_cot_qa_codex.txt",
    "musique": "gold_with_2_distractors_context_cot_qa_codex.txt",
    "iirc": "gold_with_2_distractors_context_cot_qa_codex.txt",
}

ANSWER_MARKER = "So the answer is:"


def load_cot_prompt(prompts_dir: str, dataset: str) -> str:
    """Load the official few-shot CoT prompt for a dataset."""
    fname = DATASET_PROMPT_FILE.get(dataset)
    if fname is None:
        raise ValueError(f"No prompt file known for dataset: {dataset}")
    path = Path(prompts_dir) / dataset / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Official IRCoT prompt not found: {path}\n"
            f"Clone the IRCoT repo: git clone https://github.com/StonyBrookNLP/ircot\n"
            f"and set --prompts-dir to its prompts/ directory."
        )
    # Strip metadata lines (lines starting with "# METADATA:")
    lines = path.read_text(encoding="utf-8").splitlines()
    cleaned = [l for l in lines if not l.startswith("# METADATA:")]
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, retrieval_url: str, topk: int = 6, timeout: float = 30.0) -> list[str]:
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
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return llm, tokenizer


def generate_next_step(
    prompt: str,
    llm,
    tokenizer,
    max_tokens: int = 128,
    stop: list[str] | None = None,
) -> str:
    """Generate one reasoning sentence."""
    from vllm import SamplingParams

    stop_seqs = stop or ["\n", ANSWER_MARKER + " "]
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=stop_seqs)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([text], sampling)
    return outputs[0].outputs[0].text.strip()


# ---------------------------------------------------------------------------
# IRCoT core algorithm
# ---------------------------------------------------------------------------

def format_passages(passages: list[str]) -> str:
    parts = []
    for i, p in enumerate(passages, 1):
        parts.append(f"Wikipedia Title: {p}" if not p.startswith("Wikipedia") else p)
    return "\n\n".join(parts)


def extract_answer(text: str) -> str | None:
    """Extract answer from 'So the answer is: X.' pattern."""
    m = re.search(r"So the answer is:\s*(.*?)(?:\.|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(" .")
    return None


def run_ircot(
    question: str,
    cot_prompt: str,
    retrieval_url: str,
    llm,
    tokenizer,
    max_steps: int = 7,
    topk: int = 6,
) -> dict[str, Any]:
    """Run the IRCoT algorithm for one question."""
    # Initial retrieval
    passages = retrieve(question, retrieval_url, topk=topk)
    all_passages = list(dict.fromkeys(passages))

    cot_so_far = ""
    turns = []

    for step in range(max_steps):
        # Build prompt: few-shot examples + current context + partial CoT
        context_str = format_passages(all_passages[:topk])
        prompt = (
            cot_prompt
            + f"\n\nContext:\n{context_str}\n\n"
            + f"Q: {question}\n"
            + f"A: {cot_so_far}"
        ).strip()

        next_step = generate_next_step(prompt, llm, tokenizer)
        if not next_step:
            break

        cot_so_far = (cot_so_far + " " + next_step).strip()
        turns.append({"step": step, "reasoning": next_step})

        answer = extract_answer(cot_so_far)
        if answer or ANSWER_MARKER.lower() in cot_so_far.lower():
            break

        # Use the latest sentence as retrieval query
        sentences = [s.strip() for s in next_step.split(".") if s.strip()]
        query = sentences[-1] if sentences else next_step
        new_passages = retrieve(query, retrieval_url, topk=topk)
        for p in new_passages:
            if p not in all_passages:
                all_passages.append(p)

    answer = extract_answer(cot_so_far)
    if answer is None:
        # Last resort: final word(s) after the last period
        last = cot_so_far.rsplit(".", 1)[-1].strip()
        answer = last or cot_so_far

    return {"answer": answer, "cot": cot_so_far, "turns": turns}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_PROMPT_FILE.keys()))
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:9200")
    parser.add_argument("--prompts-dir", default=None, help="Path to ircot/prompts/ directory")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    # Locate prompts
    prompts_dir = args.prompts_dir
    if prompts_dir is None:
        # Try common locations
        for candidate in [
            Path(__file__).parent.parent / "prompts",
            Path("/data2/liang/data3/code/ircot/prompts"),
        ]:
            if candidate.exists():
                prompts_dir = str(candidate)
                break
    if prompts_dir is None:
        print("ERROR: --prompts-dir not specified and no prompts directory found.")
        print("Clone https://github.com/StonyBrookNLP/ircot and pass --prompts-dir ircot/prompts/")
        sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cot_prompt = load_cot_prompt(prompts_dir, args.dataset)
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
            result = run_ircot(
                question=rec["question"],
                cot_prompt=cot_prompt,
                retrieval_url=args.retrieval_url,
                llm=llm,
                tokenizer=tokenizer,
                max_steps=args.max_steps,
                topk=args.topk,
            )
            gold = rec.get("answers", rec.get("answer", ""))
            gold_str = gold[0] if isinstance(gold, list) and gold else (gold or "")
            out_rec = {
                "id": rec["id"],
                "question": rec["question"],
                "prediction": result["answer"],
                "answer": gold_str,
                "cot": result["cot"],
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
