#!/usr/bin/env python3
"""Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.

Official source:
  https://github.com/AkariAsai/self-rag
  Paper: ICLR 2024 oral — Asai et al.

Algorithm (short-form passage reranking):
  1. Retrieve top-k passages for the question.
  2. For each passage, build a prompt:
       ### Instruction:\n{question}\n\n### Response:\n[Retrieval]<paragraph>{passage}</paragraph>
  3. Generate with logprob tracking (reflection tokens are special tokens in the vocab).
  4. Score each generation using reflection token logprobs:
       - Relevance: P([Relevant]) at position 0
       - Grounding: P([Fully supported]) + 0.5 * P([Partially supported])
       - Utility: weighted sum of [Utility:1..5]
       - Sequence score: exp(mean log-prob)
       - Total = seq + w_rel * rel + w_sup * ground + w_use * utility
  5. Select the highest-scoring generation as the prediction.

Prompt source:
  Official prompt from AkariAsai/self-rag — verbatim:
    "### Instruction:\n{question}\n\n### Response:\n"
  With retrieval:
    "{prompt}[Retrieval]<paragraph>{passage}</paragraph>"

Models:
  - selfrag/selfrag_llama2_7b
  - selfrag/selfrag_llama2_13b

Usage:
    python src/infer.py \\
        --input         ../../data/triviaqa/test.jsonl \\
        --output        /tmp/selfrag_predictions.jsonl \\
        --model         selfrag/selfrag_llama2_7b \\
        --retrieval-url http://127.0.0.1:8000/retrieve \\
        --gpus          0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Self-RAG constants
# Source: AkariAsai/self-rag/retrieval_lm/run_short_form.py
# ---------------------------------------------------------------------------

# Official prompt format (verbatim from AkariAsai/self-rag)
PROMPT = "### Instruction:\n{question}\n\n### Response:\n"

RETRIEVAL_TOKENS = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
RELEVANCE_TOKENS = ["[Irrelevant]", "[Relevant]"]
GROUND_TOKENS    = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
UTILITY_TOKENS   = [f"[Utility:{i}]" for i in range(1, 6)]

CONTROL_TOKENS = (
    GROUND_TOKENS + RETRIEVAL_TOKENS + RELEVANCE_TOKENS
    + ["<paragraph>", "</paragraph>"] + UTILITY_TOKENS
)


def clean_prediction(text: str) -> str:
    """Remove Self-RAG reflection tokens from generated text."""
    for token in CONTROL_TOKENS:
        text = text.replace(token, "")
    text = text.replace("</s>", "").replace("<|endoftext|>", "")
    return text.strip().lstrip("#:").strip()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, retrieval_url: str, topk: int = 10, timeout: float = 30.0) -> list[str]:
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
# Self-RAG scoring
# Source: logical_plan/baselines/selfrag/selfrag.py
# ---------------------------------------------------------------------------

def _token_ids(tokenizer, names: list[str]) -> dict[str, int]:
    result = {name: tokenizer.convert_tokens_to_ids(name) for name in names}
    unknown = [n for n, tid in result.items() if tid == tokenizer.unk_token_id]
    if unknown:
        raise ValueError(f"Self-RAG tokenizer is missing reflection tokens: {unknown}")
    return result


def _logprob(mapping: Any, token_id: int) -> float:
    if not mapping or token_id not in mapping:
        return -100.0
    value = mapping[token_id]
    return float(value.logprob if hasattr(value, "logprob") else value)


def _normalized_probability(logprobs: dict[str, float], positive: str) -> float:
    maximum = max(logprobs.values())
    exps = {k: math.exp(v - maximum) for k, v in logprobs.items()}
    total = sum(exps.values())
    return exps[positive] / total if total else 0.0


def score_generation(
    output: Any,
    relevance_ids: dict[str, int],
    ground_ids: dict[str, int],
    utility_ids: dict[str, int],
    use_seqscore: bool = True,
    w_rel: float = 1.0,
    w_sup: float = 1.0,
    w_use: float = 0.5,
) -> dict[str, float]:
    """Compute Self-RAG composite score from reflection token logprobs.

    Source: AkariAsai/self-rag; adapted from logical_plan/baselines/selfrag/selfrag.py
    """
    token_ids = list(output.token_ids)
    token_logprobs = output.logprobs or []
    first = token_logprobs[0] if token_logprobs else {}

    # Relevance: scored at the first generated token position
    rel_dist = {name: _logprob(first, tid) for name, tid in relevance_ids.items()}
    relevance = _normalized_probability(rel_dist, "[Relevant]")

    # Grounding: scored at the [Fully/Partially/No supported] token
    grounding = 0.0
    ground_pos = next(
        (i for i, tid in enumerate(token_ids) if tid in ground_ids.values()), None
    )
    if ground_pos is not None and ground_pos < len(token_logprobs):
        dist = {n: _logprob(token_logprobs[ground_pos], tid) for n, tid in ground_ids.items()}
        maximum = max(dist.values())
        probs = {n: math.exp(v - maximum) for n, v in dist.items()}
        total = sum(probs.values())
        if total:
            grounding = (probs["[Fully supported]"] + 0.5 * probs["[Partially supported]"]) / total

    # Utility: scored at the [Utility:N] token
    utility = 0.0
    utility_pos = next(
        (i for i, tid in enumerate(token_ids) if tid in utility_ids.values()), None
    )
    if utility_pos is not None and utility_pos < len(token_logprobs):
        dist = {n: _logprob(token_logprobs[utility_pos], tid) for n, tid in utility_ids.items()}
        maximum = max(dist.values())
        probs = {n: math.exp(v - maximum) for n, v in dist.items()}
        total = sum(probs.values())
        weights = [-1.0, -0.5, 0.0, 0.5, 1.0]
        if total:
            utility = sum(
                weights[i] * probs[f"[Utility:{i+1}]"] / total for i in range(5)
            )

    seq = math.exp(output.cumulative_logprob / max(len(token_ids), 1)) if use_seqscore else 0.0
    total_score = seq + w_rel * relevance + w_sup * grounding + w_use * utility
    return {
        "score": total_score,
        "sequence": seq,
        "relevance": relevance,
        "grounding": grounding,
        "utility": utility,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          required=True, type=Path)
    parser.add_argument("--output",         required=True, type=Path)
    parser.add_argument("--model",          default="selfrag/selfrag_llama2_7b")
    parser.add_argument("--retrieval-url",  default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--gpus",           default="0")
    parser.add_argument("--topk",           type=int, default=10)
    parser.add_argument("--max-tokens",     type=int, default=100)
    parser.add_argument("--max-logprobs",   type=int, default=100)
    parser.add_argument("--max-model-len",  type=int, default=4096)
    parser.add_argument("--batch-size",     type=int, default=8)
    parser.add_argument("--w-rel",          type=float, default=1.0)
    parser.add_argument("--w-sup",          type=float, default=1.0)
    parser.add_argument("--w-use",          type=float, default=0.5)
    parser.add_argument("--no-seqscore",    action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    n_gpu = len(args.gpus.split(","))

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    relevance_ids = _token_ids(tokenizer, RELEVANCE_TOKENS)
    ground_ids    = _token_ids(tokenizer, GROUND_TOKENS)
    utility_ids   = _token_ids(tokenizer, UTILITY_TOKENS)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=n_gpu,
        dtype="half",
        gpu_memory_utilization=0.88,
        max_model_len=args.max_model_len,
        max_logprobs=args.max_logprobs,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        logprobs=args.max_logprobs,
        skip_special_tokens=False,
    )

    records = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    done_ids: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                done_ids.add(str(json.loads(line)["id"]))
    todo = [r for r in records if str(r.get("id", r.get("_id", ""))) not in done_ids]
    print(f"Self-RAG model={args.model}; records={len(records)} todo={len(todo)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as fout:
        for batch_start in range(0, len(todo), args.batch_size):
            batch = todo[batch_start:batch_start + args.batch_size]
            questions = [r.get("question", "") for r in batch]

            # Retrieve passages for each question
            passage_batches = [
                retrieve(q, args.retrieval_url, topk=args.topk) for q in questions
            ]

            # Build prompts: one per (question, passage) pair
            prompts = []
            for q, passages in zip(questions, passage_batches):
                base_prompt = PROMPT.format(question=q)
                prompts.extend(
                    f"{base_prompt}[Retrieval]<paragraph>{p}</paragraph>"
                    for p in passages
                )

            flat_outputs = llm.generate(prompts, sampling)

            # Reassemble per-question
            offset = 0
            for rec, question, passages in zip(batch, questions, passage_batches):
                end = offset + len(passages)
                gen_slice = flat_outputs[offset:end]
                offset = end

                candidates = []
                for rank, (passage, gen) in enumerate(zip(passages, gen_slice), start=1):
                    out = gen.outputs[0]
                    scores = score_generation(
                        out,
                        relevance_ids, ground_ids, utility_ids,
                        not args.no_seqscore,
                        args.w_rel, args.w_sup, args.w_use,
                    )
                    candidates.append({
                        "rank": rank,
                        "passage": passage,
                        "raw": out.text,
                        "prediction": clean_prediction(out.text),
                        **scores,
                    })

                best = max(candidates, key=lambda c: c["score"])
                gold = rec.get("answers", rec.get("answer", ""))
                gold_str = gold[0] if isinstance(gold, list) and gold else (gold or "")
                out_rec = {
                    "id": str(rec.get("id", rec.get("_id", ""))),
                    "question": question,
                    "prediction": best["prediction"],
                    "answer": gold_str,
                    "score": best["score"],
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{out_rec['id']}] {best['prediction'][:70]}", flush=True)

    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
