#!/usr/bin/env python3
"""SearchR1 inference on QA datasets.

Official source:
  https://github.com/PeterGriffinJin/Search-R1/blob/main/infer_dataset.py

Adapted: accept unified --input jsonl (id, question, answer fields) instead of
  dataset-specific paths. Model loaded by HuggingFace ID via transformers.

Prompt source (verbatim from official repo, infer_dataset.py lines 113-126):
  "Answer the given question. You must conduct reasoning inside <think> and
   </think> first every time you get new information. After reasoning, if you
   find you lack some knowledge, you can call a search engine by
   <search> query </search>..."

Usage:
    # Start retrieval server first (see retrieval_server.py or run.sh)
    python src/infer.py \\
        --input         ../../data/2wikimhqa/validation.jsonl \\
        --output        /tmp/searchr1_predictions.jsonl \\
        --model         PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo \\
        --retrieval-url http://127.0.0.1:8000/retrieve \\
        --gpus          0
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
import torch
import transformers


# ---------------------------------------------------------------------------
# Official SearchR1 prompt and stop sequences
# Source: https://github.com/PeterGriffinJin/Search-R1/blob/main/infer_dataset.py
# ---------------------------------------------------------------------------

SEARCH_STOP_SEQUENCES = [
    "</search>",
    " </search>",
    "</search>\n",
    " </search>\n",
    "</search>\n\n",
    " </search>\n\n",
]
SEARCH_TEMPLATE = "\n\n{output_text}<information>{search_results}</information>\n\n"
QWEN25_EOS_TOKEN_IDS = {151645, 151643}


def build_prompt(question: str) -> str:
    """Official SearchR1 prompt template (verbatim from infer_dataset.py)."""
    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time "
        "you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a "
        "search engine by <search> query </search> and it will return the top "
        "searched results between <information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly "
        "provide the answer inside <answer> and </answer>, without detailed "
        "illustrations. For example, <answer> Beijing </answer>. "
        f"Question: {question}\n"
    )


def normalize_question(question: str) -> str:
    question = question.strip()
    if question and not question.endswith("?"):
        question += "?"
    return question


def extract_search_query(text: str) -> str | None:
    matches = re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def format_search_results(results: list[dict[str, Any]]) -> str:
    formatted = ""
    for i, item in enumerate(results, 1):
        contents = item.get("document", {}).get("contents", "")
        title, _, body = contents.partition("\n")
        formatted += f"Doc {i}(Title: {title}) {body}\n"
    return formatted


def do_search(query: str, retrieval_url: str, topk: int, timeout: float = 30.0) -> str:
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    resp = requests.post(retrieval_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return format_search_results(resp.json()["result"][0])


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------

class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, sequences: list[str], tokenizer):
        self.target_ids = [
            tokenizer.encode(s, add_special_tokens=False) for s in sequences
        ]
        self.target_lengths = [len(ids) for ids in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for ids, length in zip(self.target_ids, self.target_lengths):
            t = torch.as_tensor(ids, device=input_ids.device)
            if torch.equal(input_ids[0, -length:], t):
                return True
        return False


# ---------------------------------------------------------------------------
# Inferencer
# ---------------------------------------------------------------------------

class SearchR1Inferencer:
    def __init__(
        self,
        model_id: str,
        retrieval_url: str,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        topk: int = 3,
        max_search_turns: int = 10,
        gpu_id: int = 0,
    ) -> None:
        self.retrieval_url = retrieval_url
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.topk = topk
        self.max_search_turns = max_search_turns
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"Loading model: {model_id}")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map={"": gpu_id},
            trust_remote_code=True,
        )
        self.stopping_criteria = transformers.StoppingCriteriaList(
            [StopOnSequence(SEARCH_STOP_SEQUENCES, self.tokenizer)]
        )

    def infer_one(self, question: str) -> dict[str, Any]:
        question = normalize_question(question)
        prompt = build_prompt(question)
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

        turns: list[dict] = []
        for turn in range(self.max_search_turns + 1):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    stopping_criteria=self.stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_ids = output_ids[0, input_ids.shape[1]:]
            output_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            turns.append({"turn": turn, "output_text": output_text})

            answer = extract_answer(output_text)
            if answer is not None:
                return {"predicted_answer": answer, "turns": turns}

            query = extract_search_query(output_text)
            if query is None:
                break

            search_results = do_search(query, self.retrieval_url, self.topk)
            turns[-1]["search_query"] = query
            turns[-1]["search_results"] = search_results
            prompt += SEARCH_TEMPLATE.format(output_text=output_text, search_results=search_results)

        # Fallback: return last generated text
        last_output = turns[-1]["output_text"] if turns else ""
        return {"predicted_answer": last_output.strip(), "turns": turns}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-search-turns", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    gpu_id = 0

    inferencer = SearchR1Inferencer(
        model_id=args.model,
        retrieval_url=args.retrieval_url,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        topk=args.topk,
        max_search_turns=args.max_search_turns,
        gpu_id=gpu_id,
    )

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
            result = inferencer.infer_one(rec["question"])
            gold = rec.get("answers", rec.get("answer", ""))
            gold_str = gold[0] if isinstance(gold, list) and gold else (gold if isinstance(gold, str) else "")
            out_rec = {
                "id": rec["id"],
                "question": rec["question"],
                "prediction": result["predicted_answer"],
                "answer": gold_str,
                "turns": len(result.get("turns", [])),
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
