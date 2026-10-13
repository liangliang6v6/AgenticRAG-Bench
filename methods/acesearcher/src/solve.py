#!/usr/bin/env python3
"""AceSearcher Stage 2: Retrieve passages and answer sub-questions.

Official source:
  https://github.com/ritaranx/AceSearcher/blob/main/inference/main_qa.py
  https://github.com/ritaranx/AceSearcher/blob/main/inference/utils.py

Adapted: configurable --index-path and HuggingFace embedding model ID instead
  of hardcoded embeddings/ directory.

Usage:
    python src/solve.py \\
        --decomposed  /tmp/ace_decompose.jsonl \\
        --output      /tmp/ace_predictions.jsonl \\
        --index-path  /path/to/embeddings/2wikimultihopqa \\
        --embed-model intfloat/e5-large-v2 \\
        --model       AceSearcher/AceSearcher-14B \\
        --dataset     2wikimultihopqa \\
        --gpus        0,1
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Official answer prompts (verbatim from ritaranx/AceSearcher/inference/main_qa.py)
# ---------------------------------------------------------------------------

def _make_subq_prompt(sub_q: str, passages: list[str]) -> str:
    context = "\n\n".join(passages)
    return (
        f"You have the following context passages:\n{context}\n\n"
        f"Please answer the question '{sub_q}' with a short span using the context as reference.\n"
        "If no answer is found in the context, use your own knowledge. "
        "Your answer needs to be as short as possible."
    )


def _make_claim_prompt(sub_q: str, passages: list[str]) -> str:
    context = "\n\n".join(passages)
    return (
        f"You have the following context passages:\n{context}\n\n"
        f"Please verify whether the claim '{sub_q}' is correct using the context as reference.\n"
        "If no answer is found in the context, use your own knowledge.\n"
        "Please only output Yes or No and do not give any explanation."
    )


def _make_final_prompt(question: str, sub_answers: dict[str, tuple[str, str]], dataset: str, passages: list[str]) -> str:
    sub_text = "\n".join(
        f"### {k}: {q}, Answer for {k}: {a}" for k, (q, a) in sub_answers.items()
    )
    if dataset in ("hover", "exfever"):
        return (
            f"You are given some subquestions and their answers:\n{sub_text}\n\n"
            f"Please verify the correctness of the claim: '{question}' using the subquestions as reference. "
            "Please provide a concise and clear reasoning followed by a concise conclusion. "
            "Your answer should be Yes or No only.\nWrap your answer with <answer> and </answer> tags."
        )
    return (
        f"You are given some subquestions and their answers:\n{sub_text}\n\n"
        f"Please answer the question '{question}' with a short span using the subquestions as reference. "
        "Provides clear reasoning followed by a concise conclusion. "
        "If no relevant information is found, use your own knowledge.\n"
        "Wrap your answer with <answer> and </answer> tags."
    )


def zigzag_visit(lst: list) -> list:
    """Reorder list in zigzag fashion (AceSearcher official utility)."""
    n = len(lst)
    if n == 0:
        return lst
    result = [None] * n
    i, j = 0, 0
    while j < (n + 1) // 2:
        result[j] = lst[i]
        i += 2
        j += 1
    i = 1
    j = n - 1
    while j >= (n + 1) // 2:
        result[j] = lst[i]
        i += 2
        j -= 1
    return result


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def load_index(index_path: str, embed_model_name: str):
    """Load FAISS index from pre-computed 4-shard embeddings."""
    sentences: list[str] = []
    corpus_file = Path(index_path) / "corpus.tsv"
    with corpus_file.open() as f:
        import csv
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[0] == "id":
                continue
            sentences.append(row[1])

    shards = []
    for i in range(4):
        shard_path = Path(index_path) / embed_model_name / f"embeddings-{i}-of-4.pkl"
        with shard_path.open("rb") as f:
            shards.append(pkl.load(f))
    embeddings = np.concatenate(shards, axis=0)

    faiss.omp_set_num_threads(32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return sentences, index


def embed_query(query: str, tokenizer, model, model_name: str, device: str = "cpu") -> np.ndarray:
    if "e5" in model_name.lower():
        query = f"query: {query}"
    encoded = tokenizer(
        query, max_length=512, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model(**encoded)
        emb = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
    return emb.cpu().numpy()


def retrieve(query: str, index, corpus: list[str], tokenizer, embed_model, model_name: str, topk: int = 10) -> list[str]:
    emb = embed_query(query, tokenizer, embed_model, model_name)
    _, ids = index.search(emb, topk)
    return [corpus[i] for i in ids[0]]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(prompt: str, llm, llm_tokenizer, sampling_params) -> str:
    messages = [
        {"role": "system", "content": "You are a strict QA bot. Output ONLY the answer."},
        {"role": "user", "content": prompt.strip()},
    ]
    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([text], sampling_params)
    return outputs[0].outputs[0].text.strip()


def extract_final_answer(raw: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def replace_placeholders(text: str, answers: dict[str, str]) -> str:
    for m in re.findall(r"#(\d+)", text):
        key = f"Q{m}"
        if key in answers:
            text = text.replace(f"#{m}", answers[key])
    return text


def run_record(
    record: dict[str, Any],
    index,
    corpus: list[str],
    embed_tokenizer,
    embed_model,
    embed_model_name: str,
    llm,
    llm_tokenizer,
    sampling_params,
    dataset: str,
    topk: int = 10,
) -> dict[str, Any]:
    question = record["question"]
    sub_questions = record["sub_questions"]
    is_claim = dataset in ("hover", "exfever")

    answers: dict[str, str] = {}
    sub_answers: dict[str, tuple[str, str]] = {}
    all_passages: list[str] = []

    for sq in sub_questions:
        label = sq["label"]
        text = replace_placeholders(sq["text"], answers)
        passages = retrieve(text, index, corpus, embed_tokenizer, embed_model, embed_model_name, topk)
        all_passages += passages[:5 if len(sub_questions) <= 3 else 3]
        all_passages = list(dict.fromkeys(all_passages))
        reordered = zigzag_visit(passages)
        if is_claim:
            prompt = _make_claim_prompt(text, reordered)
        else:
            prompt = _make_subq_prompt(text, reordered)
        ans = call_llm(prompt, llm, llm_tokenizer, sampling_params)
        answers[label] = ans
        sub_answers[label] = (text, ans)

    final_prompt = _make_final_prompt(question, sub_answers, dataset, list(dict.fromkeys(all_passages)))
    raw_final = call_llm(final_prompt, llm, llm_tokenizer, sampling_params)
    prediction = extract_final_answer(raw_final)

    gold = record.get("answer", record.get("answers", ""))
    if isinstance(gold, list):
        gold_str = gold[0] if gold else ""
    else:
        gold_str = gold

    return {
        "id": record["id"],
        "question": question,
        "prediction": prediction,
        "answer": gold_str,
        "sub_questions": [sq["text"] for sq in sub_questions],
        "sub_answers": {k: v for k, (_, v) in sub_answers.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-path", required=True, type=str, help="Path to embeddings/<dataset>/ directory")
    parser.add_argument("--embed-model", default="intfloat/e5-large-v2")
    parser.add_argument("--embed-model-name", default=None, help="Subfolder name under index-path (default: same as embed-model)")
    parser.add_argument("--model", default="AceSearcher/AceSearcher-14B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graph compilation (faster startup)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    n_gpu = args.tensor_parallel_size or len(args.gpus.split(","))
    tokenizer_id = args.tokenizer or args.model
    embed_model_name = args.embed_model_name or args.embed_model.split("/")[-1]

    print(f"Loading embedding model {args.embed_model} ...")
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embed_model, trust_remote_code=True)
    embed_model = AutoModel.from_pretrained(args.embed_model, trust_remote_code=True)
    embed_model.eval()

    print(f"Building FAISS index from {args.index_path} ...")
    corpus, index = load_index(args.index_path, embed_model_name)

    print(f"Loading LLM {args.model} on {n_gpu} GPU(s) ...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer as AT

    llm_tokenizer = AT.from_pretrained(tokenizer_id, trust_remote_code=True)
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
    sampling_params = SamplingParams(temperature=args.temperature, top_p=0.99, repetition_penalty=1.05, max_tokens=512)

    records = [json.loads(l) for l in args.decomposed.read_text().splitlines() if l.strip()]
    done_ids: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                done_ids.add(str(json.loads(line)["id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as fout:
        for rec in records:
            if str(rec["id"]) in done_ids:
                continue
            result = run_record(
                rec, index, corpus, embed_tokenizer, embed_model, embed_model_name,
                llm, llm_tokenizer, sampling_params, args.dataset, args.topk,
            )
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
