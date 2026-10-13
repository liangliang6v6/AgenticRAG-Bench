"""Shared utilities for AceSearcher inference.

Source: https://github.com/ritaranx/AceSearcher/blob/main/inference/utils.py
Adapted: replace hardcoded GPU/device settings with configurable args.
"""
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_tokenizer(model_name: str, cache_dir: str | None = None):
    return AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )


def init_llm(
    model_path: str,
    tokenizer: str | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.88,
    max_model_len: int | None = None,
) -> LLM:
    if tokenizer is None:
        tokenizer = model_path
    kwargs = dict(
        model=model_path,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)


def make_sampling_params(
    temperature: float = 0.0,
    top_p: float = 0.99,
    max_tokens: int = 2048,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        max_tokens=max_tokens,
    )


def call_llm(
    prompt: str,
    model: LLM,
    tokenizer,
    sampling_params: SamplingParams,
) -> str:
    """Wrap prompt in system + user message and call the LLM."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict QA bot. You must answer the user's question directly "
                "based on the context provided. Do not include any conversational filler, "
                "greetings, or confirmations. Output ONLY the answer."
            ),
        },
        {"role": "user", "content": prompt.strip()},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = model.generate([text], sampling_params)
    return outputs[0].outputs[0].text.strip()


def embed_text(
    text: str,
    tokenizer,
    model,
    model_name: str,
    device: str = "cpu",
    max_length: int = 512,
) -> "np.ndarray":
    """Embed a single query string using E5-style prefix."""
    import numpy as np

    if "e5" in model_name.lower():
        text = f"query: {text}"
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**encoded)
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy()
