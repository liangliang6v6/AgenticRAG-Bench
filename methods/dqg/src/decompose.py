#!/usr/bin/env python3
"""DQG/FT-PANEL: Stage 1 — Question Decomposition.

Official source:
  https://github.com/hankelvin/complex_question_decomposition
  Paper: NAACL 2025 — Han & Gardent
  "Generating Complex Question Decompositions in the Face of Distribution Shifts"

This script uses the FT-PANEL LoRA adapter (fine-tuned on PANEL synthetic data)
on top of Qwen2.5-7B-Instruct to decompose complex multi-hop questions into
ordered atomic sub-questions.

Prompt source:
  Official prompt-v2 zero-shot instruction from the DQG repository.
  Adapted from: logical_plan/baselines/dqg/decompose.py (which cites the
  hankelvin/complex_question_decomposition repo directly).
  The INSTRUCTION constant below is the verbatim official prompt-v2.

Output format: [SQ1] sub-q-1 ; [SQ2] sub-q-2 ; ...
  - Earlier answers referenced as #N (e.g., #1 = answer to [SQ1])

Usage:
    python src/decompose.py \\
        --input   ../../data/hotpotqa/test.jsonl \\
        --output  /tmp/dqg_decomposed.jsonl \\
        --adapter /path/to/ft_panel_lora.ckpt \\
        --gpus    0
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Official DQG/FT-PANEL prompts
# Source: hankelvin/complex_question_decomposition (NAACL 2025)
#         See: github.com/hankelvin/complex_question_decomposition
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an intelligent assistant that can decompose a complex question "
    "into simpler sub-questions."
)

# Official prompt-v2 zero-shot instruction (verbatim from NAACL 2025 repo)
INSTRUCTION = (
    "I will provide you with one complex multi-hop question. Decompose it into a set of simpler "
    "sub-questions so that it will be easier to identify and retrieve the information for answering "
    "them instead of the complex question. It is important that the sub-questions (i) are not "
    "ambiguously worded, (ii) are logically connected, and (iii) that they cover all the necessary "
    "information and steps to answer the original complex question. It is very important that the "
    "decompositions must be as simple as possible, i.e. (i) there should only be as many "
    "sub-questions as absolutely necessary; and (ii) each sub-question and its answer should cover "
    "only a single atomic fact. Please provide the sub-questions in a single line, in the order "
    "that you would like them to be answered. They should be separated from each other by a \";\".\n\n"
    "When composing a sub-question, you must ensure that answers of earlier sub-questions are "
    "properly referred to whenever possible; when you need to do this, use a variable (e.g. \"#1\") "
    "for referring to the previous answer. It is very important to get the numbering for the answer "
    "variable correct. A sub-question should never refer to the answer of a sub-question that "
    "comes after it.\n\n"
    "Start your reply with your reasoning (in less than 50 words), and then immediately give the "
    "sequence of sub-questions, and stop immediately after; do not say anything else. Limit your "
    "reasoning to less than 50 words. It is very important that you prefix the start of each "
    "sub-question with a \"[SQX]\" marker that indicates the position number (X) of the "
    "sub-question, i.e. \"[SQ1] ... [SQ2] ... [SQ3] ...\""
)

ASSISTANT_ACK = (
    "I understand the instructions and I will decompose the complex question into simpler sub-questions."
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_OPERATOR_RE = re.compile(
    r"^\s*\[(?:aggregate|arithmetic|boolean|comparative|comparison|discard|"
    r"filter|group|intersection|none|project|select|superlative|union)\]\s*",
    re.IGNORECASE,
)
_SQ_MARKER_RE = re.compile(r"\[SQ(\d+)\]", re.IGNORECASE)


def parse_decomposition(raw: str) -> list[dict[str, Any]]:
    """Parse native [SQN] output with semicolon and numbered fallbacks.

    Returns list of dicts with: label (e.g. "Q1"), text, dependencies (["Q1", ...]).
    Source: logical_plan/baselines/dqg/decompose.py:parse_dqg_plan()
    """
    matches = list(_SQ_MARKER_RE.finditer(raw))
    steps: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        text = raw[match.end():end].strip().strip(" ;")
        text = _OPERATOR_RE.sub("", text).strip()
        if text:
            deps = list(dict.fromkeys(f"Q{v}" for v in re.findall(r"#(\d+)", text)))
            steps.append({"label": f"Q{int(match.group(1))}", "text": text, "dependencies": deps})

    if not steps:
        # Fallback: split on semicolons or newlines
        chunks = [c.strip(" -;\t") for c in re.split(r"\s+;\s+|\n+", raw) if c.strip()]
        for i, chunk in enumerate(chunks, 1):
            chunk = _OPERATOR_RE.sub("", chunk).strip()
            if chunk:
                deps = list(dict.fromkeys(f"Q{v}" for v in re.findall(r"#(\d+)", chunk)))
                steps.append({"label": f"Q{i}", "text": chunk, "dependencies": deps})
    return steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True, type=Path)
    parser.add_argument("--output",      required=True, type=Path)
    parser.add_argument("--adapter",     required=True, type=Path,
                        help="Path to FT-PANEL LoRA adapter directory")
    parser.add_argument("--base-model",  default=None,
                        help="Override base model (default: from adapter_config.json)")
    parser.add_argument("--gpus",        default="0")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--max-tokens",  type=int, default=500)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Load adapter config for base model
    config_path = args.adapter / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"FT-PANEL adapter not found: {args.adapter}\n"
            f"Download from: https://github.com/hankelvin/complex_question_decomposition"
        )
    with config_path.open() as f:
        adapter_config = json.load(f)
    base_model = args.base_model or adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"base_model_name_or_path missing from {config_path}")

    print(f"Adapter: {args.adapter}")
    print(f"Base model: {base_model}")

    records = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    done_ids: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                done_ids.add(str(json.loads(line)["id"]))
    todo = [r for r in records if str(r.get("id", r.get("_id", ""))) not in done_ids]
    print(f"Records: {len(records)}, todo: {len(todo)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    n_gpu = len(args.gpus.split(","))
    llm = LLM(
        model=base_model,
        tensor_parallel_size=n_gpu,
        dtype="half",
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        gpu_memory_utilization=0.88,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    lora_req = LoRARequest("ft_panel_dqg", 1, str(args.adapter.resolve()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as fout:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            prompts, meta = [], []
            for rec in batch:
                q = rec.get("question", "")
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": INSTRUCTION},
                    {"role": "assistant", "content": ASSISTANT_ACK},
                    {"role": "user", "content": f"Complex question: {q}"},
                ]
                prompts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                )
                meta.append(rec)

            outputs = llm.generate(prompts, sampling, lora_req)
            for rec, out in zip(meta, outputs):
                raw = out.outputs[0].text.strip()
                plan = parse_decomposition(raw)
                gold = rec.get("answers", rec.get("answer", ""))
                gold_str = gold[0] if isinstance(gold, list) and gold else (gold or "")
                out_rec = {
                    "id": str(rec.get("id", rec.get("_id", ""))),
                    "question": rec.get("question", ""),
                    "answer": gold_str,
                    "plan": plan,
                    "plan_raw": raw,
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                fout.flush()
            print(f"decomposed {min(start + len(batch), len(todo))}/{len(todo)}", flush=True)

    print(f"Decompositions written to {args.output}")


if __name__ == "__main__":
    main()
