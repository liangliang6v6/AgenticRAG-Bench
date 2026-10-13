#!/usr/bin/env python3
"""Evaluate a predictions.jsonl file and write metrics.json.

Each line in predictions.jsonl must have at least:
  - "prediction": str   — model's answer string
  - "answer": str | list[str]   — gold answer(s)

Optional fields (used for display only):
  - "id": str
  - "question": str

Usage:
    python eval/eval.py --predictions <path> [--output <path>]
    python eval/eval.py --predictions methods/acesearcher/test_output/hotpotqa/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import aggregate


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = load_jsonl(args.predictions)
    predictions = [r["prediction"] for r in records]
    golds = [r["answer"] for r in records]

    metrics = aggregate(predictions, golds)

    out_path = args.output or args.predictions.parent / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
