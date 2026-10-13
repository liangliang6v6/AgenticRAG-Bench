"""Standard evaluation metrics for open-domain QA benchmarks.

Implements EM, F1, and Accuracy (answer containment) following the normalization
conventions used by HotpotQA, 2WikiMHQA, MuSiQue, NQ, and TriviaQA.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def normalize_answer(text: str) -> str:
    return normalize(text)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def accuracy(prediction: str, gold: str) -> float:
    """Answer containment: 1 if normalized gold is a substring of normalized prediction."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    return float(gold_norm in pred_norm or pred_norm in gold_norm)


# ---------------------------------------------------------------------------
# Multi-answer variants (some datasets have multiple acceptable answers)
# ---------------------------------------------------------------------------

def best_exact_match(prediction: str, golds: list[str]) -> float:
    return max(exact_match(prediction, g) for g in golds) if golds else 0.0


def best_f1(prediction: str, golds: list[str]) -> float:
    return max(token_f1(prediction, g) for g in golds) if golds else 0.0


def best_accuracy(prediction: str, golds: list[str]) -> float:
    return max(accuracy(prediction, g) for g in golds) if golds else 0.0


# ---------------------------------------------------------------------------
# Aggregate over a list of (prediction, gold) pairs
# ---------------------------------------------------------------------------

def aggregate(
    predictions: list[str],
    golds: list[str | list[str]],
) -> dict[str, float]:
    """Compute mean EM, F1, Acc over parallel prediction/gold lists.

    golds may contain either a single string or a list of acceptable answers.
    """
    assert len(predictions) == len(golds), "predictions and golds must be same length"
    em_scores, f1_scores, acc_scores = [], [], []
    for pred, gold in zip(predictions, golds):
        gold_list = gold if isinstance(gold, list) else [gold]
        em_scores.append(best_exact_match(pred, gold_list))
        f1_scores.append(best_f1(pred, gold_list))
        acc_scores.append(best_accuracy(pred, gold_list))
    n = len(predictions)
    return {
        "n": n,
        "em": round(sum(em_scores) / n, 4) if n else 0.0,
        "f1": round(sum(f1_scores) / n, 4) if n else 0.0,
        "acc": round(sum(acc_scores) / n, 4) if n else 0.0,
    }
