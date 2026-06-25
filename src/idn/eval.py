from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from statistics import mean
from typing import Any


def char_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def token_f1(a: str, b: str) -> float:
    ta = a.lower().split()
    tb = b.lower().split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def pctl(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((percentile / 100) * (len(values) - 1))))
    return float(values[idx])


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    base_lat = [r["base"]["latency_ms"] for r in records]
    teacher_lat = [r["teacher"]["latency_ms"] for r in records]
    base_out = [r["base"]["output_tokens"] for r in records]
    teacher_out = [r["teacher"]["output_tokens"] for r in records]
    sims = [r["metrics"]["base_teacher_char_similarity"] for r in records]
    f1s = [r["metrics"]["base_teacher_token_f1"] for r in records]
    ref_base = [r["metrics"].get("base_reference_char_similarity") for r in records if r["metrics"].get("base_reference_char_similarity") is not None]
    ref_teacher = [r["metrics"].get("teacher_reference_char_similarity") for r in records if r["metrics"].get("teacher_reference_char_similarity") is not None]
    return {
        "num_examples": len(records),
        "base_latency_ms": {"mean": mean(base_lat) if base_lat else 0, "p50": pctl(base_lat, 50), "p95": pctl(base_lat, 95)},
        "teacher_latency_ms": {"mean": mean(teacher_lat) if teacher_lat else 0, "p50": pctl(teacher_lat, 50), "p95": pctl(teacher_lat, 95)},
        "base_output_tokens_mean": mean(base_out) if base_out else 0,
        "teacher_output_tokens_mean": mean(teacher_out) if teacher_out else 0,
        "base_teacher_char_similarity_mean": mean(sims) if sims else 0,
        "base_teacher_token_f1_mean": mean(f1s) if f1s else 0,
        "base_reference_char_similarity_mean": mean(ref_base) if ref_base else None,
        "teacher_reference_char_similarity_mean": mean(ref_teacher) if ref_teacher else None,
    }
