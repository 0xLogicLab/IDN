"""End-to-end benchmark runner for chat datasets served through IDNRuntime.

This runner deliberately does not call response non-emptiness "accuracy".
Open-ended datasets should supply a quality scorer, while teacher-forced token
metrics remain a separate evaluation because a normal runtime returns text only.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from idn.capabilities.base import Example
from idn.eval.dataset import BenchmarkDataset
from idn.framework.types import InferenceRequest, ResourceBudget


QualityScorer = Callable[[Example, str], Mapping[str, float]]


def run_runtime_benchmark(
    *,
    dataset: BenchmarkDataset,
    runtime: Any,
    split: str,
    output_dir: Path,
    limit: Optional[int] = None,
    max_new_tokens: int = 128,
    warmup_examples: int = 4,
    budget: ResourceBudget = ResourceBudget(),
    tenant_id: str = "wildchat-benchmark",
    quality_scorer: Optional[QualityScorer] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    progress_every: int = 100,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Evaluate requests through ``runtime.infer`` and write JSON artifacts.

    The measured latency covers the complete synchronous ``infer`` call,
    including routing, cache lookup, package selection, and generation. TTFT and
    TPOT require a streaming provider and are intentionally not inferred here.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if warmup_examples < 0:
        raise ValueError("warmup_examples cannot be negative")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")

    examples = dataset.load_split(split=split, limit=limit)
    if not examples:
        raise ValueError("dataset split contains no evaluable examples")

    output_dir = Path(output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "run_config.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (predictions_path, metrics_path, config_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "benchmark outputs already exist; choose a new output_dir or set overwrite=True"
        )

    config = {
        "protocol": "idn-runtime-benchmark-v1",
        "dataset": dataset.id,
        "split": split,
        "requested_limit": limit,
        "num_examples": len(examples),
        "max_new_tokens": max_new_tokens,
        "warmup_examples": min(warmup_examples, len(examples)),
        "tenant_id": tenant_id,
        "resource_budget": {
            "max_memory_gib": budget.max_memory_gib,
            "max_latency_ms": budget.max_latency_ms,
            "min_quality_lower_bound": budget.min_quality_lower_bound,
        },
        "quality_scorer": getattr(quality_scorer, "__name__", None),
        "metadata": dict(metadata or {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(config_path, config)

    # Warm up model loading and kernels without polluting measured cache keys.
    warmup_started = time.perf_counter()
    for example in examples[:warmup_examples]:
        warmup_request = InferenceRequest.from_messages(
            dataset.format_prompt(example),
            max_new_tokens=max_new_tokens,
            tenant_id=tenant_id + "-warmup",
        )
        runtime.infer(warmup_request, budget)
    warmup_time_s = time.perf_counter() - warmup_started

    records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, example in enumerate(examples):
            request = InferenceRequest.from_messages(
                dataset.format_prompt(example),
                max_new_tokens=max_new_tokens,
                tenant_id=tenant_id,
            )
            request_started = time.perf_counter()
            response = runtime.infer(request, budget)
            elapsed_ms = (time.perf_counter() - request_started) * 1000.0

            raw_text = str(response.text)
            parsed = dataset.parse_prediction(raw_text)
            record: Dict[str, Any] = {
                "index": index,
                "id": example.id,
                "gold_capability": _gold_capability(dataset, example),
                "routed_capability": response.capability,
                "route_confidence": float(response.route_confidence),
                "package_id": response.package_id,
                "used_fallback": bool(response.used_fallback),
                "fallback_reason": response.fallback_reason,
                "cache_hit": bool(response.cache_hit),
                "latency_ms": elapsed_ms,
                "response_nonempty": bool(raw_text.strip()),
                "verification_passed": bool(dataset.verify(raw_text, parsed)),
                "prediction": raw_text,
                "reference": _reference_text(dataset, example),
            }
            if quality_scorer is not None:
                scores = quality_scorer(example, raw_text)
                record["quality_scores"] = _validate_scores(scores)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)

            if progress_every and (
                (index + 1) % progress_every == 0 or index + 1 == len(examples)
            ):
                print(
                    "benchmark: {0}/{1} requests".format(index + 1, len(examples)),
                    flush=True,
                )

    elapsed_total_s = time.perf_counter() - started
    result = summarize_runtime_benchmark(records, elapsed_total_s=elapsed_total_s)
    result["config"] = config
    result["warmup_time_s"] = warmup_time_s
    _write_json(metrics_path, result)
    print("BENCHMARK_DONE", metrics_path, flush=True)
    return result


def summarize_runtime_benchmark(
    records: List[Mapping[str, Any]], *, elapsed_total_s: float
) -> Dict[str, Any]:
    """Aggregate benchmark records overall and by the dataset's gold class."""

    if not records:
        raise ValueError("cannot summarize an empty benchmark")
    if elapsed_total_s <= 0 or not math.isfinite(elapsed_total_s):
        raise ValueError("elapsed_total_s must be finite and positive")
    return {
        "protocol": "idn-runtime-benchmark-v1",
        "overall": _summarize_records(records, elapsed_total_s=elapsed_total_s),
        "by_capability": {
            capability: _summarize_records(group)
            for capability, group in sorted(_group_by_gold_capability(records).items())
        },
    }


def _summarize_records(
    records: List[Mapping[str, Any]], *, elapsed_total_s: Optional[float] = None
) -> Dict[str, Any]:
    known_labels = [record for record in records if record.get("gold_capability") is not None]
    correct_routes = sum(
        record.get("routed_capability") == record.get("gold_capability")
        for record in known_labels
    )
    count = len(records)
    result: Dict[str, Any] = {
        "requests": count,
        "route_capability_accuracy": (
            correct_routes / len(known_labels) if known_labels else None
        ),
        "route_labeled_requests": len(known_labels),
        "response_nonempty_rate": sum(bool(r["response_nonempty"]) for r in records) / count,
        "verification_pass_rate": sum(
            bool(r.get("verification_passed", r["response_nonempty"])) for r in records
        ) / count,
        "fallback_rate": sum(bool(r["used_fallback"]) for r in records) / count,
        "cache_hit_rate": sum(bool(r["cache_hit"]) for r in records) / count,
        "latency_ms": _latency_summary([float(r["latency_ms"]) for r in records]),
        "quality_scores": _mean_quality_scores(records),
    }
    if elapsed_total_s is not None:
        inference_time_s = sum(float(r["latency_ms"]) for r in records) / 1000.0
        result["benchmark_wall_time_s"] = elapsed_total_s
        result["inference_time_s"] = inference_time_s
        result["throughput_requests_per_s"] = count / inference_time_s
    return result


def _latency_summary(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
    }


def _percentile(ordered: List[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _group_by_gold_capability(
    records: List[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        capability = record.get("gold_capability") or "unknown"
        groups[str(capability)].append(record)
    return groups


def _gold_capability(dataset: BenchmarkDataset, example: Example) -> Optional[str]:
    label_method = getattr(dataset, "capability_label", None)
    if callable(label_method):
        label = label_method(example)
        return str(label) if label is not None else None
    reference = example.reference
    if isinstance(reference, Mapping) and reference.get("capability") is not None:
        return str(reference["capability"])
    capability = example.prompt_inputs.get("capability")
    return str(capability) if capability is not None else None


def _reference_text(dataset: BenchmarkDataset, example: Example) -> Optional[str]:
    try:
        return str(dataset.target_text(example))
    except (NotImplementedError, ValueError):
        pass
    if example.gold_target is not None:
        return str(example.gold_target)
    reference = example.reference
    if isinstance(reference, Mapping) and reference.get("text") is not None:
        return str(reference["text"])
    return None


def _validate_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    if not isinstance(scores, Mapping):
        raise TypeError("quality_scorer must return a mapping of metric names to numbers")
    result = {}
    for name, value in scores.items():
        numeric = float(value)
        if not str(name).strip() or not math.isfinite(numeric):
            raise ValueError("quality metric names must be non-empty and scores finite")
        result[str(name)] = numeric
    return result


def _mean_quality_scores(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        for name, value in record.get("quality_scores", {}).items():
            totals[name] += float(value)
            counts[name] += 1
    return {
        name: {"mean": totals[name] / counts[name], "n": counts[name]}
        for name in sorted(totals)
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))
