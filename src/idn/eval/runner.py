"""Evaluation runner for capability specs."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, TextIO

from idn.capabilities.base import CapabilitySpec, Example
from idn.models.loader import ChatModel


def run_eval(
    spec: CapabilitySpec,
    split: str,
    limit: Optional[int],
    base_model_path: str,
    adapter_path: Optional[str],
    output_path: Path,
    dtype: str,
    device: str,
    max_new_tokens: int,
    local_files_only: bool,
    batch_size: int = 1,
    model: Optional[Any] = None,
    verbose: bool = False,
    log_file: Optional[TextIO] = None,
) -> Dict[str, Any]:
    """Run greedy model eval and write a JSON artifact."""

    if batch_size != 1:
        # Batching is intentionally deferred until the generation wrapper supports
        # per-example decoded slicing robustly for chat templates.
        batch_size = 1

    examples = spec.load_split(split=split, limit=limit)
    log = log_file
    _print_eval_header(
        log=log,
        spec=spec,
        split=split,
        requested_limit=limit,
        num_examples=len(examples),
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        output_path=output_path,
        dtype=dtype,
        device=device,
        max_new_tokens=max_new_tokens,
        metrics=[spec.metric_name],
    )

    chat_model = model or ChatModel(
        model_path=base_model_path,
        adapter_path=adapter_path,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
    )

    records = []
    parsed_preds = []
    refs = []
    latencies_ms = []

    for ex in examples:
        record = _eval_one(
            model=chat_model,
            spec=spec,
            ex=ex,
            max_new_tokens=max_new_tokens,
        )
        if verbose:
            _print_verbose_record(log=log, index=len(records) + 1, total=len(examples), record=record)
        records.append(record)
        parsed_preds.append(record["parsed_prediction"])
        refs.append(ex.reference)
        latencies_ms.append(record["latency_ms"])

    metrics = spec.metric(parsed_preds, refs)
    result = {
        "capability": spec.id,
        "split": split,
        "num_examples": len(examples),
        "model": {
            "base_model_path": base_model_path,
            "adapter_path": adapter_path,
            "dtype": dtype,
            "local_files_only": local_files_only,
        },
        "metrics": metrics,
        "latency": {
            "p50_ms": percentile(latencies_ms, 50),
            "p95_ms": percentile(latencies_ms, 95),
        },
        "lineage": {
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "predictions": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _print_eval_footer(log=log, metrics=metrics, latency=result["latency"], output_path=output_path)
    return result


def _eval_one(
    model: Any,
    spec: CapabilitySpec,
    ex: Example,
    max_new_tokens: int,
) -> Dict[str, Any]:
    messages = spec.format_prompt(ex)
    start = time.perf_counter()
    raw = model.generate(messages=messages, max_new_tokens=max_new_tokens)
    latency_ms = (time.perf_counter() - start) * 1000.0
    parsed = spec.parse_prediction(raw)
    return {
        "id": ex.id,
        "prompt_inputs": ex.prompt_inputs,
        "raw_prediction": raw,
        "parsed_prediction": parsed,
        "reference": ex.reference,
        "latency_ms": latency_ms,
    }


def percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    if pct == 50:
        return float(median(values))
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[index])


def _print_eval_header(
    log: Optional[TextIO],
    spec: CapabilitySpec,
    split: str,
    requested_limit: Optional[int],
    num_examples: int,
    base_model_path: str,
    adapter_path: Optional[str],
    output_path: Path,
    dtype: str,
    device: str,
    max_new_tokens: int,
    metrics: List[str],
) -> None:
    print("Eval configuration", file=log)
    print("  capability: {0}".format(spec.id), file=log)
    print("  dataset split: {0}".format(split), file=log)
    print("  requested limit: {0}".format(requested_limit), file=log)
    print("  samples to evaluate: {0}".format(num_examples), file=log)
    print("  model: {0}".format(base_model_path), file=log)
    print("  adapter: {0}".format(adapter_path or "none"), file=log)
    print("  dtype: {0}".format(dtype), file=log)
    print("  device: {0}".format(device), file=log)
    print("  max_new_tokens: {0}".format(max_new_tokens), file=log)
    print("  output: {0}".format(output_path), file=log)
    print("  metrics: {0}".format(", ".join(metrics)), file=log)


def _print_verbose_record(
    log: Optional[TextIO],
    index: int,
    total: int,
    record: Dict[str, Any],
) -> None:
    print("", file=log)
    print("[{0}/{1}] id={2}".format(index, total, record["id"]), file=log)
    print("input:", file=log)
    for key, value in record["prompt_inputs"].items():
        print("  {0}: {1}".format(key, value), file=log)
    print("label: {0}".format(record["reference"]), file=log)
    print("raw prediction: {0}".format(record["raw_prediction"]), file=log)
    print("parsed prediction: {0}".format(record["parsed_prediction"]), file=log)
    print("latency_ms: {0:.2f}".format(record["latency_ms"]), file=log)


def _print_eval_footer(
    log: Optional[TextIO],
    metrics: Dict[str, float],
    latency: Dict[str, float],
    output_path: Path,
) -> None:
    print("", file=log)
    print("Eval result", file=log)
    for key, value in sorted(metrics.items()):
        print("  {0}: {1:.4f}".format(key, value), file=log)
    print("  latency_p50_ms: {0:.2f}".format(latency["p50_ms"]), file=log)
    print("  latency_p95_ms: {0:.2f}".format(latency["p95_ms"]), file=log)
    print("  wrote: {0}".format(output_path), file=log)
