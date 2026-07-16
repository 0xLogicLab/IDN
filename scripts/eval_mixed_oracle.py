#!/usr/bin/env python3
"""Evaluate the mixed IDN dataset with a base or oracle-selected LoRA model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idn.capabilities.base import Example
from idn.config import load_machine_config, resolve_relative_dir
from idn.eval.runner import percentile
from idn.models.loader import ChatModel
from idn.registry import load_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate mixed examples without ROUTE using base or oracle LoRA."
    )
    parser.add_argument("--mode", choices=("base", "lora"), required=True)
    parser.add_argument("--machine", default="3090")
    parser.add_argument("--config-dir", default="configs/capabilities")
    parser.add_argument("--dataset", default="artifacts/eval/idn_mixed_1000.jsonl")
    parser.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--adapter-root", default="artifacts/adapters")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--override", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.checkpoint_every <= 0 or args.log_every <= 0:
        raise SystemExit("checkpoint and log intervals must be positive")

    machine = load_machine_config(args.machine)
    model_path = resolve_relative_dir(
        base_dir=args.model_dir or machine.model_dir,
        relative_dir=args.model,
        field="--model",
    )
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    registry = load_registry(
        config_dir=Path(args.config_dir), machine_path=args.machine
    )
    rows = load_mixed_records(dataset_path, registry)
    if args.limit:
        rows = rows[: args.limit]

    capabilities = sorted({str(row["capability"]) for row in rows})
    adapter_paths = {
        capability: Path(args.adapter_root) / capability / "r{0}".format(args.rank)
        for capability in capabilities
    }
    if args.mode == "lora":
        validate_adapters(adapter_paths)

    state = load_or_create_state(
        output_path=output_path,
        override=args.override,
        mode=args.mode,
        dataset_path=dataset_path,
        model_path=model_path,
        dtype=args.dtype,
        rows=rows,
        adapter_paths=adapter_paths if args.mode == "lora" else {},
    )
    if state.get("status") == "complete":
        print_summary(state, output_path)
        return 0

    completed = {
        int(record["mixed_index"]): record
        for record in state.get("predictions", [])
    }
    pending = [row for row in rows if int(row["mixed_index"]) not in completed]
    print("Loading base model once: {0}".format(model_path), flush=True)
    chat_model = ChatModel(
        model_path=model_path,
        dtype=args.dtype,
        device=machine.device,
        local_files_only=machine.local_files_only,
    ).load()
    if args.mode == "lora":
        load_all_adapters(
            chat_model=chat_model,
            adapter_paths=adapter_paths,
            local_files_only=machine.local_files_only,
        )

    current_adapter: Optional[str] = None
    try:
        for offset, row in enumerate(pending, start=1):
            capability = str(row["capability"])
            registration = registry[capability]
            spec = registration.spec
            if spec is None:
                raise RuntimeError("capability is not implemented: {0}".format(capability))
            if args.mode == "lora" and capability != current_adapter:
                chat_model.model.set_adapter(capability)
                current_adapter = capability

            example = example_from_record(row)
            started = time.perf_counter()
            raw_prediction = chat_model.generate(
                messages=spec.format_prompt(example),
                max_new_tokens=registration.config.max_new_tokens,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            parsed_prediction = spec.parse_prediction(raw_prediction)
            prediction = {
                "mixed_index": int(row["mixed_index"]),
                "capability": capability,
                "example_id": example.id,
                "raw_prediction": raw_prediction,
                "parsed_prediction": parsed_prediction,
                "reference": example.reference,
                "verified": bool(spec.verify(raw_prediction, parsed_prediction)),
                "latency_ms": latency_ms,
            }
            completed[int(row["mixed_index"])] = prediction
            done = len(completed)
            if done % args.log_every == 0 or done == len(rows):
                print(
                    "[{0}] {1}/{2} capability={3} latency_ms={4:.1f}".format(
                        args.mode, done, len(rows), capability, latency_ms
                    ),
                    flush=True,
                )
            if done % args.checkpoint_every == 0:
                checkpoint(state, completed, output_path, status="running")
    except BaseException:
        checkpoint(state, completed, output_path, status="running")
        raise

    checkpoint(state, completed, output_path, status="complete", registry=registry)
    print_summary(state, output_path)
    return 0


def load_mixed_records(path: Path, registry: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit("mixed evaluation dataset does not exist: {0}".format(path))
    rows = []
    seen_indices = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            capability = record.get("capability")
            if capability not in registry:
                raise SystemExit(
                    "unknown capability on line {0}: {1}".format(
                        line_number, capability
                    )
                )
            if not isinstance(record.get("prompt_inputs"), dict):
                raise SystemExit("missing prompt_inputs on line {0}".format(line_number))
            if "reference" not in record:
                raise SystemExit("missing reference on line {0}".format(line_number))
            mixed_index = int(record.get("mixed_index", line_number - 1))
            if mixed_index in seen_indices:
                raise SystemExit("duplicate mixed_index: {0}".format(mixed_index))
            seen_indices.add(mixed_index)
            record["mixed_index"] = mixed_index
            rows.append(record)
    if not rows:
        raise SystemExit("mixed evaluation dataset is empty: {0}".format(path))
    return sorted(rows, key=lambda record: int(record["mixed_index"]))


def example_from_record(record: Mapping[str, Any]) -> Example:
    return Example(
        id=str(record.get("example_id", record["mixed_index"])),
        prompt_inputs=dict(record["prompt_inputs"]),
        reference=record["reference"],
        gold_target=(
            str(record["gold_target"])
            if record.get("gold_target") is not None
            else None
        ),
    )


def validate_adapters(adapter_paths: Mapping[str, Path]) -> None:
    for capability, path in adapter_paths.items():
        if not (path / "adapter_config.json").exists():
            raise SystemExit(
                "missing adapter for {0}: {1}".format(capability, path)
            )


def load_all_adapters(
    *,
    chat_model: ChatModel,
    adapter_paths: Mapping[str, Path],
    local_files_only: bool,
) -> None:
    from peft import PeftModel

    if chat_model.model is None:
        raise RuntimeError("base model was not loaded")
    capabilities = sorted(adapter_paths)
    first = capabilities[0]
    print("Loading adapter: {0}".format(first), flush=True)
    model = PeftModel.from_pretrained(
        chat_model.model,
        str(adapter_paths[first]),
        adapter_name=first,
        is_trainable=False,
        local_files_only=local_files_only,
    )
    for capability in capabilities[1:]:
        print("Loading adapter: {0}".format(capability), flush=True)
        model.load_adapter(
            str(adapter_paths[capability]),
            adapter_name=capability,
            is_trainable=False,
            local_files_only=local_files_only,
        )
    model.eval()
    chat_model.model = model


def load_or_create_state(
    *,
    output_path: Path,
    override: bool,
    mode: str,
    dataset_path: Path,
    model_path: str,
    dtype: str,
    rows: Sequence[Mapping[str, Any]],
    adapter_paths: Mapping[str, Path],
) -> Dict[str, Any]:
    if output_path.exists() and not override:
        with output_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if (
            state.get("mode") != mode
            or state.get("dataset") != str(dataset_path)
            or state.get("num_examples") != len(rows)
            or state.get("model", {}).get("base_model_path") != model_path
            or state.get("model", {}).get("dtype") != dtype
        ):
            raise SystemExit(
                "existing output belongs to another run; pass --override: {0}".format(
                    output_path
                )
            )
        return state
    return {
        "artifact_version": 1,
        "status": "running",
        "mode": mode,
        "oracle_capability_selection": mode == "lora",
        "route_used": False,
        "dataset": str(dataset_path),
        "num_examples": len(rows),
        "completed_examples": 0,
        "model": {
            "base_model_path": model_path,
            "dtype": dtype,
            "adapters": {
                capability: str(path)
                for capability, path in sorted(adapter_paths.items())
            },
        },
        "aggregate_metrics": {},
        "per_capability": {},
        "predictions": [],
        "lineage": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def checkpoint(
    state: Dict[str, Any],
    completed: Mapping[int, Mapping[str, Any]],
    output_path: Path,
    *,
    status: str,
    registry: Optional[Mapping[str, Any]] = None,
) -> None:
    state["status"] = status
    state["completed_examples"] = len(completed)
    state["predictions"] = [dict(completed[index]) for index in sorted(completed)]
    state["lineage"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    if status == "complete":
        if registry is None:
            raise RuntimeError("registry is required to finalize metrics")
        finalize_metrics(state, registry)
    write_json_atomic(output_path, state)


def finalize_metrics(state: Dict[str, Any], registry: Mapping[str, Any]) -> None:
    """Compute native metrics, then average the six primary scores equally."""

    predictions = state["predictions"]
    per_capability = {}
    primary_scores = []
    weighted_total = 0.0
    total_samples = 0
    for capability in sorted({record["capability"] for record in predictions}):
        records = [
            record for record in predictions if record["capability"] == capability
        ]
        registration = registry[capability]
        spec = registration.spec
        if spec is None:
            raise RuntimeError("capability is not implemented: {0}".format(capability))
        metrics = spec.metric(
            [record["parsed_prediction"] for record in records],
            [record["reference"] for record in records],
        )
        primary_metric = registration.config.metric_name
        primary_score = float(metrics[primary_metric])
        latencies = [float(record["latency_ms"]) for record in records]
        per_capability[capability] = {
            "samples": len(records),
            "primary_metric": primary_metric,
            "primary_score": primary_score,
            "metrics": metrics,
            "verification_rate": sum(
                int(bool(record["verified"])) for record in records
            )
            / len(records),
            "latency": {
                "p50_ms": percentile(latencies, 50),
                "p95_ms": percentile(latencies, 95),
            },
        }
        primary_scores.append(primary_score)
        weighted_total += primary_score * len(records)
        total_samples += len(records)
    state["per_capability"] = per_capability
    state["aggregate_metrics"] = {
        "macro_capability_score": sum(primary_scores) / len(primary_scores),
        "sample_weighted_capability_score": weighted_total / total_samples,
        "num_capabilities": len(primary_scores),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{0}.tmp".format(path.name))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def print_summary(state: Mapping[str, Any], output_path: Path) -> None:
    print("Evaluation complete: {0}".format(state["mode"]))
    for capability, result in sorted(state.get("per_capability", {}).items()):
        print(
            "{0} {1}: {2:.4f}".format(
                capability, result["primary_metric"], result["primary_score"]
            )
        )
    aggregate = state.get("aggregate_metrics", {})
    if aggregate:
        print(
            "macro_capability_score: {0:.4f}".format(
                aggregate["macro_capability_score"]
            )
        )
    print("output: {0}".format(output_path))


if __name__ == "__main__":
    raise SystemExit(main())
