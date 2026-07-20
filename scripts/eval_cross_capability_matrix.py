#!/usr/bin/env python3
"""Evaluate every configured task with every capability LoRA."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from eval_mixed_oracle import load_all_adapters, validate_adapters
from idn.config import load_machine_config, resolve_relative_dir
from idn.models.loader import ChatModel
from idn.registry import load_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="h100")
    parser.add_argument("--config-dir", default="configs/capabilities")
    parser.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--adapter-root", default="artifacts/adapters")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--gsm8k-adapter",
        default="artifacts/adapters/gsm8k/r16_answer_weighted_s0.1",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--output", default="artifacts/eval/cross_capability_6x6_limit100.json"
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--override", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    machine = load_machine_config(args.machine)
    model_path = resolve_relative_dir(
        base_dir=args.model_dir or machine.model_dir,
        relative_dir=args.model,
        field="--model",
    )
    registry = load_registry(
        config_dir=Path(args.config_dir), machine_path=args.machine
    )
    capabilities = tuple(sorted(registry))
    adapter_paths = {
        capability: Path(args.adapter_root) / capability / "r{0}".format(args.rank)
        for capability in capabilities
    }
    adapter_paths["gsm8k"] = Path(args.gsm8k_adapter)
    validate_adapters(adapter_paths)
    output_path = Path(args.output)
    state = load_or_create_state(
        output_path=output_path,
        override=args.override,
        model_path=model_path,
        dtype=args.dtype,
        capabilities=capabilities,
        adapter_paths=adapter_paths,
        limit=args.limit,
    )
    examples = {
        capability: registry[capability].spec.load_split(
            split=registry[capability].config.eval_split,
            limit=args.limit,
        )
        for capability in capabilities
    }
    for capability, rows in examples.items():
        if len(rows) != args.limit:
            raise SystemExit(
                "{0} has only {1} evaluation examples, expected {2}".format(
                    capability, len(rows), args.limit
                )
            )

    print("Loading base model once: {0}".format(model_path), flush=True)
    chat_model = ChatModel(
        model_path=model_path,
        dtype=args.dtype,
        device=machine.device,
        local_files_only=machine.local_files_only,
    ).load()
    batch_size = args.batch_size or machine.eval_batch_size

    for capability in capabilities:
        run_cell(
            state=state,
            output_path=output_path,
            section="base",
            task=capability,
            adapter=None,
            chat_model=chat_model,
            spec=registry[capability].spec,
            examples=examples[capability],
            max_new_tokens=registry[capability].config.max_new_tokens,
            metric_name=registry[capability].config.metric_name,
            batch_size=batch_size,
            log_every=args.log_every,
        )

    load_all_adapters(
        chat_model=chat_model,
        adapter_paths=adapter_paths,
        local_files_only=machine.local_files_only,
    )
    if chat_model.model is None:
        raise RuntimeError("PEFT model was not loaded")
    for task in capabilities:
        for adapter in capabilities:
            chat_model.model.set_adapter(adapter)
            run_cell(
                state=state,
                output_path=output_path,
                section="matrix",
                task=task,
                adapter=adapter,
                chat_model=chat_model,
                spec=registry[task].spec,
                examples=examples[task],
                max_new_tokens=registry[task].config.max_new_tokens,
                metric_name=registry[task].config.metric_name,
                batch_size=batch_size,
                log_every=args.log_every,
            )

    state["status"] = "complete"
    state["lineage"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(output_path, state)
    print("Cross-capability evaluation complete: {0}".format(output_path))
    return 0


def run_cell(
    *,
    state: Dict[str, Any],
    output_path: Path,
    section: str,
    task: str,
    adapter: Optional[str],
    chat_model: ChatModel,
    spec: Any,
    examples: Sequence[Any],
    max_new_tokens: int,
    metric_name: str,
    batch_size: int,
    log_every: int,
) -> None:
    key = task if section == "base" else "{0}__{1}".format(task, adapter)
    cell = state[section].setdefault(
        key,
        {
            "task": task,
            "adapter": adapter,
            "status": "running",
            "completed_examples": 0,
            "predictions": [],
            "metrics": {},
        },
    )
    if cell.get("status") == "complete":
        print("Skipping complete cell: {0}".format(key), flush=True)
        return
    completed = len(cell.get("predictions", []))
    if completed > len(examples):
        raise RuntimeError("cell has more predictions than examples: {0}".format(key))
    print(
        "Evaluating cell {0}: {1}/{2}".format(key, completed, len(examples)),
        flush=True,
    )
    for offset in range(completed, len(examples), batch_size):
        batch = list(examples[offset : offset + batch_size])
        messages = [spec.format_prompt(example) for example in batch]
        started = time.perf_counter()
        if len(batch) > 1:
            raw_predictions = chat_model.generate_batch(
                messages_batch=messages, max_new_tokens=max_new_tokens
            )
        else:
            raw_predictions = [
                chat_model.generate(
                    messages=messages[0], max_new_tokens=max_new_tokens
                )
            ]
        latency_ms = 1000.0 * (time.perf_counter() - started) / len(batch)
        for example, raw_prediction in zip(batch, raw_predictions):
            cell["predictions"].append(
                {
                    "id": example.id,
                    "raw_prediction": raw_prediction,
                    "parsed_prediction": spec.parse_prediction(raw_prediction),
                    "reference": example.reference,
                    "latency_ms": latency_ms,
                }
            )
        cell["completed_examples"] = len(cell["predictions"])
        state["lineage"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(output_path, state)
        if (
            cell["completed_examples"] % log_every == 0
            or cell["completed_examples"] == len(examples)
        ):
            print(
                "[{0}] {1}/{2}".format(
                    key, cell["completed_examples"], len(examples)
                ),
                flush=True,
            )
    records = cell["predictions"]
    metrics = spec.metric(
        [record["parsed_prediction"] for record in records],
        [record["reference"] for record in records],
    )
    cell["metrics"] = metrics
    cell["primary_metric"] = metric_name
    cell["primary_score"] = float(metrics[metric_name])
    cell["status"] = "complete"
    write_json_atomic(output_path, state)
    print(
        "Completed {0}: {1}={2:.4f}".format(
            key, metric_name, cell["primary_score"]
        ),
        flush=True,
    )


def load_or_create_state(
    *,
    output_path: Path,
    override: bool,
    model_path: str,
    dtype: str,
    capabilities: Sequence[str],
    adapter_paths: Mapping[str, Path],
    limit: int,
) -> Dict[str, Any]:
    if output_path.exists() and not override:
        with output_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        expected = {
            "model_path": model_path,
            "dtype": dtype,
            "capabilities": list(capabilities),
            "limit_per_task": limit,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise SystemExit(
                    "existing output has different {0}; pass --override".format(key)
                )
        return state
    now = datetime.now(timezone.utc).isoformat()
    return {
        "artifact_version": 1,
        "status": "running",
        "experiment": "full_cross_capability_lora_matrix",
        "model_path": model_path,
        "dtype": dtype,
        "capabilities": list(capabilities),
        "limit_per_task": limit,
        "adapter_paths": {
            capability: str(path)
            for capability, path in sorted(adapter_paths.items())
        },
        "protocol": (
            "First configured evaluation examples per task; native task prompts, "
            "metrics, and decoding limits; one active LoRA per cell."
        ),
        "base": {},
        "matrix": {},
        "lineage": {"created_at": now, "updated_at": now},
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{0}.tmp".format(path.name))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


if __name__ == "__main__":
    raise SystemExit(main())
