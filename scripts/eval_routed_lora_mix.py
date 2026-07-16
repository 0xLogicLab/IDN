#!/usr/bin/env python3
"""Evaluate per-request ROUTE-weighted LoRA mixtures on the mixed IDN dataset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_mixed_oracle import (
    checkpoint,
    example_from_record,
    load_all_adapters,
    load_mixed_records,
    load_or_create_state,
    print_summary,
    validate_adapters,
)
from idn.config import load_machine_config, resolve_relative_dir
from idn.models.loader import ChatModel
from idn.registry import load_registry
from idn.routing.router import SemanticEmbedder, SemanticRouter
from idn.routing.space import RoutingSpace


MODE = "routed_weighted_lora"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate exact ROUTE-weighted mixtures of all capability LoRAs."
    )
    parser.add_argument("--machine", default="3090")
    parser.add_argument("--config-dir", default="configs/capabilities")
    parser.add_argument("--dataset", default="artifacts/eval/idn_mixed_1000.jsonl")
    parser.add_argument("--routing-space", default="artifacts/routing/space.npz")
    parser.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--adapter-root", default="artifacts/adapters")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--output", default="artifacts/eval/idn_mixed_routed_lora.json"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--route-batch-size", type=int, default=None)
    parser.add_argument("--route-max-length", type=int, default=None)
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
    routing_space_path = Path(args.routing_space)
    output_path = Path(args.output)
    registry = load_registry(
        config_dir=Path(args.config_dir), machine_path=args.machine
    )
    rows = load_mixed_records(dataset_path, registry)
    if args.limit:
        rows = rows[: args.limit]

    space = RoutingSpace.load(routing_space_path)
    capabilities = list(space.capabilities)
    expected_capabilities = sorted(registry)
    if sorted(capabilities) != expected_capabilities:
        raise SystemExit(
            "routing capabilities do not match configured capabilities: {0} vs {1}".format(
                sorted(capabilities), expected_capabilities
            )
        )
    adapter_paths = {
        capability: Path(args.adapter_root) / capability / "r{0}".format(args.rank)
        for capability in capabilities
    }
    validate_adapters(adapter_paths)

    state = load_or_create_state(
        output_path=output_path,
        override=args.override,
        mode=MODE,
        dataset_path=dataset_path,
        model_path=model_path,
        dtype=args.dtype,
        rows=rows,
        adapter_paths=adapter_paths,
    )
    initialize_state(
        state=state,
        routing_space_path=routing_space_path,
        rank=args.rank,
        capabilities=capabilities,
    )
    if state.get("status") == "complete":
        print_routed_summary(state, output_path)
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

    print("Computing semantic ROUTE weights for {0} requests".format(len(pending)))
    route_started = time.perf_counter()
    embedder = SemanticEmbedder(
        model_path=model_path,
        dtype=args.dtype,
        device=machine.device,
        local_files_only=machine.local_files_only,
        max_length=(
            args.route_max_length
            or int(space.metadata.get("max_length", 512))
        ),
        batch_size=args.route_batch_size or machine.eval_batch_size,
    )
    embedder.chat_model = chat_model
    router = SemanticRouter(space=space, embedder=embedder)
    decisions = router.route_many([str(row["text"]) for row in pending])
    state["routing_runtime_seconds"] = float(
        state.get("routing_runtime_seconds", 0.0)
        + (time.perf_counter() - route_started)
    )
    decisions_by_index = {
        int(row["mixed_index"]): decision
        for row, decision in zip(pending, decisions)
    }

    load_all_adapters(
        chat_model=chat_model,
        adapter_paths=adapter_paths,
        local_files_only=machine.local_files_only,
    )
    if chat_model.model is None:
        raise RuntimeError("PEFT model was not loaded")
    chat_model.model.base_model.set_adapter(capabilities)

    try:
        for row in pending:
            mixed_index = int(row["mixed_index"])
            decision = decisions_by_index[mixed_index]
            weights = validate_routing_weights(decision.weights, capabilities)
            apply_routing_weights(chat_model.model, weights)

            capability = str(row["capability"])
            registration = registry[capability]
            spec = registration.spec
            if spec is None:
                raise RuntimeError("capability is not implemented: {0}".format(capability))
            example = example_from_record(row)
            started = time.perf_counter()
            raw_prediction = chat_model.generate(
                messages=spec.format_prompt(example),
                max_new_tokens=registration.config.max_new_tokens,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            parsed_prediction = spec.parse_prediction(raw_prediction)
            completed[mixed_index] = {
                "mixed_index": mixed_index,
                "capability": capability,
                "example_id": example.id,
                "raw_prediction": raw_prediction,
                "parsed_prediction": parsed_prediction,
                "reference": example.reference,
                "verified": bool(spec.verify(raw_prediction, parsed_prediction)),
                "latency_ms": latency_ms,
                "routing": decision.to_dict(),
                "adapter_weights": weights,
            }
            done = len(completed)
            if done % args.log_every == 0 or done == len(rows):
                print(
                    "[{0}] {1}/{2} truth={3} routed={4} confidence={5:.4f}".format(
                        MODE,
                        done,
                        len(rows),
                        capability,
                        decision.capability,
                        decision.confidence,
                    ),
                    flush=True,
                )
            if done % args.checkpoint_every == 0:
                checkpoint(state, completed, output_path, status="running")
    except BaseException:
        checkpoint(state, completed, output_path, status="running")
        raise

    finalize_routing_metrics(state, completed, capabilities)
    checkpoint(state, completed, output_path, status="complete", registry=registry)
    print_routed_summary(state, output_path)
    return 0


def initialize_state(
    *,
    state: Dict[str, Any],
    routing_space_path: Path,
    rank: int,
    capabilities: Sequence[str],
) -> None:
    existing_space = state.get("routing_space")
    if existing_space is not None and existing_space != str(routing_space_path):
        raise SystemExit("existing output uses another routing space; pass --override")
    state.update(
        {
            "oracle_capability_selection": False,
            "route_used": True,
            "routing_space": str(routing_space_path),
            "combination": {
                "method": "exact_dynamic_multi_adapter_scaling",
                "capabilities": list(capabilities),
                "source_adapter_rank": int(rank),
                "effective_concatenated_rank": int(rank * len(capabilities)),
                "top_k": len(capabilities),
            },
        }
    )


def validate_routing_weights(
    raw_weights: Mapping[str, float], capabilities: Sequence[str]
) -> Dict[str, float]:
    if set(raw_weights) != set(capabilities):
        raise RuntimeError("routing decision does not contain all capability weights")
    weights = {capability: float(raw_weights[capability]) for capability in capabilities}
    if any(weight < 0.0 for weight in weights.values()):
        raise RuntimeError("routing weights must be non-negative")
    total = sum(weights.values())
    if abs(total - 1.0) > 1.0e-5:
        raise RuntimeError("routing weights must sum to one, got {0}".format(total))
    return weights


def apply_routing_weights(model: Any, weights: Mapping[str, float]) -> None:
    """Activate the exact weighted sum by changing each LoRA layer's scale."""

    from peft.tuners.lora import LoraLayer

    updated_layers = 0
    for module in model.modules():
        if not isinstance(module, LoraLayer):
            continue
        for capability, weight in weights.items():
            module.set_scale(capability, float(weight))
        updated_layers += 1
    if updated_layers == 0:
        raise RuntimeError("model contains no LoRA layers")


def finalize_routing_metrics(
    state: Dict[str, Any],
    completed: Mapping[int, Mapping[str, Any]],
    capabilities: Sequence[str],
) -> None:
    confusion = {
        expected: {predicted: 0 for predicted in capabilities}
        for expected in capabilities
    }
    correct = 0
    covered = 0
    confidence_sum = 0.0
    entropy_sum = 0.0
    for record in completed.values():
        expected = str(record["capability"])
        routing = record["routing"]
        predicted = str(routing["capability"])
        confusion[expected][predicted] += 1
        correct += int(expected == predicted)
        covered += int(bool(routing["in_scope"]))
        confidence_sum += float(routing["confidence"])
        entropy_sum += float(routing["entropy"])
    count = len(completed)
    state["routing_metrics"] = {
        "samples": count,
        "accuracy": correct / count,
        "coverage": covered / count,
        "mean_confidence": confidence_sum / count,
        "mean_entropy": entropy_sum / count,
        "confusion_matrix": confusion,
    }


def print_routed_summary(state: Mapping[str, Any], output_path: Path) -> None:
    print_summary(state, output_path)
    routing = state.get("routing_metrics", {})
    if routing:
        print("routing_accuracy: {0:.4f}".format(routing["accuracy"]))
        print("routing_coverage: {0:.4f}".format(routing["coverage"]))


if __name__ == "__main__":
    raise SystemExit(main())
