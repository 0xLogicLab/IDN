#!/usr/bin/env python3
"""Evaluate semantic routing in a decomposed shared LoRA basis space."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping


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
from eval_routed_lora_mix import (
    finalize_routing_metrics,
    validate_routing_weights,
)
from idn.compose.shared_basis import (
    SharedBasisError,
    compile_shared_scales,
    load_shared_basis_manifest,
    resolve_adapter_paths,
)
from idn.config import load_machine_config, resolve_relative_dir
from idn.models.loader import ChatModel
from idn.registry import load_registry
from idn.routing.router import SemanticEmbedder, SemanticRouter
from idn.routing.space import RoutingSpace


MODE = "routed_shared_basis_lora"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ROUTE-weighted shared LoRA bases plus a top-1 residual."
    )
    parser.add_argument("--machine", default="3090")
    parser.add_argument("--config-dir", default="configs/capabilities")
    parser.add_argument("--dataset", default="artifacts/eval/idn_mixed_1000.jsonl")
    parser.add_argument("--routing-space", default="artifacts/routing/space.npz")
    parser.add_argument("--basis", default="artifacts/shared_basis/manifest.json")
    parser.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--output", default="artifacts/eval/idn_mixed_shared_basis.json"
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
    basis_manifest_path = Path(args.basis)
    output_path = Path(args.output)
    registry = load_registry(
        config_dir=Path(args.config_dir), machine_path=args.machine
    )
    rows = load_mixed_records(dataset_path, registry)
    if args.limit:
        rows = rows[: args.limit]

    try:
        manifest = load_shared_basis_manifest(basis_manifest_path)
        adapter_paths = resolve_adapter_paths(manifest, basis_manifest_path)
    except SharedBasisError as exc:
        raise SystemExit("shared-basis error: {0}".format(exc)) from exc
    validate_adapters(adapter_paths)

    space = RoutingSpace.load(routing_space_path)
    capabilities = list(space.capabilities)
    if set(capabilities) != set(manifest["capabilities"]):
        raise SystemExit(
            "routing capabilities do not match shared-basis capabilities: {0} vs {1}".format(
                sorted(capabilities), sorted(manifest["capabilities"])
            )
        )
    if set(capabilities) != set(registry):
        raise SystemExit("routing capabilities do not match configured capabilities")

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
        basis_manifest_path=basis_manifest_path,
        basis_manifest_sha256=_sha256(basis_manifest_path),
        manifest=manifest,
    )
    if state.get("status") == "complete":
        print_basis_summary(state, output_path)
        return 0

    completed = {
        int(record["mixed_index"]): record for record in state.get("predictions", [])
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
        max_length=args.route_max_length or int(space.metadata.get("max_length", 512)),
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
    try:
        for row in pending:
            mixed_index = int(row["mixed_index"])
            decision = decisions_by_index[mixed_index]
            route_weights = validate_routing_weights(decision.weights, capabilities)
            compiled = compile_shared_scales(manifest, route_weights)
            chat_model.model.base_model.set_adapter(compiled["active_adapters"])
            apply_adapter_scales(chat_model.model, compiled["adapter_scales"])

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
                "shared_basis": {
                    "coordinates": compiled["coordinates"],
                    "selected_residual": compiled["selected_residual"],
                    "residual_scale": compiled["residual_scale"],
                    "active_adapters": compiled["active_adapters"],
                    "adapter_scales": compiled["adapter_scales"],
                },
            }
            done = len(completed)
            if done % args.log_every == 0 or done == len(rows):
                print(
                    "[{0}] {1}/{2} truth={3} routed={4} residual_scale={5:.4f}".format(
                        MODE,
                        done,
                        len(rows),
                        capability,
                        compiled["selected_residual"],
                        compiled["residual_scale"],
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
    print_basis_summary(state, output_path)
    return 0


def initialize_state(
    *,
    state: Dict[str, Any],
    routing_space_path: Path,
    basis_manifest_path: Path,
    basis_manifest_sha256: str,
    manifest: Mapping[str, Any],
) -> None:
    existing_space = state.get("routing_space")
    existing_basis_hash = state.get("shared_basis", {}).get("manifest_sha256")
    if existing_space is not None and existing_space != str(routing_space_path):
        raise SystemExit("existing output uses another routing space; pass --override")
    if existing_basis_hash is not None and existing_basis_hash != basis_manifest_sha256:
        raise SystemExit("existing output uses another shared basis; pass --override")
    state.update(
        {
            "oracle_capability_selection": False,
            "route_used": True,
            "routing_space": str(routing_space_path),
            "shared_basis": {
                "manifest": str(basis_manifest_path),
                "manifest_sha256": basis_manifest_sha256,
                "components": int(manifest["components"]),
                "source_rank": int(manifest["source_rank"]),
                "materialized_rank": int(manifest["materialized_rank"]),
                "retained_energy_ratio": float(
                    manifest["decomposition"]["retained_energy_ratio"]
                ),
            },
            "combination": {
                "method": "routing_weighted_shared_coordinates_top1_residual",
                "basis_count": int(manifest["components"]),
                "residual_policy": "top1_probability_scaled",
                "capabilities": list(manifest["capabilities"]),
            },
        }
    )


def apply_adapter_scales(model: Any, scales: Mapping[str, float]) -> None:
    from peft.tuners.lora import LoraLayer

    updated_layers = 0
    for module in model.modules():
        if not isinstance(module, LoraLayer):
            continue
        for adapter_name, scale in scales.items():
            module.set_scale(adapter_name, float(scale))
        updated_layers += 1
    if updated_layers == 0:
        raise RuntimeError("model contains no LoRA layers")


def print_basis_summary(state: Mapping[str, Any], output_path: Path) -> None:
    print_summary(state, output_path)
    routing = state.get("routing_metrics", {})
    if routing:
        print("routing_accuracy: {0:.4f}".format(routing["accuracy"]))
        print("routing_coverage: {0:.4f}".format(routing["coverage"]))
    basis = state.get("shared_basis", {})
    if basis:
        print("basis_components: {0}".format(basis["components"]))
        print("retained_energy_ratio: {0:.6f}".format(basis["retained_energy_ratio"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
