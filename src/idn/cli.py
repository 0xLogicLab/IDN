"""Command line entry point for the IDN experiment pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from idn.config import (
    DEFAULT_CAPABILITY_DIR,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MACHINE,
    ConfigError,
    load_machine_config,
    load_resolved_config,
    resolve_relative_dir,
)
from idn.extraction.carve import CarveError, carve_qwen2_homogeneous, validate_keep_ratio
from idn.extraction.certify import CertifyError, certify_qwen2
from idn.extraction.heal import HealError, heal_qwen2_lora
from idn.extraction.locate import LocateError, locate_qwen2_units
from idn.eval.runner import run_eval
from idn.registry import iter_capabilities, spec_from_config
from idn.routing.space import RoutingError
from idn.serving.gate import GateError, evaluate_certificate_gate
from idn.training.train_lora import (
    TrainLoRAError,
    load_lora_training_defaults,
    normalize_target_modules,
    train_lora,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="idn")
    parser.add_argument(
        "--machine",
        default=DEFAULT_MACHINE,
        help="Machine config to use: 3090, h100, or a YAML path.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list-capabilities",
        help="List configured capabilities and resolved local paths.",
    )
    list_parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CAPABILITY_DIR),
        help="Directory containing capability YAML files.",
    )
    list_parser.set_defaults(func=cmd_list_capabilities)

    datasets_parser = subparsers.add_parser(
        "list_datasets",
        help="List configured datasets with descriptions and sample counts.",
    )
    datasets_parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CAPABILITY_DIR),
        help="Directory containing capability YAML files.",
    )
    datasets_parser.add_argument(
        "--no-inspect",
        action="store_true",
        help="Do not load local datasets; print config and static metadata only.",
    )
    datasets_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    datasets_parser.set_defaults(func=cmd_list_datasets)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a base or adapter model.")
    eval_parser.add_argument("--config", required=True, help="Capability YAML.")
    eval_parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to evaluate. Defaults to capability eval_split.",
    )
    eval_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum labeled examples to evaluate. Use 0 for all samples.",
    )
    eval_parser.add_argument(
        "--model",
        default=None,
        required=True,
        help="Model directory relative to --model-dir or machine model_dir.",
    )
    eval_parser.add_argument(
        "--model-dir",
        default=None,
        help="Override the machine config model_dir used to resolve --model.",
    )
    eval_parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter path.",
    )
    eval_parser.add_argument(
        "--output",
        default=None,
        help="Output eval JSON path.",
    )
    eval_parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="bfloat16 or float32.",
    )
    eval_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override capability max_new_tokens.",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Generation batch size. Defaults to the machine eval batch size.",
    )
    eval_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each eval input, reference label, raw prediction, and parsed prediction.",
    )
    eval_parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite an existing eval output without prompting.",
    )
    eval_parser.set_defaults(func=cmd_eval)

    viz_parser = subparsers.add_parser(
        "visualize-embeddings",
        help="Visualize sampled request embeddings from all configured datasets.",
    )
    viz_parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CAPABILITY_DIR),
        help="Directory containing capability YAML files.",
    )
    viz_parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to sample. Use 'configured' for each capability train_split.",
    )
    viz_parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=500,
        help="Maximum examples to sample from each dataset.",
    )
    viz_parser.add_argument(
        "--output-dir",
        default="artifacts/analysis",
        help="Directory for plot, coordinates CSV, and diagnostics JSON.",
    )
    viz_parser.add_argument(
        "--projection",
        choices=("tsne", "pca"),
        default="tsne",
        help="2D projection method.",
    )
    viz_parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Sampling and projection seed.",
    )
    viz_parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum request text characters per example before embedding.",
    )
    viz_parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity; ignored for PCA.",
    )
    viz_parser.set_defaults(func=cmd_visualize_embeddings)

    routing_parser = subparsers.add_parser(
        "fit-routing",
        help="Fit a semantic capability routing space with the 1.5B base model.",
    )
    routing_parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CAPABILITY_DIR),
        help="Directory containing capability YAML files.",
    )
    routing_parser.add_argument("--samples-per-capability", type=int, default=500)
    routing_parser.add_argument(
        "--eval-samples-per-capability", type=int, default=200
    )
    routing_parser.add_argument(
        "--model", default="Qwen2.5-1.5B-Instruct", help="Base embedding model."
    )
    routing_parser.add_argument(
        "--model-dir", default=None, help="Override machine model_dir."
    )
    routing_parser.add_argument("--dtype", default="bfloat16")
    routing_parser.add_argument("--components", type=int, default=64)
    routing_parser.add_argument("--shrinkage", type=float, default=0.1)
    routing_parser.add_argument("--temperature", type=float, default=1.0)
    routing_parser.add_argument("--ood-percentile", type=float, default=95.0)
    routing_parser.add_argument("--max-chars", type=int, default=2000)
    routing_parser.add_argument("--max-length", type=int, default=512)
    routing_parser.add_argument("--batch-size", type=int, default=None)
    routing_parser.add_argument("--seed", type=int, default=13)
    routing_parser.add_argument(
        "--output", default="artifacts/routing/space.npz"
    )
    routing_parser.add_argument(
        "--report", default="artifacts/routing/report.json"
    )
    routing_parser.add_argument(
        "--override", action="store_true", help="Replace existing routing artifacts."
    )
    routing_parser.set_defaults(func=cmd_fit_routing)

    route_parser = subparsers.add_parser(
        "route", help="Route one request using base-model semantic embeddings."
    )
    route_parser.add_argument("--input", required=True, help="Request text to route.")
    route_parser.add_argument(
        "--space", default="artifacts/routing/space.npz", help="Routing space NPZ."
    )
    route_parser.add_argument(
        "--model", default=None, help="Override the base model recorded in the space."
    )
    route_parser.add_argument("--model-dir", default=None)
    route_parser.add_argument("--dtype", default=None)
    route_parser.add_argument("--max-length", type=int, default=None)
    route_parser.set_defaults(func=cmd_route)

    activation_parser = subparsers.add_parser(
        "visualize-activations",
        help="Visualize per-layer hidden-unit activations for configured capabilities.",
    )
    activation_parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CAPABILITY_DIR),
        help="Directory containing capability YAML files.",
    )
    activation_parser.add_argument(
        "--model",
        required=True,
        help="Model directory relative to machine model_dir.",
    )
    activation_parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter path.",
    )
    activation_parser.add_argument(
        "--capabilities",
        default=None,
        help="Optional comma-separated capability ids such as gsm8k,mbpp,cnn_dailymail.",
    )
    activation_parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to sample. Use 'configured' for each capability train_split.",
    )
    activation_parser.add_argument(
        "--samples-per-capability",
        type=int,
        default=10,
        help="Maximum request samples per capability.",
    )
    activation_parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Sampling seed.",
    )
    activation_parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum request text characters per sample before tokenization.",
    )
    activation_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ARTIFACT_DIR / "analysis"),
        help="Directory for HTML, JSON, and CSV activation outputs.",
    )
    activation_parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="bfloat16 or float32.",
    )
    activation_parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum prompt tokens before truncation. Use 0 to disable.",
    )
    activation_parser.add_argument(
        "--neuron-buckets",
        type=int,
        default=96,
        help="Number of contiguous hidden-unit buckets shown in the layer heatmap.",
    )
    activation_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of capability-contrasting hidden units listed per capability.",
    )
    activation_parser.set_defaults(func=cmd_visualize_activations)

    locate_parser = subparsers.add_parser(
        "locate",
        help="Score Qwen2 prunable units for a capability.",
    )
    locate_parser.add_argument("--config", required=True, help="Capability YAML.")
    locate_parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to score. Defaults to capability train_split.",
    )
    locate_parser.add_argument(
        "--num-examples",
        type=int,
        default=32,
        help="Number of examples to score.",
    )
    locate_parser.add_argument(
        "--model",
        required=True,
        help="Model directory relative to --model-dir or machine model_dir.",
    )
    locate_parser.add_argument(
        "--model-dir",
        default=None,
        help="Override the machine config model_dir used to resolve --model.",
    )
    locate_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory for scores.json and report.json. Defaults to "
            "artifacts/locate/{capability}/{split}_{model}_n{num_examples}_{dtype}."
        ),
    )
    locate_parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="bfloat16 or float32.",
    )
    locate_parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum prompt+target tokens before truncation.",
    )
    locate_parser.set_defaults(func=cmd_locate)

    carve_parser = subparsers.add_parser(
        "carve",
        help="Create a carved model artifact from a local HF checkpoint.",
    )
    carve_parser.add_argument(
        "--model",
        required=True,
        help="Model directory relative to machine model_dir.",
    )
    carve_parser.add_argument(
        "--keep-ratio",
        type=float,
        default=1.0,
        help="Fraction of FFN channels to keep; attention defaults to full keep.",
    )
    carve_parser.add_argument(
        "--attention-keep-ratio",
        type=float,
        default=None,
        help="Optional attention-head keep ratio. Defaults to 1.0.",
    )
    carve_parser.add_argument(
        "--ffn-keep-ratio",
        type=float,
        default=None,
        help="Optional FFN-channel keep ratio overriding --keep-ratio.",
    )
    carve_parser.add_argument(
        "--scores",
        default=None,
        help="Optional locate scores path used to select units.",
    )
    carve_parser.add_argument(
        "--output",
        default=None,
        help="Output artifact directory. Defaults to artifacts/carving/{model}/carve_r{keep_ratio}.",
    )
    carve_parser.add_argument(
        "--override",
        action="store_true",
        help="Replace an existing output artifact directory.",
    )
    carve_parser.set_defaults(func=cmd_carve)

    heal_parser = subparsers.add_parser(
        "heal",
        help="Heal a carved Qwen2 model with teacher-guided LoRA distillation.",
    )
    heal_parser.add_argument("--config", required=True, help="Capability YAML.")
    heal_parser.add_argument(
        "--carved",
        required=True,
        help="Carved model artifact directory.",
    )
    heal_parser.add_argument(
        "--teacher",
        required=True,
        help="Teacher model relative to --model-dir or machine model_dir.",
    )
    heal_parser.add_argument(
        "--model-dir",
        default=None,
        help="Override the machine config model_dir used to resolve --teacher.",
    )
    heal_parser.add_argument(
        "--split",
        default=None,
        help="Unlabeled prompt split. Defaults to capability train_split.",
    )
    heal_parser.add_argument("--num-examples", type=int, default=128)
    heal_parser.add_argument("--steps", type=int, default=500)
    heal_parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    heal_parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    heal_parser.add_argument("--max-length", type=int, default=512)
    heal_parser.add_argument("--max-new-tokens", type=int, default=None)
    heal_parser.add_argument("--temperature", type=float, default=2.0)
    heal_parser.add_argument("--kl-weight", type=float, default=1.0)
    heal_parser.add_argument("--hidden-weight", type=float, default=0.1)
    heal_parser.add_argument("--hidden-layer-stride", type=int, default=4)
    heal_parser.add_argument("--lora-rank", type=int, default=16)
    heal_parser.add_argument("--lora-alpha", type=int, default=None)
    heal_parser.add_argument("--lora-dropout", type=float, default=0.05)
    heal_parser.add_argument(
        "--target-modules",
        default="gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module suffixes.",
    )
    heal_parser.add_argument("--warmup-ratio", type=float, default=0.03)
    heal_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    heal_parser.add_argument("--dtype", default="bfloat16")
    heal_parser.add_argument("--seed", type=int, default=13)
    heal_parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable student gradient checkpointing.",
    )
    heal_parser.add_argument(
        "--allow-invalid-teacher",
        action="store_true",
        help="Keep non-empty teacher answers that fail capability verification.",
    )
    heal_parser.add_argument(
        "--output",
        default=None,
        help="Output directory for adapter and training reports.",
    )
    heal_parser.add_argument(
        "--override",
        action="store_true",
        help="Replace an existing heal output directory.",
    )
    heal_parser.set_defaults(func=cmd_heal)

    certify_parser = subparsers.add_parser(
        "certify",
        help="Certify a healed Qwen2 carve against its teacher.",
    )
    certify_parser.add_argument("--config", required=True, help="Capability YAML.")
    certify_parser.add_argument("--carved", required=True, help="Carved model directory.")
    certify_parser.add_argument("--adapter", required=True, help="HEAL adapter directory.")
    certify_parser.add_argument(
        "--teacher",
        required=True,
        help="Teacher model relative to --model-dir or machine model_dir.",
    )
    certify_parser.add_argument(
        "--model-dir",
        default=None,
        help="Override the machine config model_dir used to resolve --teacher.",
    )
    certify_parser.add_argument(
        "--split",
        default=None,
        help="Held-out probe split. Defaults to capability eval_split.",
    )
    certify_parser.add_argument(
        "--num-examples",
        type=int,
        default=300,
        help="Probe examples. Use 0 for the full split.",
    )
    certify_parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Accepted metric gap. Defaults to capability epsilon.",
    )
    certify_parser.add_argument("--max-length", type=int, default=512)
    certify_parser.add_argument("--max-new-tokens", type=int, default=None)
    certify_parser.add_argument("--temperature", type=float, default=1.0)
    certify_parser.add_argument("--hidden-layer-stride", type=int, default=4)
    certify_parser.add_argument("--scope-layer", type=int, default=4)
    certify_parser.add_argument("--scope-components", type=int, default=32)
    certify_parser.add_argument("--scope-regularization", type=float, default=1.0e-4)
    certify_parser.add_argument("--scope-percentile", type=float, default=95.0)
    certify_parser.add_argument(
        "--scope-split",
        default=None,
        help="Unlabeled scope calibration split. Defaults to capability train_split.",
    )
    certify_parser.add_argument(
        "--scope-num-examples",
        type=int,
        default=300,
        help="Scope calibration examples. Use 0 for the full split.",
    )
    certify_parser.add_argument("--dtype", default="bfloat16")
    certify_parser.add_argument("--output", default=None)
    certify_parser.add_argument(
        "--override",
        action="store_true",
        help="Replace an existing certificate output directory.",
    )
    certify_parser.set_defaults(func=cmd_certify)

    gate_eval_parser = subparsers.add_parser(
        "gate-eval",
        help="Simulate scope routing from a certificate probe artifact.",
    )
    gate_eval_parser.add_argument("--config", required=True, help="Capability YAML.")
    gate_eval_parser.add_argument(
        "--certificate",
        required=True,
        help="Certificate directory containing certificate.json and probe_records.json.",
    )
    gate_eval_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the certificate Mahalanobis threshold.",
    )
    gate_eval_parser.add_argument("--output", default=None)
    gate_eval_parser.set_defaults(func=cmd_gate_eval)

    train_parser = subparsers.add_parser(
        "train-lora",
        help="Train a gold-supervised LoRA specialist for one capability.",
    )
    train_parser.add_argument("--config", required=True, help="Capability YAML.")
    train_parser.add_argument(
        "--training-config",
        default="configs/train/lora_default.yaml",
        help="LoRA training defaults YAML.",
    )
    train_parser.add_argument(
        "--model",
        default="Qwen2.5-1.5B-Instruct",
        help="Base model directory relative to --model-dir or machine model_dir.",
    )
    train_parser.add_argument(
        "--model-dir",
        default=None,
        help="Override the machine config model_dir used to resolve --model.",
    )
    train_parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to train on. Defaults to capability train_split.",
    )
    train_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum training examples. Use 0 for the full split.",
    )
    train_parser.add_argument("--output", default=None, help="Adapter output directory.")
    train_parser.add_argument("--rank", type=int, choices=(4, 8, 16, 32, 64), default=None)
    train_parser.add_argument("--alpha", type=int, default=None)
    train_parser.add_argument("--dropout", type=float, default=None)
    train_parser.add_argument(
        "--target-modules",
        default=None,
        help="Comma-separated LoRA targets, or the alias qkvo/all.",
    )
    train_parser.add_argument("--learning-rate", type=float, default=None)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional optimizer-step cap, primarily for smoke runs.",
    )
    train_parser.add_argument("--max-length", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=None
    )
    train_parser.add_argument("--warmup-ratio", type=float, default=None)
    train_parser.add_argument("--max-grad-norm", type=float, default=None)
    train_parser.add_argument("--dtype", default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--log-steps", type=int, default=None)
    train_parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable model gradient checkpointing.",
    )
    train_parser.add_argument(
        "--override",
        action="store_true",
        help="Replace an existing adapter output directory.",
    )
    train_parser.set_defaults(func=cmd_train_lora)

    for name in (
        "gen-teacher",
        "build-sft",
        "export",
        "serve",
        "report",
    ):
        stub = subparsers.add_parser(name, help="{0} (not implemented in Phase 0)".format(name))
        stub.set_defaults(func=cmd_not_implemented)

    return parser


def cmd_list_capabilities(args: argparse.Namespace) -> int:
    rows = list(
        iter_capabilities(
            config_dir=Path(args.config_dir),
            machine_path=args.machine,
        )
    )
    if not rows:
        print("No capabilities found.")
        return 0

    print(
        "{:<14} {:<16} {:<11} {:<42} {:<42}".format(
            "capability", "metric", "eval_split", "dataset_path", "model_dir"
        )
    )
    print("-" * 132)
    for row in rows:
        print(
            "{:<14} {:<16} {:<11} {:<42} {:<42}".format(
                row.id,
                row.config.metric_name,
                row.config.eval_split,
                row.config.dataset_path,
                row.machine.model_dir,
            )
        )
    print("\ndata_dir: {0}".format(rows[0].machine.data_dir))
    print("machine: {0}".format(rows[0].machine.name))
    return 0


def cmd_list_datasets(args: argparse.Namespace) -> int:
    from idn.data.dataset_info import (
        collect_dataset_infos,
        dataset_infos_to_json,
        format_dataset_infos,
    )

    rows = list(
        iter_capabilities(
            config_dir=Path(args.config_dir),
            machine_path=args.machine,
        )
    )
    infos = collect_dataset_infos(rows, inspect_local=not args.no_inspect)
    if args.json:
        print(dataset_infos_to_json(infos))
    else:
        print(format_dataset_infos(infos))
    return 0


def cmd_visualize_embeddings(args: argparse.Namespace) -> int:
    from idn.analysis.embedding_viz import visualize_dataset_embeddings

    split = None if args.split == "configured" else args.split
    result = visualize_dataset_embeddings(
        config_dir=Path(args.config_dir),
        machine_path=args.machine,
        split=split,
        samples_per_dataset=args.samples_per_dataset,
        output_dir=Path(args.output_dir),
        projection=args.projection,
        seed=args.seed,
        max_chars=args.max_chars,
        perplexity=args.perplexity,
    )
    print("Embedding visualization complete.")
    print("samples: {0}".format(result["samples"]))
    print("datasets: {0}".format(", ".join(result["datasets"])))
    print("silhouette: {0}".format(result["diagnostics"]["silhouette"]))
    print(
        "linear_cv_accuracy_mean: {0}".format(
            result["diagnostics"]["linear_cv_accuracy_mean"]
        )
    )
    print("plot: {0}".format(result["outputs"]["plot"]))
    print("coordinates: {0}".format(result["outputs"]["csv"]))
    print("diagnostics: {0}".format(result["outputs"]["json"]))
    return 0


def cmd_fit_routing(args: argparse.Namespace) -> int:
    from idn.routing.space import fit_routing_from_configs

    machine = load_machine_config(args.machine)
    model_path = resolve_relative_dir(
        base_dir=args.model_dir or machine.model_dir,
        relative_dir=args.model,
        field="--model",
    )
    output_path = Path(args.output)
    report_path = Path(args.report)
    if not args.override and output_path.exists() and report_path.exists():
        print("Routing artifacts already exist; skipping fit.")
        print("space: {0}".format(output_path))
        print("report: {0}".format(report_path))
        return 0
    report = fit_routing_from_configs(
        config_dir=Path(args.config_dir),
        machine_path=args.machine,
        model_path=model_path,
        output_path=output_path,
        report_path=report_path,
        dtype=args.dtype,
        device=machine.device,
        local_files_only=machine.local_files_only,
        samples_per_capability=args.samples_per_capability,
        eval_samples_per_capability=args.eval_samples_per_capability,
        components=args.components,
        shrinkage=args.shrinkage,
        temperature=args.temperature,
        ood_percentile=args.ood_percentile,
        max_chars=args.max_chars,
        max_length=args.max_length,
        batch_size=args.batch_size or machine.eval_batch_size,
        seed=args.seed,
        override=args.override,
        progress=print_routing_progress,
    )
    health = report["health"]
    print("Routing fit complete.")
    print("representation: base_model_mean_pooled_hidden_state")
    print("model: {0}".format(model_path))
    print("embedding_size: {0}".format(report["embedding_size"]))
    print("dimensions: {0}".format(report["dimensions"]))
    print("held_out_accuracy: {0}".format(health["accuracy"]))
    print("coverage: {0}".format(health["coverage"]))
    print("ood_rejection_rate: {0}".format(health["ood_rejection_rate"]))
    print("gate_passed: {0}".format(report["notes"]["gate_passed"]))
    print("space: {0}".format(output_path))
    print("report: {0}".format(report_path))
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    from idn.routing.router import SemanticEmbedder, SemanticRouter
    from idn.routing.space import RoutingSpace

    machine = load_machine_config(args.machine)
    space = RoutingSpace.load(Path(args.space))
    if args.model:
        model_path = resolve_relative_dir(
            base_dir=args.model_dir or machine.model_dir,
            relative_dir=args.model,
            field="--model",
        )
    else:
        model_path = str(space.metadata.get("model_path", ""))
        if not model_path:
            raise RoutingError("routing space does not record a model path; pass --model")
    embedder = SemanticEmbedder(
        model_path=model_path,
        dtype=args.dtype or str(space.metadata.get("dtype", "bfloat16")),
        device=machine.device,
        local_files_only=machine.local_files_only,
        max_length=args.max_length or int(space.metadata.get("max_length", 512)),
        batch_size=1,
    ).load()
    decision = SemanticRouter(space=space, embedder=embedder).route(args.input)
    print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_visualize_activations(args: argparse.Namespace) -> int:
    from idn.analysis.activation_viz import visualize_model_activations

    machine = load_machine_config(args.machine)
    model_path = resolve_relative_dir(
        base_dir=machine.model_dir,
        relative_dir=args.model,
        field="--model",
    )
    capabilities = (
        [item.strip() for item in args.capabilities.split(",") if item.strip()]
        if args.capabilities
        else None
    )
    split = None if args.split == "configured" else args.split
    result = visualize_model_activations(
        model_path=model_path,
        config_dir=Path(args.config_dir),
        machine_path=args.machine,
        adapter_path=args.adapter,
        dtype=args.dtype,
        device=machine.device,
        local_files_only=machine.local_files_only,
        capabilities=capabilities,
        split=split,
        samples_per_capability=args.samples_per_capability,
        seed=args.seed,
        max_chars=args.max_chars,
        output_dir=Path(args.output_dir),
        max_length=args.max_length,
        neuron_buckets=args.neuron_buckets,
        top_k=args.top_k,
    )
    print("Activation visualization complete.")
    print("model: {0}".format(result["model_path"]))
    print("capabilities: {0}".format(result["num_capabilities"]))
    print("prompts: {0}".format(result["num_prompts"]))
    print("layers: {0}".format(result["num_layers"]))
    print("hidden_size: {0}".format(result["hidden_size"]))
    print("html: {0}".format(result["outputs"]["html"]))
    print("json: {0}".format(result["outputs"]["json"]))
    print("top_neurons_csv: {0}".format(result["outputs"]["top_neurons_csv"]))
    return 0


def cmd_not_implemented(args: argparse.Namespace) -> int:
    raise SystemExit(
        "Command '{0}' is planned but not implemented in Phase 0.".format(args.command)
    )


def cmd_train_lora(args: argparse.Namespace) -> int:
    resolved = load_resolved_config(Path(args.config), args.machine)
    spec = spec_from_config(resolved.capability)
    if spec is None:
        raise SystemExit(
            "Capability '{0}' is not implemented yet.".format(resolved.capability.id)
        )
    defaults = load_lora_training_defaults(Path(args.training_config))
    model_base_dir = args.model_dir or resolved.machine.model_dir
    model_path = Path(
        resolve_relative_dir(
            base_dir=model_base_dir,
            relative_dir=args.model,
            field="--model",
        )
    )
    split = args.split or resolved.capability.train_split
    limit = None if args.limit in (None, 0) else args.limit
    rank = defaults.rank if args.rank is None else args.rank
    alpha = defaults.alpha if args.alpha is None else args.alpha
    dropout = defaults.dropout if args.dropout is None else args.dropout
    target_modules = (
        defaults.target_modules
        if args.target_modules is None
        else normalize_target_modules(args.target_modules)
    )
    learning_rate = (
        defaults.learning_rate
        if args.learning_rate is None
        else args.learning_rate
    )
    epochs = defaults.epochs if args.epochs is None else args.epochs
    max_length = defaults.max_length if args.max_length is None else args.max_length
    batch_size = (
        resolved.machine.train_batch_size
        if args.batch_size is None
        else args.batch_size
    )
    gradient_accumulation_steps = (
        defaults.gradient_accumulation_steps
        if args.gradient_accumulation_steps is None
        else args.gradient_accumulation_steps
    )
    warmup_ratio = (
        defaults.warmup_ratio if args.warmup_ratio is None else args.warmup_ratio
    )
    max_grad_norm = (
        defaults.max_grad_norm if args.max_grad_norm is None else args.max_grad_norm
    )
    dtype = defaults.dtype if args.dtype is None else args.dtype
    seed = defaults.seed if args.seed is None else args.seed
    log_steps = defaults.log_steps if args.log_steps is None else args.log_steps
    gradient_checkpointing = (
        defaults.gradient_checkpointing and not args.no_gradient_checkpointing
    )
    output_path = Path(args.output) if args.output else default_train_lora_output_path(
        artifact_dir=DEFAULT_ARTIFACT_DIR,
        capability_id=resolved.capability.id,
        rank=rank,
    )

    if output_path.exists() and not args.override:
        print("LoRA adapter already exists; skipping training.")
        print("adapter: {0}".format(output_path))
        return 0

    print("Starting LoRA specialist training.")
    print("capability: {0}".format(resolved.capability.id))
    print("split: {0}".format(split))
    print("examples: {0}".format("full" if limit is None else limit))
    print("target_source: gold")
    print("model: {0}".format(model_path))
    print("rank: {0}".format(rank))
    print("target_modules: {0}".format(",".join(target_modules)))
    print("output: {0}".format(output_path))
    result = train_lora(
        base_model_path=model_path,
        spec=spec,
        split=split,
        output_dir=output_path,
        limit=limit,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        learning_rate=learning_rate,
        scheduler=defaults.scheduler,
        warmup_ratio=warmup_ratio,
        epochs=epochs,
        max_steps=args.max_steps,
        max_length=max_length,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        dtype=dtype,
        device=resolved.machine.device,
        local_files_only=resolved.machine.local_files_only,
        seed=seed,
        log_steps=log_steps,
        gradient_checkpointing=gradient_checkpointing,
        override=args.override,
        progress=print_train_lora_progress,
    )
    print("LoRA specialist training complete.")
    print("adapter: {0}".format(result.adapter_dir))
    print("train_meta: {0}".format(output_path / "train_meta.json"))
    print("training_log: {0}".format(output_path / "training_log.json"))
    return 0


def cmd_locate(args: argparse.Namespace) -> int:
    resolved = load_resolved_config(Path(args.config), args.machine)
    spec = spec_from_config(resolved.capability)
    if spec is None:
        raise SystemExit(
            "Capability '{0}' is not implemented yet.".format(resolved.capability.id)
        )
    model_base_dir = args.model_dir or resolved.machine.model_dir
    model_path = Path(
        resolve_relative_dir(
            base_dir=model_base_dir,
            relative_dir=args.model,
            field="--model",
        )
    )
    split = args.split or resolved.capability.train_split
    output_path = Path(args.output) if args.output else default_locate_output_path(
        artifact_dir=DEFAULT_ARTIFACT_DIR,
        capability_id=resolved.capability.id,
        split=split,
        model_path=model_path,
        num_examples=args.num_examples,
        dtype=args.dtype,
    )
    print("Starting locate.")
    print("capability: {0}".format(resolved.capability.id))
    print("split: {0}".format(split))
    print("examples: {0}".format(args.num_examples))
    print("model: {0}".format(model_path))
    print("output: {0}".format(output_path))
    result = locate_qwen2_units(
        model_path=model_path,
        spec=spec,
        split=split,
        limit=args.num_examples,
        output_dir=output_path,
        dtype=args.dtype,
        device=resolved.machine.device,
        local_files_only=resolved.machine.local_files_only,
        max_length=args.max_length,
        progress=print_locate_progress,
    )
    print("Locate complete.")
    print("scores: {0}".format(len(result.scores)))
    print("scores_json: {0}".format(output_path / "scores.json"))
    print("report_json: {0}".format(output_path / "report.json"))
    return 0


def cmd_carve(args: argparse.Namespace) -> int:
    machine = load_machine_config(args.machine)
    model_path = Path(
        resolve_relative_dir(
            base_dir=machine.model_dir,
            relative_dir=args.model,
            field="--model",
        )
    )
    keep_ratio = validate_keep_ratio(args.keep_ratio)
    attention_keep_ratio = (
        validate_keep_ratio(args.attention_keep_ratio)
        if args.attention_keep_ratio is not None
        else None
    )
    ffn_keep_ratio = (
        validate_keep_ratio(args.ffn_keep_ratio)
        if args.ffn_keep_ratio is not None
        else None
    )
    output_path = Path(args.output) if args.output else default_carve_output_path(
        artifact_dir=DEFAULT_ARTIFACT_DIR,
        model_path=model_path,
        keep_ratio=keep_ratio,
        attention_keep_ratio=attention_keep_ratio,
        ffn_keep_ratio=ffn_keep_ratio,
    )
    print("Starting carve.")
    print("model: {0}".format(model_path))
    print("output: {0}".format(output_path))
    print("keep_ratio: {0}".format(keep_ratio))
    print(
        "attention_keep_ratio: {0}".format(
            attention_keep_ratio if attention_keep_ratio is not None else 1.0
        )
    )
    print(
        "ffn_keep_ratio: {0}".format(
            ffn_keep_ratio if ffn_keep_ratio is not None else keep_ratio
        )
    )
    manifest = carve_qwen2_homogeneous(
        model_path=model_path,
        output_dir=output_path,
        keep_ratio=keep_ratio,
        attention_keep_ratio=attention_keep_ratio,
        ffn_keep_ratio=ffn_keep_ratio,
        scores_path=Path(args.scores) if args.scores else None,
        override=args.override,
        progress=print_progress,
    )
    print("Carve artifact created.")
    print("teacher_hash: {0}".format(manifest.teacher_hash))
    print("units: {0}".format(manifest.num_units))
    return 0


def cmd_heal(args: argparse.Namespace) -> int:
    resolved = load_resolved_config(Path(args.config), args.machine)
    spec = spec_from_config(resolved.capability)
    if spec is None:
        raise SystemExit(
            "Capability '{0}' is not implemented yet.".format(resolved.capability.id)
        )
    model_base_dir = args.model_dir or resolved.machine.model_dir
    teacher_path = Path(
        resolve_relative_dir(
            base_dir=model_base_dir,
            relative_dir=args.teacher,
            field="--teacher",
        )
    )
    carved_path = Path(args.carved)
    split = args.split or resolved.capability.train_split
    target_modules = tuple(
        item.strip() for item in args.target_modules.split(",") if item.strip()
    )
    if not target_modules:
        raise HealError("--target-modules must contain at least one module")
    output_path = Path(args.output) if args.output else default_heal_output_path(
        artifact_dir=DEFAULT_ARTIFACT_DIR,
        capability_id=resolved.capability.id,
        carved_model_path=carved_path,
        steps=args.steps,
        lora_rank=args.lora_rank,
    )
    max_new_tokens = args.max_new_tokens or resolved.capability.max_new_tokens
    print("Starting heal.")
    print("capability: {0}".format(resolved.capability.id))
    print("split: {0}".format(split))
    print("examples: {0}".format(args.num_examples))
    print("steps: {0}".format(args.steps))
    print("teacher: {0}".format(teacher_path))
    print("carved: {0}".format(carved_path))
    print("output: {0}".format(output_path))
    result = heal_qwen2_lora(
        carved_model_path=carved_path,
        teacher_model_path=teacher_path,
        spec=spec,
        split=split,
        num_examples=args.num_examples,
        output_dir=output_path,
        steps=args.steps,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        kl_weight=args.kl_weight,
        hidden_weight=args.hidden_weight,
        hidden_layer_stride=args.hidden_layer_stride,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        dtype=args.dtype,
        device=resolved.machine.device,
        local_files_only=resolved.machine.local_files_only,
        seed=args.seed,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        require_valid_teacher=not args.allow_invalid_teacher,
        override=args.override,
        progress=print_heal_progress,
    )
    print("Heal complete.")
    print("adapter: {0}".format(result.adapter_dir))
    print("report: {0}".format(output_path / "report.json"))
    print("training_log: {0}".format(output_path / "training_log.json"))
    return 0


def cmd_certify(args: argparse.Namespace) -> int:
    resolved = load_resolved_config(Path(args.config), args.machine)
    spec = spec_from_config(resolved.capability)
    if spec is None:
        raise SystemExit(
            "Capability '{0}' is not implemented yet.".format(resolved.capability.id)
        )
    model_base_dir = args.model_dir or resolved.machine.model_dir
    teacher_path = Path(
        resolve_relative_dir(
            base_dir=model_base_dir,
            relative_dir=args.teacher,
            field="--teacher",
        )
    )
    carved_path = Path(args.carved)
    adapter_path = Path(args.adapter)
    split = args.split or resolved.capability.eval_split
    limit = None if args.num_examples == 0 else args.num_examples
    epsilon = resolved.capability.epsilon if args.epsilon is None else args.epsilon
    max_new_tokens = args.max_new_tokens or resolved.capability.max_new_tokens
    scope_split = args.scope_split or resolved.capability.train_split
    scope_limit = None if args.scope_num_examples == 0 else args.scope_num_examples
    output_path = Path(args.output) if args.output else default_certify_output_path(
        artifact_dir=DEFAULT_ARTIFACT_DIR,
        capability_id=resolved.capability.id,
        adapter_path=adapter_path,
        num_examples=limit,
    )
    print("Starting certification.")
    print("capability: {0}".format(resolved.capability.id))
    print("split: {0}".format(split))
    print("examples: {0}".format("full" if limit is None else limit))
    print("epsilon: {0}".format(epsilon))
    print(
        "scope_calibration: {0} ({1})".format(
            scope_split,
            "full" if scope_limit is None else scope_limit,
        )
    )
    print("teacher: {0}".format(teacher_path))
    print("carved: {0}".format(carved_path))
    print("adapter: {0}".format(adapter_path))
    print("output: {0}".format(output_path))
    result = certify_qwen2(
        carved_model_path=carved_path,
        adapter_path=adapter_path,
        teacher_model_path=teacher_path,
        spec=spec,
        split=split,
        limit=limit,
        output_dir=output_path,
        epsilon=epsilon,
        max_length=args.max_length,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        hidden_layer_stride=args.hidden_layer_stride,
        scope_layer=args.scope_layer,
        scope_components=args.scope_components,
        scope_regularization=args.scope_regularization,
        scope_percentile=args.scope_percentile,
        scope_split=scope_split,
        scope_limit=scope_limit,
        dtype=args.dtype,
        device=resolved.machine.device,
        local_files_only=resolved.machine.local_files_only,
        override=args.override,
        progress=print_certify_progress,
    )
    quality = result.certificate["quality"]
    print("Certification complete.")
    print("status: {0}".format(result.certificate["status"]))
    print(
        "teacher_{0}: {1}".format(
            spec.metric_name,
            quality["teacher_metrics"][spec.metric_name],
        )
    )
    print(
        "student_{0}: {1}".format(
            spec.metric_name,
            quality["student_metrics"][spec.metric_name],
        )
    )
    print("metric_gap: {0}".format(quality["metric_gap"]))
    print("certificate: {0}".format(result.certificate_path))
    return 0


def cmd_gate_eval(args: argparse.Namespace) -> int:
    resolved = load_resolved_config(Path(args.config), args.machine)
    spec = spec_from_config(resolved.capability)
    if spec is None:
        raise SystemExit(
            "Capability '{0}' is not implemented yet.".format(resolved.capability.id)
        )
    certificate_dir = Path(args.certificate)
    output_path = Path(args.output) if args.output else default_gate_eval_output_path(
        artifact_dir=DEFAULT_ARTIFACT_DIR,
        capability_id=resolved.capability.id,
        certificate_dir=certificate_dir,
    )
    result = evaluate_certificate_gate(
        spec=spec,
        certificate_dir=certificate_dir,
        output_path=output_path,
        threshold=args.threshold,
    )
    print("Gate evaluation complete.")
    print("local_coverage: {0}".format(result["local_coverage"]))
    print("fallback_rate: {0}".format(result["fallback_rate"]))
    print(
        "routed_{0}: {1}".format(
            spec.metric_name,
            result["routed_metrics"][spec.metric_name],
        )
    )
    print("output: {0}".format(output_path))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    resolved = load_resolved_config(Path(args.config), args.machine)
    spec = spec_from_config(resolved.capability)
    if spec is None:
        raise SystemExit(
            "Capability '{0}' is not implemented yet.".format(resolved.capability.id)
        )

    split = args.split or resolved.capability.eval_split
    if args.limit == 0:
        limit = None
    elif args.limit is not None:
        limit = args.limit
    else:
        limit = resolved.capability.eval_limit
    model_base_dir = args.model_dir or resolved.machine.model_dir
    model_path = resolve_relative_dir(
        base_dir=model_base_dir,
        relative_dir=args.model,
        field="--model",
    )
    dtype = args.dtype
    output_path = Path(args.output) if args.output else default_eval_output_path(
        artifact_dir=DEFAULT_ARTIFACT_DIR,
        capability_id=resolved.capability.id,
        split=split,
        model_path=model_path,
        adapter_path=args.adapter,
        limit=limit,
        dtype=dtype,
    )
    if (
        output_path.exists()
        and not args.override
        and not is_empty_eval_output(output_path)
    ):
        if not confirm_overwrite(output_path):
            print("Aborted; output already exists: {0}".format(output_path))
            return 1

    run_eval(
        spec=spec,
        split=split,
        limit=limit,
        base_model_path=model_path,
        adapter_path=args.adapter,
        output_path=output_path,
        dtype=dtype,
        device=resolved.machine.device,
        max_new_tokens=args.max_new_tokens or resolved.capability.max_new_tokens,
        local_files_only=resolved.machine.local_files_only,
        batch_size=args.batch_size or resolved.machine.eval_batch_size,
        verbose=args.verbose,
    )
    return 0


def default_eval_output_path(
    artifact_dir: Path,
    capability_id: str,
    split: str,
    model_path: str,
    adapter_path: Optional[str],
    limit: Optional[int],
    dtype: str,
) -> Path:
    model_name = safe_slug(Path(model_path).name or model_path)
    adapter_name = safe_slug(Path(adapter_path).name) if adapter_path else "base"
    limit_name = "full" if limit is None else str(limit)
    filename = "{capability}_{split}_{model}_{adapter}_limit{limit}_{dtype}.json".format(
        capability=safe_slug(capability_id),
        split=safe_slug(split),
        model=model_name,
        adapter=adapter_name,
        limit=limit_name,
        dtype=safe_slug(dtype),
    )
    return artifact_dir / "eval" / filename


def default_carve_output_path(
    artifact_dir: Path,
    model_path: Path,
    keep_ratio: float,
    attention_keep_ratio: Optional[float] = None,
    ffn_keep_ratio: Optional[float] = None,
) -> Path:
    model_name = safe_slug(Path(model_path).name or str(model_path))
    ratio_name = format_keep_ratio(keep_ratio)
    dirname = "carve_r{0}".format(ratio_name)
    if attention_keep_ratio is not None or ffn_keep_ratio is not None:
        attention_name = format_keep_ratio(
            1.0 if attention_keep_ratio is None else attention_keep_ratio
        )
        ffn_name = format_keep_ratio(
            keep_ratio if ffn_keep_ratio is None else ffn_keep_ratio
        )
        dirname += "_attn{0}_ffn{1}".format(attention_name, ffn_name)
    return artifact_dir / "carving" / model_name / dirname


def default_locate_output_path(
    artifact_dir: Path,
    capability_id: str,
    split: str,
    model_path: Path,
    num_examples: int,
    dtype: str,
) -> Path:
    model_name = safe_slug(Path(model_path).name or str(model_path))
    dirname = "{split}_{model}_n{num_examples}_{dtype}".format(
        split=safe_slug(split),
        model=model_name,
        num_examples=num_examples,
        dtype=safe_slug(dtype),
    )
    return artifact_dir / "locate" / safe_slug(capability_id) / dirname


def default_heal_output_path(
    artifact_dir: Path,
    capability_id: str,
    carved_model_path: Path,
    steps: int,
    lora_rank: int,
) -> Path:
    carved_name = safe_slug(Path(carved_model_path).name or str(carved_model_path))
    dirname = "{0}_lora_r{1}_s{2}".format(carved_name, lora_rank, steps)
    return artifact_dir / "heal" / safe_slug(capability_id) / dirname


def default_train_lora_output_path(
    artifact_dir: Path,
    capability_id: str,
    rank: int,
) -> Path:
    return artifact_dir / "adapters" / safe_slug(capability_id) / "r{0}".format(rank)


def default_certify_output_path(
    artifact_dir: Path,
    capability_id: str,
    adapter_path: Path,
    num_examples: Optional[int],
) -> Path:
    adapter_path = Path(adapter_path)
    identity = adapter_path.parent.name if adapter_path.name == "adapter" else adapter_path.name
    count = "full" if num_examples is None else "n{0}".format(num_examples)
    dirname = "{0}_{1}".format(safe_slug(identity), count)
    return artifact_dir / "certify" / safe_slug(capability_id) / dirname


def default_gate_eval_output_path(
    artifact_dir: Path,
    capability_id: str,
    certificate_dir: Path,
) -> Path:
    identity = safe_slug(Path(certificate_dir).name or str(certificate_dir))
    return artifact_dir / "gate_eval" / safe_slug(capability_id) / "{0}.json".format(identity)


def format_keep_ratio(keep_ratio: float) -> str:
    ratio = "{0:.6f}".format(float(keep_ratio)).rstrip("0").rstrip(".")
    if "." not in ratio:
        ratio = "{0}.0".format(ratio)
    return ratio


def print_progress(message: str) -> None:
    print("[carve] {0}".format(message), flush=True)


def print_locate_progress(message: str) -> None:
    print("[locate] {0}".format(message), flush=True)


def print_heal_progress(message: str) -> None:
    print("[heal] {0}".format(message), flush=True)


def print_train_lora_progress(message: str) -> None:
    print("[train-lora] {0}".format(message), flush=True)


def print_routing_progress(message: str) -> None:
    print("[routing] {0}".format(message), flush=True)


def print_certify_progress(message: str) -> None:
    print("[certify] {0}".format(message), flush=True)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    slug = slug.strip("-._")
    return slug or "unknown"


def confirm_overwrite(path: Path) -> bool:
    if not sys.stdin.isatty():
        print(
            "Output already exists: {0}. Re-run with --override to overwrite.".format(path),
            file=sys.stderr,
        )
        return False
    while True:
        answer = input("Output already exists: {0}. Override? [y/n]: ".format(path))
        answer = answer.strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")


def is_empty_eval_output(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("completed_examples") == 0


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        func = getattr(args, "func")
        return int(func(args))
    except ConfigError as exc:
        print("config error: {0}".format(exc), file=sys.stderr)
        return 2
    except CarveError as exc:
        print("carve error: {0}".format(exc), file=sys.stderr)
        return 2
    except LocateError as exc:
        print("locate error: {0}".format(exc), file=sys.stderr)
        return 2
    except HealError as exc:
        print("heal error: {0}".format(exc), file=sys.stderr)
        return 2
    except TrainLoRAError as exc:
        print("train-lora error: {0}".format(exc), file=sys.stderr)
        return 2
    except CertifyError as exc:
        print("certify error: {0}".format(exc), file=sys.stderr)
        return 2
    except GateError as exc:
        print("gate error: {0}".format(exc), file=sys.stderr)
        return 2
    except RoutingError as exc:
        print("routing error: {0}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
