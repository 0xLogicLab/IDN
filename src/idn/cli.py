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
from idn.extraction.locate import LocateError, locate_qwen2_units
from idn.eval.runner import run_eval
from idn.registry import iter_capabilities, spec_from_config


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
        help="Fraction of units to keep.",
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

    for name in (
        "gen-teacher",
        "build-sft",
        "train-lora",
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
    output_path = Path(args.output) if args.output else default_carve_output_path(
        artifact_dir=DEFAULT_ARTIFACT_DIR,
        model_path=model_path,
        keep_ratio=keep_ratio,
    )
    print("Starting carve.")
    print("model: {0}".format(model_path))
    print("output: {0}".format(output_path))
    print("keep_ratio: {0}".format(keep_ratio))
    manifest = carve_qwen2_homogeneous(
        model_path=model_path,
        output_dir=output_path,
        keep_ratio=keep_ratio,
        scores_path=Path(args.scores) if args.scores else None,
        override=args.override,
        progress=print_progress,
    )
    print("Carve artifact created.")
    print("teacher_hash: {0}".format(manifest.teacher_hash))
    print("units: {0}".format(manifest.num_units))
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
        batch_size=resolved.machine.eval_batch_size,
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
) -> Path:
    model_name = safe_slug(Path(model_path).name or str(model_path))
    ratio_name = format_keep_ratio(keep_ratio)
    return artifact_dir / "carving" / model_name / "carve_r{0}".format(ratio_name)


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


def format_keep_ratio(keep_ratio: float) -> str:
    ratio = "{0:.6f}".format(float(keep_ratio)).rstrip("0").rstrip(".")
    if "." not in ratio:
        ratio = "{0}.0".format(ratio)
    return ratio


def print_progress(message: str) -> None:
    print("[carve] {0}".format(message), flush=True)


def print_locate_progress(message: str) -> None:
    print("[locate] {0}".format(message), flush=True)


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


if __name__ == "__main__":
    raise SystemExit(main())
