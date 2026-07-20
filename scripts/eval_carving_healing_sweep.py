#!/usr/bin/env python3
"""Heal every completed SST-2 carving point and evaluate the healed models."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="h100")
    parser.add_argument("--config", default="configs/capabilities/sst2.yaml")
    parser.add_argument("--teacher", default="Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--carving-summary",
        default="artifacts/reports/carving_accuracy_sweep/summary.json",
    )
    parser.add_argument("--healing-root", required=True)
    parser.add_argument(
        "--output-dir", default="artifacts/reports/carving_accuracy_sweep"
    )
    parser.add_argument("--num-examples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--hidden-weight", type=float, default=0.1)
    parser.add_argument("--hidden-layer-stride", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="gate_proj,up_proj,down_proj")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary_path = Path(args.carving_summary)
    with summary_path.open("r", encoding="utf-8") as handle:
        carving_summary = json.load(handle)
    carving_rows = sorted(
        carving_summary.get("results", []),
        key=lambda row: float(row["ffn_keep_ratio"]),
    )
    if not carving_rows:
        raise SystemExit("carving summary contains no completed points")

    output_dir = Path(args.output_dir)
    eval_dir = output_dir / "healed_eval"
    healing_root = Path(args.healing_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    healing_root.mkdir(parents=True, exist_ok=True)
    original_parameters = int(carving_rows[-1]["parameter_count"])
    rows: List[Dict[str, Any]] = []

    for carved in carving_rows:
        ratio = float(carved["ffn_keep_ratio"])
        ratio_name = format_ratio(ratio)
        if ratio == 1.0:
            row = build_unhealed_baseline_row(carved)
        else:
            model_path = Path(str(carved["model_path"]))
            heal_path = healing_root / "ffn_r{0}_lora_r{1}_s{2}".format(
                ratio_name, args.lora_rank, args.steps
            )
            ensure_healing(args=args, model_path=model_path, output_path=heal_path)
            eval_path = eval_dir / "sst2_ffn_r{0}_healed.json".format(ratio_name)
            ensure_evaluation(
                args=args,
                model_path=model_path,
                adapter_path=heal_path / "adapter",
                output_path=eval_path,
            )
            row = build_healed_row(
                carved=carved,
                heal_path=heal_path,
                eval_path=eval_path,
                original_parameters=original_parameters,
            )
        rows.append(row)
        write_outputs(output_dir, rows, args, carving_summary)
        print(
            "ratio={0:.1f} carving={1:.2%} healed={2:.2%} gain={3:+.2f} pp".format(
                ratio,
                float(row["carving_accuracy"]),
                float(row["healed_accuracy"]),
                100.0 * float(row["accuracy_gain"]),
            ),
            flush=True,
        )

    print("summary: {0}".format(output_dir / "healing_summary.json"))
    return 0


def ensure_healing(*, args: argparse.Namespace, model_path: Path, output_path: Path) -> None:
    report_path = output_path / "report.json"
    adapter_path = output_path / "adapter" / "adapter_model.safetensors"
    if report_path.exists() and adapter_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("status") == "complete" and healing_matches(report, args):
            print("Reusing healing: {0}".format(output_path), flush=True)
            return
    command = [
        sys.executable,
        "-m",
        "idn.cli",
        "--machine",
        args.machine,
        "heal",
        "--config",
        args.config,
        "--carved",
        str(model_path),
        "--teacher",
        args.teacher,
        "--num-examples",
        str(args.num_examples),
        "--steps",
        str(args.steps),
        "--learning-rate",
        str(args.learning_rate),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--max-length",
        str(args.max_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--kl-weight",
        str(args.kl_weight),
        "--hidden-weight",
        str(args.hidden_weight),
        "--hidden-layer-stride",
        str(args.hidden_layer_stride),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--target-modules",
        args.target_modules,
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--output",
        str(output_path),
    ]
    if output_path.exists():
        command.append("--override")
    run(command)


def healing_matches(report: Mapping[str, Any], args: argparse.Namespace) -> bool:
    lora = report.get("lora", {})
    optimization = report.get("optimization", {})
    return (
        int(report.get("steps", -1)) == args.steps
        and int(report.get("requested_examples", -1)) == args.num_examples
        and int(report.get("seed", -1)) == args.seed
        and int(lora.get("rank", -1)) == args.lora_rank
        and int(lora.get("alpha", -1)) == args.lora_alpha
        and float(optimization.get("learning_rate", -1.0)) == args.learning_rate
        and int(optimization.get("gradient_accumulation_steps", -1))
        == args.gradient_accumulation_steps
    )


def ensure_evaluation(
    *,
    args: argparse.Namespace,
    model_path: Path,
    adapter_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            evaluation = json.load(handle)
        if evaluation.get("status") == "complete":
            print("Reusing evaluation: {0}".format(output_path), flush=True)
            return
    run(
        [
            sys.executable,
            "-m",
            "idn.cli",
            "--machine",
            args.machine,
            "eval",
            "--config",
            args.config,
            "--model-dir",
            str(model_path.parent),
            "--model",
            model_path.name,
            "--adapter",
            str(adapter_path),
            "--limit",
            "0",
            "--batch-size",
            str(args.batch_size),
            "--dtype",
            args.dtype,
            "--output",
            str(output_path),
            "--override",
        ]
    )


def build_healed_row(
    *,
    carved: Mapping[str, Any],
    heal_path: Path,
    eval_path: Path,
    original_parameters: int,
) -> Dict[str, Any]:
    with (heal_path / "report.json").open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    with eval_path.open("r", encoding="utf-8") as handle:
        evaluation = json.load(handle)
    adapter_parameters = int(report["trainable_parameters"])
    base_parameters = int(carved["parameter_count"])
    healed_accuracy = float(evaluation["metrics"]["accuracy"])
    carving_accuracy = float(carved["sst2_accuracy"])
    return {
        "ffn_keep_ratio": float(carved["ffn_keep_ratio"]),
        "parameter_retention": float(carved["parameter_retention"]),
        "base_parameters": base_parameters,
        "adapter_parameters": adapter_parameters,
        "deployed_parameters": base_parameters + adapter_parameters,
        "deployed_parameter_retention": (base_parameters + adapter_parameters)
        / original_parameters,
        "carving_accuracy": carving_accuracy,
        "healed_accuracy": healed_accuracy,
        "accuracy_gain": healed_accuracy - carving_accuracy,
        "samples": int(evaluation["completed_examples"]),
        "healing_applied": True,
        "model_path": str(carved["model_path"]),
        "adapter_path": str(heal_path / "adapter"),
        "heal_report": str(heal_path / "report.json"),
        "eval_artifact": str(eval_path),
    }


def build_unhealed_baseline_row(carved: Mapping[str, Any]) -> Dict[str, Any]:
    parameters = int(carved["parameter_count"])
    accuracy = float(carved["sst2_accuracy"])
    return {
        "ffn_keep_ratio": float(carved["ffn_keep_ratio"]),
        "parameter_retention": float(carved["parameter_retention"]),
        "base_parameters": parameters,
        "adapter_parameters": 0,
        "deployed_parameters": parameters,
        "deployed_parameter_retention": float(carved["parameter_retention"]),
        "carving_accuracy": accuracy,
        "healed_accuracy": accuracy,
        "accuracy_gain": 0.0,
        "samples": int(carved["samples"]),
        "healing_applied": False,
        "model_path": str(carved["model_path"]),
        "adapter_path": None,
        "heal_report": None,
        "eval_artifact": str(carved["eval_artifact"]),
    }


def write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    carving_summary: Mapping[str, Any],
) -> None:
    with (output_dir / "healing_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "artifact_version": 1,
        "experiment": "ffn_only_carving_with_lora_healing_accuracy_sweep",
        "capability": "sst2",
        "carving_strategy": carving_summary.get("strategy", {}),
        "healing_strategy": {
            "method": "teacher_generated_lora_distillation",
            "teacher": args.teacher,
            "num_examples": args.num_examples,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "kl_weight": args.kl_weight,
            "hidden_weight": args.hidden_weight,
            "hidden_layer_stride": args.hidden_layer_stride,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules.split(","),
            "seed": args.seed,
            "full_model_endpoint_is_unhealed": True,
        },
        "results": list(rows),
    }
    (output_dir / "healing_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(command: Sequence[str]) -> None:
    print("Running: {0}".format(" ".join(command)), flush=True)
    subprocess.run(list(command), check=True, cwd=str(ROOT))


def format_ratio(value: float) -> str:
    return ("{0:.3f}".format(value)).rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
