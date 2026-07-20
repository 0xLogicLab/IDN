#!/usr/bin/env python3
"""Calibrate and evaluate selective 1.5B+LoRA to 7B fallback."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idn.registry import load_registry
from idn.serving.selective import (
    GatePolicy,
    bootstrap_macro_gap,
    calibrate_policy,
    evaluate_policy,
    route_features,
    stratified_calibration_indices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate an IDN local/7B fallback gate on matched predictions."
    )
    parser.add_argument("--machine", default="3090")
    parser.add_argument("--config-dir", default="configs/capabilities")
    parser.add_argument(
        "--student-artifact",
        default="artifacts/eval/idn_mixed_scaled_gsm8k/routed_top1_lora.json",
    )
    parser.add_argument(
        "--teacher-artifact", default="artifacts/eval/idn_mixed_oracle/7b_base.json"
    )
    parser.add_argument("--student-latency-artifact", default=None)
    parser.add_argument("--teacher-latency-artifact", default=None)
    parser.add_argument(
        "--latency-warmup-examples",
        type=int,
        default=1,
        help="Exclude this many leading matched generations as model warmup.",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--quality-tolerance", type=float, default=0.01)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument(
        "--confidence-z",
        type=float,
        default=1.645,
        help="One-sided calibration safety multiplier (1.645 is approximately 95%%).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--local-capabilities",
        default="cnn_dailymail,cola,squad,sst2",
        help=(
            "Comma-separated capabilities certified for local service. "
            "Use 'all' to make every routed capability eligible."
        ),
    )
    parser.add_argument(
        "--output", default="artifacts/eval/idn_selective_fallback/report.json"
    )
    parser.add_argument(
        "--plot", default="artifacts/eval/idn_selective_fallback/results.png"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = load_registry(
        config_dir=Path(args.config_dir), machine_path=args.machine
    )
    student_artifact = _load_artifact(Path(args.student_artifact))
    teacher_artifact = _load_artifact(Path(args.teacher_artifact))
    examples = _build_examples(student_artifact, teacher_artifact, registry)
    eligible_capabilities = _parse_capabilities(args.local_capabilities, registry)
    calibration_indices, test_indices = stratified_calibration_indices(
        examples, fraction=args.calibration_fraction, seed=args.seed
    )
    policy, candidates = calibrate_policy(
        examples,
        calibration_indices,
        quality_tolerance=args.quality_tolerance,
        grid_size=args.grid_size,
        confidence_z=args.confidence_z,
        eligible_capabilities=eligible_capabilities,
    )
    calibration = evaluate_policy(examples, policy, calibration_indices)
    test = evaluate_policy(examples, policy, test_indices)
    full = evaluate_policy(examples, policy)
    bootstrap = bootstrap_macro_gap(
        examples,
        policy,
        test_indices,
        samples=args.bootstrap_samples,
        seed=args.seed + 1,
    )
    latency = _latency_report(args, policy)

    report = {
        "artifact_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "selective_routed_lora_with_7b_fallback",
        "inputs": {
            "student_artifact": str(args.student_artifact),
            "teacher_artifact": str(args.teacher_artifact),
            "student_latency_artifact": args.student_latency_artifact,
            "teacher_latency_artifact": args.teacher_latency_artifact,
        },
        "protocol": {
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "quality_tolerance": args.quality_tolerance,
            "grid_size": args.grid_size,
            "confidence_z": args.confidence_z,
            "eligible_local_capabilities": list(eligible_capabilities),
            "selection": (
                "maximum calibration coverage subject to the one-sided lower "
                "confidence bound for hybrid-minus-teacher macro score being >= "
                "-quality_tolerance"
            ),
            "inference_features": [
                "predicted_capability",
                "minimum_mahalanobis_distance",
                "route_probability",
                "top2_distance_margin",
            ],
            "calibration_examples": len(calibration_indices),
            "test_examples": len(test_indices),
            "calibration_mixed_indices": [
                int(examples[index]["mixed_index"]) for index in calibration_indices
            ],
            "test_mixed_indices": [
                int(examples[index]["mixed_index"]) for index in test_indices
            ],
        },
        "selected_policy": policy.to_dict(),
        "calibration": calibration,
        "test": test,
        "full_dataset_diagnostic": full,
        "bootstrap_test_gap": bootstrap,
        "candidate_frontier": _pareto_rows(candidates),
        "latency": latency,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    output_path = Path(args.output)
    _write_json_atomic(output_path, report)
    _plot_report(report, Path(args.plot))
    _print_summary(report, output_path, Path(args.plot))
    return 0


def _parse_capabilities(
    value: str, registry: Mapping[str, Any]
) -> Sequence[str]:
    if value.strip().lower() == "all":
        return tuple(sorted(registry))
    capabilities = tuple(
        sorted({item.strip() for item in value.split(",") if item.strip()})
    )
    if not capabilities:
        raise SystemExit("--local-capabilities cannot be empty")
    unknown = sorted(set(capabilities) - set(registry))
    if unknown:
        raise SystemExit("unknown local capabilities: {0}".format(unknown))
    return capabilities


def _load_artifact(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit("evaluation artifact does not exist: {0}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") not in (None, "complete"):
        raise SystemExit("evaluation artifact is incomplete: {0}".format(path))
    if not payload.get("predictions"):
        raise SystemExit("evaluation artifact has no predictions: {0}".format(path))
    return payload


def _build_examples(
    student_artifact: Mapping[str, Any],
    teacher_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    student = {
        int(record["mixed_index"]): record
        for record in student_artifact["predictions"]
    }
    teacher = {
        int(record["mixed_index"]): record
        for record in teacher_artifact["predictions"]
    }
    if set(student) != set(teacher):
        raise SystemExit("student and teacher artifacts cover different mixed indices")
    examples = []
    for mixed_index in sorted(student):
        local_record = student[mixed_index]
        teacher_record = teacher[mixed_index]
        capability = str(local_record["capability"])
        if capability != str(teacher_record["capability"]):
            raise SystemExit("capability mismatch at mixed_index {0}".format(mixed_index))
        if capability not in registry or registry[capability].spec is None:
            raise SystemExit("unimplemented capability: {0}".format(capability))
        spec = registry[capability].spec
        local_score = float(
            spec.metric(
                [local_record["parsed_prediction"]], [local_record["reference"]]
            )[registry[capability].config.metric_name]
        )
        teacher_score = float(
            spec.metric(
                [teacher_record["parsed_prediction"]], [teacher_record["reference"]]
            )[registry[capability].config.metric_name]
        )
        example = {
            "mixed_index": mixed_index,
            "capability": capability,
            "local_score": local_score,
            "teacher_score": teacher_score,
        }
        example.update(route_features(local_record))
        examples.append(example)
    return examples


def _pareto_rows(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for candidate in candidates:
        dominated = any(
            float(other["coverage"]) >= float(candidate["coverage"])
            and float(other["macro_hybrid_score"])
            >= float(candidate["macro_hybrid_score"])
            and (
                float(other["coverage"]) > float(candidate["coverage"])
                or float(other["macro_hybrid_score"])
                > float(candidate["macro_hybrid_score"])
            )
            for other in candidates
        )
        if not dominated:
            rows.append(dict(candidate))
    return sorted(rows, key=lambda row: float(row["coverage"]))


def _latency_report(args: argparse.Namespace, policy: GatePolicy) -> Optional[Dict[str, Any]]:
    if not args.student_latency_artifact and not args.teacher_latency_artifact:
        return None
    if not args.student_latency_artifact or not args.teacher_latency_artifact:
        raise SystemExit("both student and teacher latency artifacts are required")
    student_artifact = _load_artifact(Path(args.student_latency_artifact))
    teacher_artifact = _load_artifact(Path(args.teacher_latency_artifact))
    student = {
        int(record["mixed_index"]): record
        for record in student_artifact["predictions"]
    }
    teacher = {
        int(record["mixed_index"]): record
        for record in teacher_artifact["predictions"]
    }
    all_common = sorted(set(student) & set(teacher))
    if not all_common:
        raise SystemExit("latency artifacts have no common mixed indices")
    if args.latency_warmup_examples < 0:
        raise SystemExit("--latency-warmup-examples must be non-negative")
    common = all_common[args.latency_warmup_examples :]
    if not common:
        raise SystemExit("latency warmup excludes every matched example")
    route_ms = 1000.0 * float(student_artifact.get("routing_runtime_seconds", 0.0)) / len(student)
    local_latencies = []
    teacher_latencies = []
    hybrid_latencies = []
    avoided_teacher_ms = 0.0
    total_teacher_ms = 0.0
    accepted = 0
    for mixed_index in common:
        local = student[mixed_index]
        remote = teacher[mixed_index]
        features = route_features(local)
        use_local = policy.accepts(features)
        local_ms = float(local["latency_ms"])
        teacher_ms = float(remote["latency_ms"])
        local_latencies.append(local_ms)
        teacher_latencies.append(teacher_ms)
        hybrid_latencies.append(route_ms + (local_ms if use_local else teacher_ms))
        accepted += int(use_local)
        avoided_teacher_ms += teacher_ms if use_local else 0.0
        total_teacher_ms += teacher_ms
    result = {
        "samples": len(common),
        "warmup_examples_excluded": args.latency_warmup_examples,
        "coverage": accepted / len(common),
        "route_mean_ms": route_ms,
        "local_generation": _latency_stats(local_latencies),
        "teacher_generation": _latency_stats(teacher_latencies),
        "hybrid_end_to_end": _latency_stats(hybrid_latencies),
    }
    result["mean_speedup_vs_teacher"] = (
        result["teacher_generation"]["mean_ms"]
        / result["hybrid_end_to_end"]["mean_ms"]
    )
    result["p50_speedup_vs_teacher"] = (
        result["teacher_generation"]["p50_ms"]
        / result["hybrid_end_to_end"]["p50_ms"]
    )
    result["p95_speedup_vs_teacher"] = (
        result["teacher_generation"]["p95_ms"]
        / result["hybrid_end_to_end"]["p95_ms"]
    )
    result["cloud_7b_compute_avoided_fraction"] = (
        avoided_teacher_ms / total_teacher_ms
    )
    return result


def _latency_stats(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": _percentile(ordered, 50.0),
        "p95_ms": _percentile(ordered, 95.0),
    }


def _percentile(ordered: Sequence[float], percentile: float) -> float:
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _plot_report(report: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    test = report["test"]
    frontier = report["candidate_frontier"]
    axes[0].plot(
        [100.0 * float(row["coverage"]) for row in frontier],
        [float(row["macro_hybrid_score"]) for row in frontier],
        color="#2563A6",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Calibration frontier",
    )
    axes[0].axhline(
        float(test["macro_teacher_score"]),
        color="#B5483A",
        linestyle="--",
        linewidth=1.5,
        label="7B baseline (test)",
    )
    axes[0].scatter(
        [100.0 * float(test["coverage"])],
        [float(test["macro_hybrid_score"])],
        color="#12866F",
        s=70,
        zorder=4,
        label="Selected IDN (test)",
    )
    axes[0].set_xlabel("Requests served by 1.5B + LoRA (%)")
    axes[0].set_ylabel("Macro capability score")
    axes[0].set_title("Quality-coverage operating point")
    axes[0].legend(fontsize=8)

    capabilities = sorted(test["per_capability"])
    accepted = [test["per_capability"][capability]["accepted"] for capability in capabilities]
    fallback = [
        test["per_capability"][capability]["samples"]
        - test["per_capability"][capability]["accepted"]
        for capability in capabilities
    ]
    positions = list(range(len(capabilities)))
    axes[1].barh(positions, accepted, color="#2563A6", label="1.5B + LoRA")
    axes[1].barh(
        positions, fallback, left=accepted, color="#D87945", label="7B fallback"
    )
    axes[1].set_yticks(positions, [capability.replace("_", "\n") for capability in capabilities])
    axes[1].set_xlabel("Test requests")
    axes[1].set_title("Routing outcome by capability")
    axes[1].legend(fontsize=8)

    latency = report.get("latency")
    if latency:
        labels = ["1.5B+LoRA", "IDN", "7B"]
        means = [
            latency["local_generation"]["mean_ms"],
            latency["hybrid_end_to_end"]["mean_ms"],
            latency["teacher_generation"]["mean_ms"],
        ]
        p95 = [
            latency["local_generation"]["p95_ms"],
            latency["hybrid_end_to_end"]["p95_ms"],
            latency["teacher_generation"]["p95_ms"],
        ]
        axes[2].bar(labels, means, color=["#2563A6", "#12866F", "#B5483A"])
        axes[2].scatter(range(3), p95, marker="_", s=250, color="#222222", label="p95")
        axes[2].set_ylabel("Latency (ms/request)")
        axes[2].set_title(
            "Latency: mean {0:.2f}x; p50 {1:.2f}x".format(
                latency["mean_speedup_vs_teacher"],
                latency["p50_speedup_vs_teacher"],
            )
        )
        axes[2].legend(fontsize=8)
    else:
        labels = ["1.5B+LoRA", "IDN", "7B"]
        scores = [
            test["macro_local_score"],
            test["macro_hybrid_score"],
            test["macro_teacher_score"],
        ]
        axes[2].bar(labels, scores, color=["#2563A6", "#12866F", "#B5483A"])
        axes[2].set_ylabel("Macro capability score")
        axes[2].set_title("Final test quality")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=0)
    figure.suptitle("IDN selective serving: calibrated local inference with 7B fallback")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{0}.tmp".format(path.name))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _print_summary(report: Mapping[str, Any], output: Path, plot: Path) -> None:
    test = report["test"]
    bootstrap = report["bootstrap_test_gap"]
    print("selected_policy: {0}".format(report["selected_policy"]))
    print("test_samples: {0}".format(test["samples"]))
    print("edge_coverage: {0:.4f}".format(test["coverage"]))
    print("fallback_ratio: {0:.4f}".format(test["fallback_ratio"]))
    print("idn_macro_score: {0:.4f}".format(test["macro_hybrid_score"]))
    print("7b_macro_score: {0:.4f}".format(test["macro_teacher_score"]))
    print("gap_to_7b: {0:+.4f}".format(test["macro_gap_to_teacher"]))
    print(
        "gap_bootstrap_95ci: [{0:+.4f}, {1:+.4f}]".format(
            bootstrap["ci95_low"], bootstrap["ci95_high"]
        )
    )
    if report.get("latency"):
        print(
            "mean_latency_speedup_vs_7b: {0:.3f}x".format(
                report["latency"]["mean_speedup_vs_teacher"]
            )
        )
    print("report: {0}".format(output))
    print("plot: {0}".format(plot))


if __name__ == "__main__":
    raise SystemExit(main())
