#!/usr/bin/env python3
"""Create paper-ready fallback performance figures and tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


CAPABILITY_LABELS = {
    "cnn_dailymail": "CNN/DM",
    "cola": "CoLA",
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "squad": "SQuAD",
    "sst2": "SST-2",
}

METRIC_LABELS = {
    "cnn_dailymail": "ROUGE-L",
    "cola": "Accuracy",
    "gsm8k": "Exact match",
    "mbpp": "pass@1",
    "squad": "Exact match",
    "sst2": "Accuracy",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot and tabulate an IDN selective fallback report."
    )
    parser.add_argument(
        "--report", default="artifacts/eval/idn_selective_fallback/report.json"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/eval/idn_selective_fallback/paper"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report_path = Path(args.report)
    output_dir = Path(args.output_dir)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not report.get("test") or not report.get("latency"):
        raise SystemExit("report must contain test and clean latency results")

    output_dir.mkdir(parents=True, exist_ok=True)
    system_rows = build_system_rows(report)
    capability_rows = build_capability_rows(report)
    write_csv(output_dir / "system_comparison.csv", system_rows)
    write_csv(output_dir / "capability_comparison.csv", capability_rows)
    write_markdown(
        output_dir / "fallback_performance.md",
        report=report,
        system_rows=system_rows,
        capability_rows=capability_rows,
    )
    plot_path = output_dir / "fallback_performance.png"
    plot_performance(report, plot_path)
    plot_performance(report, output_dir / "fallback_performance.pdf")
    frontier_path = output_dir / "quality_coverage_frontier.png"
    plot_quality_coverage_frontier(report, frontier_path)
    plot_quality_coverage_frontier(
        report, output_dir / "quality_coverage_frontier.pdf"
    )
    print("figure: {0}".format(plot_path))
    print("frontier figure: {0}".format(frontier_path))
    print("table: {0}".format(output_dir / "fallback_performance.md"))
    return 0


def build_system_rows(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    test = report["test"]
    latency = report["latency"]
    teacher_score = float(test["macro_teacher_score"])
    return [
        {
            "system": "Routed 1.5B + LoRA only",
            "macro_score": float(test["macro_local_score"]),
            "delta_vs_7b": float(test["macro_local_score"]) - teacher_score,
            "edge_request_share": 1.0,
            "fallback_share": 0.0,
            "avoided_7b_compute": None,
            "mean_latency_ms": float(latency["local_generation"]["mean_ms"]),
            "p50_latency_ms": float(latency["local_generation"]["p50_ms"]),
            "p95_latency_ms": float(latency["local_generation"]["p95_ms"]),
        },
        {
            "system": "7B only",
            "macro_score": teacher_score,
            "delta_vs_7b": 0.0,
            "edge_request_share": 0.0,
            "fallback_share": 1.0,
            "avoided_7b_compute": 0.0,
            "mean_latency_ms": float(latency["teacher_generation"]["mean_ms"]),
            "p50_latency_ms": float(latency["teacher_generation"]["p50_ms"]),
            "p95_latency_ms": float(latency["teacher_generation"]["p95_ms"]),
        },
        {
            "system": "IDN fallback",
            "macro_score": float(test["macro_hybrid_score"]),
            "delta_vs_7b": float(test["macro_gap_to_teacher"]),
            "edge_request_share": float(test["coverage"]),
            "fallback_share": float(test["fallback_ratio"]),
            "avoided_7b_compute": float(
                latency["cloud_7b_compute_avoided_fraction"]
            ),
            "mean_latency_ms": float(latency["hybrid_end_to_end"]["mean_ms"]),
            "p50_latency_ms": float(latency["hybrid_end_to_end"]["p50_ms"]),
            "p95_latency_ms": float(latency["hybrid_end_to_end"]["p95_ms"]),
        },
    ]


def build_capability_rows(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    eligible = set(report["protocol"]["eligible_local_capabilities"])
    rows = []
    for capability, values in sorted(report["test"]["per_capability"].items()):
        rows.append(
            {
                "capability": CAPABILITY_LABELS.get(capability, capability),
                "metric": METRIC_LABELS.get(capability, "Primary metric"),
                "locally_certified": capability in eligible,
                "edge_coverage": float(values["coverage"]),
                "local_score": float(values["local_score"]),
                "teacher_score": float(values["teacher_score"]),
                "idn_score": float(values["hybrid_score"]),
                "idn_delta_vs_7b": (
                    float(values["hybrid_score"])
                    - float(values["teacher_score"])
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    report: Mapping[str, Any],
    system_rows: Sequence[Mapping[str, Any]],
    capability_rows: Sequence[Mapping[str, Any]],
) -> None:
    test = report["test"]
    latency = report["latency"]
    bootstrap = report["bootstrap_test_gap"]
    lines = [
        "# IDN Selective Fallback Performance",
        "",
        "## System comparison",
        "",
        "| System | Macro score | Delta vs 7B | Edge requests | 7B fallback | Avoided 7B compute | Mean latency | p50 | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in system_rows:
        avoided = (
            "N/A"
            if row["avoided_7b_compute"] is None
            else _percent(row["avoided_7b_compute"])
        )
        lines.append(
            "| {system} | {macro_score:.4f} | {delta_vs_7b:+.4f} | {edge} | {fallback} | {avoided} | {mean:.1f} ms | {p50:.1f} ms | {p95:.1f} ms |".format(
                system=row["system"],
                macro_score=row["macro_score"],
                delta_vs_7b=row["delta_vs_7b"],
                edge=_percent(row["edge_request_share"]),
                fallback=_percent(row["fallback_share"]),
                avoided=avoided,
                mean=row["mean_latency_ms"],
                p50=row["p50_latency_ms"],
                p95=row["p95_latency_ms"],
            )
        )
    lines.extend(
        [
            "",
            "Quality uses the untouched test partition (n={0}). Latency uses clean sequential H100 runs (n={1}, after {2} warmup request).".format(
                test["samples"],
                latency["samples"],
                latency["warmup_examples_excluded"],
            ),
            "The IDN-minus-7B macro gap is {0:+.4f}, with paired bootstrap 95% CI [{1:+.4f}, {2:+.4f}].".format(
                test["macro_gap_to_teacher"],
                bootstrap["ci95_low"],
                bootstrap["ci95_high"],
            ),
            "",
            "## Capability comparison",
            "",
            "| Capability | Metric | Certified local | Edge coverage | Routed local | 7B | IDN fallback | IDN vs 7B |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in capability_rows:
        lines.append(
            "| {capability} | {metric} | {certified} | {coverage} | {local:.4f} | {teacher:.4f} | {idn:.4f} | {delta:+.4f} |".format(
                capability=row["capability"],
                metric=row["metric"],
                certified="Yes" if row["locally_certified"] else "No",
                coverage=_percent(row["edge_coverage"]),
                local=row["local_score"],
                teacher=row["teacher_score"],
                idn=row["idn_score"],
                delta=row["idn_delta_vs_7b"],
            )
        )
    lines.extend(
        [
            "",
            "Request coverage and compute-weighted coverage differ: IDN serves {0} of test requests locally but avoids only {1} of measured 7B GPU time because GSM8K and MBPP dominate the fallback tail.".format(
                _percent(test["coverage"]),
                _percent(latency["cloud_7b_compute_avoided_fraction"]),
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_performance(report: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    test = report["test"]
    latency = report["latency"]
    capability_rows = build_capability_rows(report)
    colors = {"local": "#2869A6", "idn": "#14866D", "teacher": "#BE4D3D"}
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    systems = ["1.5B+LoRA\nonly", "7B only", "IDN\nfallback"]
    scores = [
        test["macro_local_score"],
        test["macro_teacher_score"],
        test["macro_hybrid_score"],
    ]
    bars = axes[0, 0].bar(
        systems,
        scores,
        color=[colors["local"], colors["teacher"], colors["idn"]],
        width=0.62,
    )
    axes[0, 0].set_ylim(max(0.0, min(scores) - 0.05), min(1.0, max(scores) + 0.04))
    axes[0, 0].set_ylabel("Macro capability score")
    axes[0, 0].set_title("A. End-to-end quality (test n={0})".format(test["samples"]))
    for bar, score in zip(bars, scores):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.004,
            "{0:.4f}".format(score),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    local_request = 100.0 * test["coverage"]
    local_compute = 100.0 * latency["cloud_7b_compute_avoided_fraction"]
    bars_y = ["Request count", "7B GPU-time avoided"]
    local_values = [local_request, local_compute]
    remote_values = [100.0 - local_request, 100.0 - local_compute]
    axes[0, 1].barh(bars_y, local_values, color=colors["local"], label="Served/avoided locally")
    axes[0, 1].barh(
        bars_y,
        remote_values,
        left=local_values,
        color="#D98648",
        label="7B fallback/remains",
    )
    for index, value in enumerate(local_values):
        axes[0, 1].text(value / 2, index, "{0:.1f}%".format(value), ha="center", va="center", color="white", fontweight="bold")
    axes[0, 1].set_xlim(0, 100)
    axes[0, 1].set_xlabel("Share (%)")
    axes[0, 1].set_title("B. Request vs compute-weighted coverage")
    axes[0, 1].legend(loc="lower right", fontsize=8)

    labels = [row["capability"] for row in capability_rows]
    positions = np.arange(len(labels))
    width = 0.24
    axes[1, 0].bar(
        positions - width,
        [row["local_score"] for row in capability_rows],
        width,
        label="Routed 1.5B+LoRA",
        color=colors["local"],
    )
    axes[1, 0].bar(
        positions,
        [row["teacher_score"] for row in capability_rows],
        width,
        label="7B",
        color=colors["teacher"],
    )
    axes[1, 0].bar(
        positions + width,
        [row["idn_score"] for row in capability_rows],
        width,
        label="IDN fallback",
        color=colors["idn"],
    )
    axes[1, 0].set_xticks(positions, labels)
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_ylabel("Native primary metric")
    axes[1, 0].set_title("C. Quality by capability")
    axes[1, 0].legend(fontsize=8, ncol=1, loc="upper left")

    latency_labels = ["p50", "Mean", "p95"]
    teacher_latency = [
        latency["teacher_generation"]["p50_ms"],
        latency["teacher_generation"]["mean_ms"],
        latency["teacher_generation"]["p95_ms"],
    ]
    idn_latency = [
        latency["hybrid_end_to_end"]["p50_ms"],
        latency["hybrid_end_to_end"]["mean_ms"],
        latency["hybrid_end_to_end"]["p95_ms"],
    ]
    latency_positions = np.arange(len(latency_labels))
    axes[1, 1].bar(
        latency_positions - 0.18,
        teacher_latency,
        0.36,
        label="7B",
        color=colors["teacher"],
    )
    axes[1, 1].bar(
        latency_positions + 0.18,
        idn_latency,
        0.36,
        label="IDN fallback",
        color=colors["idn"],
    )
    axes[1, 1].set_xticks(latency_positions, latency_labels)
    axes[1, 1].set_ylabel("Latency (ms/request)")
    axes[1, 1].set_title(
        "D. Clean H100 latency (n={0})".format(latency["samples"])
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].text(
        0.02,
        0.96,
        "p50 speedup: {0:.2f}x\nMean speedup: {1:.2f}x".format(
            latency["p50_speedup_vs_teacher"],
            latency["mean_speedup_vs_teacher"],
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=9,
    )

    figure.suptitle(
        "IDN selective fallback: teacher-level quality at 60% edge request coverage",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_quality_coverage_frontier(
    report: Mapping[str, Any], path: Path
) -> None:
    """Render only the calibration frontier and 7B test baseline."""

    import matplotlib.pyplot as plt

    test = report["test"]
    frontier = report["candidate_frontier"]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    axis.plot(
        [100.0 * float(row["coverage"]) for row in frontier],
        [float(row["macro_hybrid_score"]) for row in frontier],
        color="#2869A6",
        marker="o",
        markersize=4.0,
        linewidth=1.9,
        label="Calibration frontier",
    )
    axis.axhline(
        float(test["macro_teacher_score"]),
        color="#BE4D3D",
        linestyle="--",
        linewidth=1.9,
        label="7B baseline (test)",
    )
    axis.set_xlabel("Requests served by 1.5B + LoRA (%)", fontsize=13)
    axis.set_ylabel("Macro capability score", fontsize=13)
    axis.set_title("Quality-coverage operating frontier", fontsize=15)
    axis.tick_params(axis="both", labelsize=11)
    axis.legend(fontsize=11)
    figure.tight_layout()
    figure.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(figure)


def _percent(value: float) -> str:
    return "{0:.1f}%".format(100.0 * float(value))


if __name__ == "__main__":
    raise SystemExit(main())
