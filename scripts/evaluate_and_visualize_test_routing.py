#!/usr/bin/env python3
"""Evaluate semantic routing on held-out splits and visualize the results."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from idn.routing.router import SemanticEmbedder, SemanticRouter
from idn.routing.space import RoutingSpace, _collect_configured_texts


DISPLAY_NAMES = {
    "cnn_dailymail": "CNN/DailyMail",
    "cola": "CoLA",
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "squad": "SQuAD",
    "sst2": "SST-2",
}
EVAL_SPLITS = {
    "cnn_dailymail": "validation",
    "cola": "validation",
    "gsm8k": "test",
    "mbpp": "test",
    "squad": "validation",
    "sst2": "validation",
}


def _configure_font() -> None:
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})


def _embed(
    texts: Sequence[str],
    space: RoutingSpace,
    cache_path: Path,
    batch_size: int,
    recompute: bool,
) -> np.ndarray:
    if recompute and cache_path.exists():
        cache_path.unlink()
    if cache_path.exists():
        with np.load(str(cache_path), allow_pickle=False) as cached:
            embeddings = np.asarray(cached["embeddings"], dtype=np.float64)
        if len(embeddings) != len(texts):
            raise ValueError("cached test embeddings do not match held-out inputs")
        print("Loaded cached test embeddings: {0}".format(cache_path), flush=True)
        return embeddings
    embedder = SemanticEmbedder(
        model_path=str(space.metadata["model_path"]),
        dtype="bfloat16",
        device="cuda",
        local_files_only=True,
        max_length=int(space.metadata.get("max_length", 512)),
        batch_size=batch_size,
    ).load()
    embeddings = embedder.embed(texts, progress=lambda message: print(message, flush=True))
    np.savez_compressed(str(cache_path), embeddings=np.asarray(embeddings, dtype=np.float32))
    del embedder
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    return np.asarray(embeddings, dtype=np.float64)


def _evaluate(
    space: RoutingSpace,
    embeddings: np.ndarray,
    texts: Sequence[str],
    labels: Sequence[str],
) -> tuple[List[Dict[str, Any]], np.ndarray, List[Dict[str, Any]]]:
    router = SemanticRouter(space=space)
    decisions = router.route_embeddings(embeddings)
    capabilities = list(space.capabilities)
    indices = {capability: index for index, capability in enumerate(capabilities)}
    confusion = np.zeros((len(capabilities), len(capabilities)), dtype=int)
    rows: List[Dict[str, Any]] = []
    for sample_index, (text, label, decision) in enumerate(zip(texts, labels, decisions)):
        confusion[indices[label], indices[decision.capability]] += 1
        rows.append(
            {
                "sample_index": sample_index,
                "true_capability": label,
                "predicted_capability": decision.capability,
                "correct": int(decision.capability == label),
                "accepted": int(decision.in_scope),
                "confidence": decision.confidence,
                "entropy": decision.entropy,
                "minimum_distance": min(decision.distances.values()),
                "route_weights": json.dumps(decision.weights, sort_keys=True),
                "input": text,
            }
        )

    stats: List[Dict[str, Any]] = []
    for capability in capabilities:
        selected = [row for row in rows if row["true_capability"] == capability]
        accepted = [row for row in selected if row["accepted"]]
        stats.append(
            {
                "capability": capability,
                "split": EVAL_SPLITS[capability],
                "samples": len(selected),
                "accuracy": sum(row["correct"] for row in selected) / len(selected),
                "coverage": sum(row["accepted"] for row in selected) / len(selected),
                "selective_accuracy": (
                    sum(row["correct"] for row in accepted) / len(accepted)
                    if accepted
                    else None
                ),
                "mean_confidence": float(np.mean([row["confidence"] for row in selected])),
            }
        )
    return rows, confusion, stats


def _render(
    path: Path,
    capabilities: Sequence[str],
    confusion: np.ndarray,
    stats: Sequence[Mapping[str, Any]],
) -> None:
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_totals, where=row_totals != 0) * 100.0
    overall_accuracy = float(np.trace(confusion) / confusion.sum()) * 100
    total_accepted = sum(float(row["coverage"]) * int(row["samples"]) for row in stats)
    total_samples = sum(int(row["samples"]) for row in stats)
    coverage = total_accepted / total_samples * 100
    accepted_correct = sum(
        float(row["selective_accuracy"] or 0.0)
        * float(row["coverage"])
        * int(row["samples"])
        for row in stats
    )
    selective_accuracy = accepted_correct / total_accepted * 100 if total_accepted else 0.0

    fig, (ax_matrix, ax_bars) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
    image = ax_matrix.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
    labels = [DISPLAY_NAMES[name] for name in capabilities]
    ax_matrix.set_xticks(np.arange(len(capabilities)), labels, rotation=35, ha="right", fontsize=9.5)
    ax_matrix.set_yticks(np.arange(len(capabilities)), labels, fontsize=9.5)
    ax_matrix.set_xlabel("预测 capability", fontsize=11)
    ax_matrix.set_ylabel("真实 capability", fontsize=11)
    ax_matrix.set_title("测试集混淆矩阵", fontsize=14, weight="bold", pad=12)
    for row in range(len(capabilities)):
        for column in range(len(capabilities)):
            count = int(confusion[row, column])
            if count == 0:
                text = "0"
            else:
                text = "{0:.1f}%\n({1})".format(normalized[row, column], count)
            ax_matrix.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if normalized[row, column] > 55 else "#222222",
            )
    colorbar = fig.colorbar(image, ax=ax_matrix, fraction=0.045, pad=0.03)
    colorbar.set_label("行归一化比例（%）", fontsize=9)

    y = np.arange(len(stats))[::-1]
    accuracy = np.asarray([float(row["accuracy"]) * 100 for row in stats])
    class_coverage = np.asarray([float(row["coverage"]) * 100 for row in stats])
    height = 0.32
    accuracy_bars = ax_bars.barh(y + height / 2, accuracy, height, color="#4c78a8", label="Top-1 准确率")
    coverage_bars = ax_bars.barh(y - height / 2, class_coverage, height, color="#f58518", label="OOD 接受率")
    ax_bars.set_yticks(y, [DISPLAY_NAMES[str(row["capability"])] for row in stats], fontsize=10)
    ax_bars.set_xlim(0, 108)
    ax_bars.set_ylim(-0.7, 6.25)
    ax_bars.set_xlabel("比例（%）", fontsize=11)
    ax_bars.set_title("分类别路由效果", fontsize=14, weight="bold", pad=12)
    ax_bars.grid(axis="x", alpha=0.2)
    ax_bars.spines[["top", "right", "left"]].set_visible(False)
    ax_bars.tick_params(axis="y", length=0)
    ax_bars.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.52, 1.01), ncol=2, fontsize=10)
    ax_bars.bar_label(accuracy_bars, fmt="%.1f", padding=2, fontsize=8.5)
    ax_bars.bar_label(coverage_bars, fmt="%.1f", padding=2, fontsize=8.5)

    fig.suptitle(
        "Held-out 测试集语义路由（n={0:,}）\nTop-1 准确率 {1:.2f}%  |  OOD 接受率 {2:.2f}%  |  接受样本准确率 {3:.2f}%".format(
            int(confusion.sum()), overall_accuracy, coverage, selective_accuracy
        ),
        fontsize=15,
        weight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.015,
        "validation：CNN/DailyMail、CoLA、SQuAD、SST-2；test：GSM8K、MBPP。所有样本均未用于拟合路由中心。",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(w_pad=2.8)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="artifacts/routing/space.npz")
    parser.add_argument("--report", default="artifacts/routing/report.json")
    parser.add_argument("--config-dir", default="configs/capabilities")
    parser.add_argument("--machine", default="3090")
    parser.add_argument("--samples-per-capability", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/analysis/routing_test_eval")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    _configure_font()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    space = RoutingSpace.load(Path(args.space))
    seed = int(space.metadata.get("seed", 13)) + 1
    texts, labels, counts = _collect_configured_texts(
        config_dir=Path(args.config_dir),
        machine_path=args.machine,
        split_kind="eval",
        samples_per_capability=args.samples_per_capability,
        seed=seed,
        max_chars=2000,
    )
    embeddings = _embed(
        texts,
        space,
        cache_path=output_dir / "routing_test_embeddings.npz",
        batch_size=args.batch_size,
        recompute=args.recompute,
    )
    rows, confusion, stats = _evaluate(space, embeddings, texts, labels)
    output = output_dir / "routing_test_confusion_and_metrics.png"
    _render(output, space.capabilities, confusion, stats)
    _write_csv(output_dir / "routing_test_predictions.csv", rows)
    _write_csv(output_dir / "routing_test_errors.csv", [row for row in rows if not row["correct"]])
    _write_csv(output_dir / "routing_test_by_capability.csv", stats)

    summary = {
        "space": args.space,
        "samples": len(rows),
        "samples_by_capability": counts,
        "splits": EVAL_SPLITS,
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "coverage": sum(row["accepted"] for row in rows) / len(rows),
        "selective_accuracy": (
            sum(row["correct"] for row in rows if row["accepted"])
            / sum(row["accepted"] for row in rows)
        ),
        "confusion_matrix": confusion.tolist(),
        "capabilities": list(space.capabilities),
        "by_capability": stats,
        "output": str(output),
    }
    report_path = Path(args.report)
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            recorded = json.load(handle)["health"]
        summary["matches_recorded_report"] = {
            "accuracy": bool(np.isclose(summary["accuracy"], recorded["accuracy"])),
            "coverage": bool(np.isclose(summary["coverage"], recorded["coverage"])),
            "selective_accuracy": bool(np.isclose(summary["selective_accuracy"], recorded["selective_accuracy"])),
        }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
