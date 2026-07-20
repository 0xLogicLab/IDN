#!/usr/bin/env python3
"""Plot all benchmark scores and a controlled GSM8K difficulty curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from matplotlib import font_manager
from scipy.interpolate import PchipInterpolator


TASKS = [
    ("sst2", "SST-2", "accuracy"),
    ("cola", "CoLA", "accuracy"),
    ("squad", "SQuAD", "exact_match"),
    ("cnn_dailymail", "CNN/DailyMail", "rougeL"),
    ("gsm8k", "GSM8K", "exact_match"),
    ("mbpp", "MBPP", "pass@1"),
]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "exact_match": "Exact Match",
    "rougeL": "ROUGE-L",
    "pass@1": "pass@1",
}
BIN_LABELS = ["≤1", "2", "3", "4", "5", "≥6"]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_number(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return f"{number:.12g}"


def _gold_operation_count(answer: str) -> int:
    return len(re.findall(r"<<[^<>]+>>", str(answer)))


def _difficulty_bin(operation_count: int) -> int:
    if operation_count <= 1:
        return 0
    if operation_count >= 6:
        return 5
    return operation_count - 1


def _wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _all_task_rows(small_dir: Path, large_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task, label, metric in TASKS:
        small = _load_json(small_dir / f"{task}.json")
        large = _load_json(large_dir / f"{task}.json")
        small_score = float(small["metrics"][metric])
        large_score = float(large["metrics"][metric])
        rows.append(
            {
                "task": task,
                "label": label,
                "metric": metric,
                "n": int(small.get("completed_examples", small.get("num_examples", 0))),
                "small_score": small_score,
                "large_score": large_score,
                "gap": large_score - small_score,
            }
        )
    return rows


def _gsm8k_rows(small_path: Path, large_path: Path, dataset_path: Path) -> List[Dict[str, Any]]:
    small = _load_json(small_path)
    large = _load_json(large_path)
    small_records = small.get("predictions", small.get("records", []))
    large_records = large.get("predictions", large.get("records", []))
    if len(small_records) != len(large_records):
        raise ValueError("GSM8K artifacts do not contain the same number of predictions")
    dataset = load_from_disk(str(dataset_path))["test"]
    bins: List[List[Tuple[bool, bool, int]]] = [[] for _ in BIN_LABELS]
    for small_record, large_record in zip(small_records, large_records):
        if small_record["id"] != large_record["id"] or small_record["reference"] != large_record["reference"]:
            raise ValueError(f"unpaired GSM8K record: {small_record['id']}")
        dataset_index = int(small_record["id"])
        gold = dataset[dataset_index]
        if gold["question"] != small_record["prompt_inputs"]["question"]:
            raise ValueError(f"dataset/artifact mismatch at GSM8K row {dataset_index}")
        operations = _gold_operation_count(gold["answer"])
        reference = _normalize_number(small_record["reference"])
        small_ok = _normalize_number(small_record["parsed_prediction"]) == reference
        large_ok = _normalize_number(large_record["parsed_prediction"]) == reference
        bins[_difficulty_bin(operations)].append((small_ok, large_ok, operations))

    rows: List[Dict[str, Any]] = []
    for index, values in enumerate(bins):
        n = len(values)
        small_correct = sum(int(item[0]) for item in values)
        large_correct = sum(int(item[1]) for item in values)
        small_low, small_high = _wilson_interval(small_correct, n)
        large_low, large_high = _wilson_interval(large_correct, n)
        rows.append(
            {
                "difficulty_bin": BIN_LABELS[index],
                "n": n,
                "mean_gold_operations": float(np.mean([item[2] for item in values])),
                "small_accuracy": small_correct / n,
                "small_ci95_low": small_low,
                "small_ci95_high": small_high,
                "large_accuracy": large_correct / n,
                "large_ci95_low": large_low,
                "large_ci95_high": large_high,
                "gap": (large_correct - small_correct) / n,
            }
        )
    return rows


def _configure_font() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 10.5})


def _smooth(x: np.ndarray, y: Sequence[float], dense_x: np.ndarray) -> np.ndarray:
    return np.clip(PchipInterpolator(x, np.asarray(y, dtype=float))(dense_x), 0.0, 100.0)


def _plot(output: Path, task_rows: Sequence[Dict[str, Any]], difficulty_rows: Sequence[Dict[str, Any]]) -> None:
    _configure_font()
    small_color = "#3f6f9f"
    large_color = "#d66b4c"
    fig, (ax_tasks, ax_curve) = plt.subplots(1, 2, figsize=(14.2, 6.4), gridspec_kw={"width_ratios": [1, 1.35]})

    y = np.arange(len(task_rows))[::-1]
    small_scores = np.asarray([row["small_score"] * 100 for row in task_rows])
    large_scores = np.asarray([row["large_score"] * 100 for row in task_rows])
    for yi, small, large in zip(y, small_scores, large_scores):
        ax_tasks.plot([small, large], [yi, yi], color="#c7cdd3", linewidth=3, zorder=1)
    small_handle = ax_tasks.scatter(small_scores, y, color=small_color, s=65, label="Qwen2.5-1.5B", zorder=3)
    large_handle = ax_tasks.scatter(large_scores, y, color=large_color, s=65, label="Qwen2.5-7B", zorder=3)
    labels = [f"{row['label']}\n{METRIC_LABELS[row['metric']]}" for row in task_rows]
    ax_tasks.set_yticks(y, labels)
    ax_tasks.set_xlim(0, 100)
    ax_tasks.set_xlabel("任务 primary score (%)")
    ax_tasks.set_title("A  全部已有任务的模型对比", loc="left", fontsize=13, weight="bold", pad=13)
    ax_tasks.grid(axis="x", alpha=0.2)
    ax_tasks.spines[["top", "right", "left"]].set_visible(False)
    ax_tasks.tick_params(axis="y", length=0)
    for yi, row in zip(y, task_rows):
        right = max(row["small_score"], row["large_score"]) * 100
        ax_tasks.text(min(right + 2.0, 97), yi, f"{row['gap'] * 100:+.1f} pp", va="center", fontsize=9, color="#555555")

    x = np.arange(len(difficulty_rows), dtype=float)
    dense_x = np.linspace(x.min(), x.max(), 300)
    small = np.asarray([row["small_accuracy"] * 100 for row in difficulty_rows])
    large = np.asarray([row["large_accuracy"] * 100 for row in difficulty_rows])
    small_low = np.asarray([row["small_ci95_low"] * 100 for row in difficulty_rows])
    small_high = np.asarray([row["small_ci95_high"] * 100 for row in difficulty_rows])
    large_low = np.asarray([row["large_ci95_low"] * 100 for row in difficulty_rows])
    large_high = np.asarray([row["large_ci95_high"] * 100 for row in difficulty_rows])
    ax_curve.fill_between(dense_x, _smooth(x, small_low, dense_x), _smooth(x, small_high, dense_x), color=small_color, alpha=0.12)
    ax_curve.fill_between(dense_x, _smooth(x, large_low, dense_x), _smooth(x, large_high, dense_x), color=large_color, alpha=0.12)
    ax_curve.plot(dense_x, _smooth(x, small, dense_x), color=small_color, linewidth=2.8)
    ax_curve.plot(dense_x, _smooth(x, large, dense_x), color=large_color, linewidth=2.8)
    ax_curve.scatter(x, small, color=small_color, edgecolor="white", linewidth=1.2, s=70, zorder=3)
    ax_curve.scatter(x, large, color=large_color, edgecolor="white", linewidth=1.2, s=70, zorder=3)
    ax_curve.set_xticks(x, [f"{row['difficulty_bin']}\n(n={row['n']})" for row in difficulty_rows])
    ax_curve.set_xlabel("标准答案中的运算步骤数")
    ax_curve.set_ylabel("GSM8K Exact Match (%)")
    ax_curve.set_ylim(30, 100)
    ax_curve.set_title("B  同一任务内：难度越高，模型差距越明显", loc="left", fontsize=13, weight="bold", pad=13)
    ax_curve.grid(axis="y", alpha=0.2)
    ax_curve.spines[["top", "right"]].set_visible(False)
    for xi, row in zip(x, difficulty_rows):
        midpoint = (row["small_accuracy"] + row["large_accuracy"]) * 50
        ax_curve.text(xi, midpoint, f"{row['gap'] * 100:+.1f}", ha="center", va="center", fontsize=8.5, color="#444444", bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9})
    ax_curve.text(0.99, 0.02, "阴影：Wilson 95% CI；曲线：PCHIP，仅用于连接分箱估计", transform=ax_curve.transAxes, ha="right", va="bottom", fontsize=8.5, color="#666666")

    fig.suptitle("Qwen2.5-1.5B 与 7B：全任务表现及受控难度实验", fontsize=16, weight="bold", y=0.99)
    fig.legend([small_handle, large_handle], ["Qwen2.5-1.5B", "Qwen2.5-7B"], frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.28, 0.925), fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3.5)
    fig.savefig(output, dpi=230, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_curve_only(output: Path, difficulty_rows: Sequence[Dict[str, Any]]) -> None:
    """Export the controlled GSM8K difficulty panel as a standalone figure."""
    _configure_font()
    small_color = "#3f6f9f"
    large_color = "#d66b4c"
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(difficulty_rows), dtype=float)
    dense_x = np.linspace(x.min(), x.max(), 300)
    small = np.asarray([row["small_accuracy"] * 100 for row in difficulty_rows])
    large = np.asarray([row["large_accuracy"] * 100 for row in difficulty_rows])
    small_low = np.asarray([row["small_ci95_low"] * 100 for row in difficulty_rows])
    small_high = np.asarray([row["small_ci95_high"] * 100 for row in difficulty_rows])
    large_low = np.asarray([row["large_ci95_low"] * 100 for row in difficulty_rows])
    large_high = np.asarray([row["large_ci95_high"] * 100 for row in difficulty_rows])
    ax.fill_between(dense_x, _smooth(x, small_low, dense_x), _smooth(x, small_high, dense_x), color=small_color, alpha=0.14)
    ax.fill_between(dense_x, _smooth(x, large_low, dense_x), _smooth(x, large_high, dense_x), color=large_color, alpha=0.14)
    ax.plot(dense_x, _smooth(x, small, dense_x), color=small_color, linewidth=3.0, label="Qwen2.5-1.5B")
    ax.plot(dense_x, _smooth(x, large, dense_x), color=large_color, linewidth=3.0, label="Qwen2.5-7B")
    ax.scatter(x, small, color=small_color, edgecolor="white", linewidth=1.3, s=78, zorder=3)
    ax.scatter(x, large, color=large_color, edgecolor="white", linewidth=1.3, s=78, zorder=3)
    ax.set_xticks(x, [row["difficulty_bin"] for row in difficulty_rows], fontsize=12)
    ax.set_xlabel("标准答案中的运算步骤数", fontsize=14, labelpad=8)
    ax.set_ylabel("GSM8K准确率（%）", fontsize=14, labelpad=8)
    ax.set_ylim(30, 100)
    ax.set_title("GSM8K 难度曲线", fontsize=18, weight="bold", pad=14)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=1, loc="lower left", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=230, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-dir", default="artifacts/eval/base_all_limit1000")
    parser.add_argument("--large-dir", default="artifacts/eval/base_all_limit1000_7B")
    parser.add_argument("--gsm8k-dataset", default="/data/huggingface/dataset/gsm8k")
    parser.add_argument("--output-dir", default="artifacts/reports/model_scale_difficulty")
    args = parser.parse_args()

    small_dir = Path(args.small_dir)
    large_dir = Path(args.large_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_rows = _all_task_rows(small_dir, large_dir)
    difficulty_rows = _gsm8k_rows(small_dir / "gsm8k.json", large_dir / "gsm8k.json", Path(args.gsm8k_dataset))
    output = output_dir / "all_tasks_and_gsm8k_difficulty.png"
    _plot(output, task_rows, difficulty_rows)
    _plot_curve_only(output_dir / "gsm8k_difficulty_curve.png", difficulty_rows)
    _write_csv(output_dir / "all_task_scores.csv", task_rows)
    _write_csv(output_dir / "gsm8k_difficulty_bins.csv", difficulty_rows)
    with (output_dir / "all_tasks_and_difficulty.json").open("w", encoding="utf-8") as handle:
        json.dump({"models": ["Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct"], "task_scores": task_rows, "gsm8k_difficulty": difficulty_rows}, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output), "difficulty_bins": difficulty_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
