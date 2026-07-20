#!/usr/bin/env python3
"""Compare Qwen2.5-1.5B, 7B, and selected 1.5B LoRA specialists."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


TASKS = [
    ("cnn_dailymail", "CNN/DailyMail", "ROUGE-L", "rougeL"),
    ("cola", "CoLA", "Accuracy", "accuracy"),
    ("gsm8k", "GSM8K", "Exact match", "exact_match"),
    ("mbpp", "MBPP", "pass@1", "pass@1"),
    ("squad", "SQuAD", "Exact match", "exact_match"),
    ("sst2", "SST-2", "Accuracy", "accuracy"),
]


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _font() -> None:
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})


def _rows(
    base_dir: Path,
    large_dir: Path,
    lora_dir: Path,
    lora_overrides: Mapping[str, Path],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task, label, metric_label, metric in TASKS:
        base = float(_load(base_dir / f"{task}.json")["metrics"][metric])
        large = float(_load(large_dir / f"{task}.json")["metrics"][metric])
        lora_path = lora_overrides.get(task, lora_dir / f"{task}.json")
        lora = float(_load(lora_path)["metrics"][metric])
        rows.append(
            {
                "task": task,
                "label": label,
                "metric": metric,
                "small_base": base,
                "large_7b": large,
                "small_lora": lora,
                "lora_artifact": str(lora_path),
                "base_vs_7b_pp": (base - large) * 100,
                "lora_vs_7b_pp": (lora - large) * 100,
                "lora_gain_pp": (lora - base) * 100,
                "display_label": f"{label}\n{metric_label}",
            }
        )
    return rows


def _plot(path: Path, rows: List[Dict[str, Any]]) -> None:
    _font()
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    x = np.arange(len(rows))
    width = 0.24
    base_values = np.asarray([row["small_base"] * 100 for row in rows])
    large_values = np.asarray([row["large_7b"] * 100 for row in rows])
    lora_values = np.asarray([row["small_lora"] * 100 for row in rows])
    base_bars = ax.bar(x - width, base_values, width, color="#8fa3b8", label="Qwen2.5-1.5B")
    large_bars = ax.bar(x, large_values, width, color="#687078", label="Qwen2.5-7B")
    lora_bars = ax.bar(x + width, lora_values, width, color="#d66b4c", label="1.5B + 领域 LoRA")
    ax.set_xticks(x, [row["display_label"] for row in rows], fontsize=10)
    ax.set_ylabel("任务 primary score（%）", fontsize=13, labelpad=8)
    ax.set_title(
        "六类任务上的 1.5B、7B 与 1.5B + 领域 LoRA",
        fontsize=18,
        weight="bold",
        pad=14,
    )
    ax.set_ylim(0, 116)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=10.5)
    for bars in (base_bars, large_bars, lora_bars):
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=230, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="artifacts/eval/base_all_limit1000")
    parser.add_argument("--large-dir", default="artifacts/eval/base_all_limit1000_7B")
    parser.add_argument("--lora-dir", default="artifacts/eval/lora_all_limit1000")
    parser.add_argument(
        "--gsm8k-lora-artifact",
        default="artifacts/eval/gsm8k_r16_answer_weighted_s0.1_limit1000.json",
        help="Selected GSM8K specialist artifact; overrides --lora-dir for GSM8K.",
    )
    parser.add_argument("--output-dir", default="artifacts/reports/model_scale_difficulty")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(
        Path(args.base_dir),
        Path(args.large_dir),
        Path(args.lora_dir),
        {"gsm8k": Path(args.gsm8k_lora_artifact)},
    )
    output = output_dir / "lora_specialist_vs_7b.png"
    _plot(output, rows)
    with (output_dir / "lora_specialist_vs_7b.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "display_label"])
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key != "display_label"} for row in rows])
    print(json.dumps({"output": str(output), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
