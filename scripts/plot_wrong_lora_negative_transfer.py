#!/usr/bin/env python3
"""Visualize negative transfer caused by applying the wrong capability LoRA."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle


LABELS = {
    "cnn_dailymail": "CNN/DM",
    "cola": "CoLA",
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
    "squad": "SQuAD",
    "sst2": "SST-2",
}


def _font() -> None:
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})


def _load(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "cross_capability_matrix" in payload:
        return payload["cross_capability_matrix"]
    if payload.get("experiment") != "full_cross_capability_lora_matrix":
        raise ValueError("unsupported cross-capability report: {0}".format(path))
    if payload.get("status") != "complete":
        raise ValueError("cross-capability report is incomplete: {0}".format(path))
    capabilities = [str(item) for item in payload["capabilities"]]
    return {
        "protocol": payload["protocol"],
        "rows_are_evaluation_tasks": True,
        "columns_are_adapters": capabilities,
        "metrics_by_task": {
            task: payload["base"][task]["primary_metric"]
            for task in capabilities
        },
        "base": {
            task: payload["base"][task]["primary_score"]
            for task in capabilities
        },
        "values": {
            task: {
                adapter: payload["matrix"]["{0}__{1}".format(task, adapter)][
                    "primary_score"
                ]
                for adapter in capabilities
            }
            for task in capabilities
        },
    }


def _rows(matrix: Mapping[str, Any]) -> List[Dict[str, Any]]:
    adapters = [str(item) for item in matrix["columns_are_adapters"]]
    rows = []
    for task in adapters:
        base = float(matrix["base"][task])
        for adapter in adapters:
            score = float(matrix["values"][task][adapter])
            rows.append(
                {
                    "request_capability": task,
                    "loaded_lora": adapter,
                    "capability_match": task == adapter,
                    "base_score": base,
                    "lora_score": score,
                    "delta_percentage_points": (score - base) * 100.0,
                }
            )
    return rows


def _plot(path: Path, rows: List[Mapping[str, Any]]) -> None:
    _font()
    capabilities = list(
        dict.fromkeys(str(row["request_capability"]) for row in rows)
    )
    by_pair = {
        (str(row["request_capability"]), str(row["loaded_lora"])): row
        for row in rows
    }
    deltas = np.asarray(
        [
            [
                float(by_pair[(task, adapter)]["delta_percentage_points"])
                for adapter in capabilities
            ]
            for task in capabilities
        ]
    )
    off_diagonal = [
        float(row["delta_percentage_points"])
        for row in rows
        if not bool(row["capability_match"])
    ]
    degraded = sum(value < 0.0 for value in off_diagonal)
    mean_delta = sum(off_diagonal) / len(off_diagonal)

    count = len(capabilities)
    figure_size = (6.3, 5.2) if count <= 3 else (8.0, 7.0)
    figure, axis = plt.subplots(figsize=figure_size)
    limit = max(20.0, float(np.abs(deltas).max()))
    image = axis.imshow(
        deltas,
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )
    for row_index, task in enumerate(capabilities):
        for column_index, adapter in enumerate(capabilities):
            record = by_pair[(task, adapter)]
            delta = float(record["delta_percentage_points"])
            score = 100.0 * float(record["lora_score"])
            text_color = "white" if abs(delta) >= 40.0 else "#161616"
            axis.text(
                column_index,
                row_index,
                "{0:.1f}%\n({1:+.1f} pp)".format(score, delta),
                ha="center",
                va="center",
                fontsize=12 if count <= 3 else 9.5,
                fontweight="bold" if task == adapter else "normal",
                color=text_color,
            )
            if task == adapter:
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.48, row_index - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#183B56",
                        linewidth=2.4,
                    )
                )

    axis.set_xticks(
        np.arange(len(capabilities)),
        ["{0}\nLoRA".format(LABELS[item]) for item in capabilities],
        fontsize=11 if count <= 3 else 10.5,
    )
    axis.set_yticks(
        np.arange(len(capabilities)),
        [
            "{0} 请求\n(基座 {1:.1f}%)".format(
                LABELS[item], 100.0 * float(by_pair[(item, item)]["base_score"])
            )
            for item in capabilities
        ],
        fontsize=11 if count <= 3 else 10.5,
    )
    axis.set_xlabel("加载的能力 LoRA", fontsize=14, labelpad=9)
    axis.set_ylabel("实际请求能力", fontsize=14, labelpad=9)
    axis.set_title(
        "错误能力 LoRA 往往造成负迁移\n"
        "错误组合中 {0}/{1} 个下降，平均 {2:+.1f} 个百分点".format(
            degraded, len(off_diagonal), mean_delta
        ),
        fontsize=16 if count <= 3 else 17,
        weight="bold",
        pad=16,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.05)
    colorbar.set_label("相对 1.5B 基座的变化（百分点）", fontsize=12)
    colorbar.ax.tick_params(labelsize=11)
    figure.text(
        0.5,
        0.015,
        "单元格：LoRA 得分（相对基座变化）；对角框：能力匹配。每个任务使用前 100 个留出样本。",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", default="artifacts/eval/cross_capability_6x6_limit100.json"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/reports/negative_transfer"
    )
    args = parser.parse_args()
    matrix = _load(Path(args.report))
    rows = _rows(matrix)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "wrong_lora_negative_transfer.png"
    _plot(output, rows)
    with (output_dir / "wrong_lora_negative_transfer.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
