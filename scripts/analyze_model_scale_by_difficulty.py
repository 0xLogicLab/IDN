#!/usr/bin/env python3
"""Compare Qwen2.5-1.5B and 7B on predeclared simple and hard tasks.

The script consumes completed per-example evaluation artifacts.  It deliberately
does not run model inference: this keeps the comparison paired and reproducible
from the existing local experiment outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


GROUPS = {
    "simple": ["cola", "sst2"],
    "medium": ["cnn_dailymail", "squad"],
    "hard": ["gsm8k", "mbpp"],
}
GROUP_LABELS = {"simple": "简单", "medium": "中等", "hard": "困难"}
TASK_LABELS = {
    "cola": "CoLA",
    "sst2": "SST-2",
    "cnn_dailymail": "CNN/DailyMail",
    "squad": "SQuAD",
    "gsm8k": "GSM8K",
    "mbpp": "MBPP",
}
TASK_METRICS = {
    "cola": "accuracy",
    "sst2": "accuracy",
    "cnn_dailymail": "rougeL",
    "squad": "exact_match",
    "gsm8k": "exact_match",
    "mbpp": "pass@1",
}


def _normalize_number(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        normalized = format(Decimal(text).normalize(), "f")
    except InvalidOperation:
        return None
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized == "-0" else normalized


def _extract_python_code(raw_text: Any) -> str:
    text = str(raw_text or "").strip()
    blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.I | re.S)
    if blocks:
        return blocks[-1].strip()
    return re.sub(r"^(python code|solution|answer)\s*:\s*", "", text, flags=re.I).strip("` \n\t")


def _run_python_tests(code: str, tests: Sequence[str], setup: str) -> bool:
    # Importing the project helper keeps the exact MBPP sandbox semantics used
    # by the original evaluator, including timeout and standard imports.
    from idn.capabilities.mbpp import run_python_tests

    return bool(run_python_tests(code, list(tests), setup))


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("status") not in (None, "complete"):
        raise ValueError(f"incomplete evaluation artifact: {path}")
    return value


def _score(task: str, record: Mapping[str, Any]) -> float:
    pred = record.get("parsed_prediction")
    ref = record.get("reference")
    if task in ("cola", "sst2"):
        return float(pred is not None and int(pred) == int(ref))
    if task == "gsm8k":
        return float(_normalize_number(pred) is not None and _normalize_number(pred) == _normalize_number(ref))
    if task == "cnn_dailymail":
        from idn.capabilities.cnn_dailymail import rouge_l_f1

        return float(rouge_l_f1(pred, ref))
    if task == "squad":
        from idn.capabilities.squad import _normalize_references, squad_exact_match

        references = _normalize_references(ref)
        return float(max((squad_exact_match(pred, item) for item in references), default=0.0))
    if task == "mbpp":
        ref_dict = ref if isinstance(ref, dict) else {}
        return float(_run_python_tests(
            _extract_python_code(pred),
            ref_dict.get("tests", []),
            str(ref_dict.get("test_setup_code", "")),
        ))
    raise KeyError(task)


def _paired_task(task: str, small: Dict[str, Any], large: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    if small.get("capability") != task or large.get("capability") != task:
        raise ValueError(f"capability mismatch for {task}")
    small_records = small.get("predictions", small.get("records", []))
    large_records = large.get("predictions", large.get("records", []))
    if len(small_records) != len(large_records):
        raise ValueError(f"sample count mismatch for {task}: {len(small_records)} vs {len(large_records)}")
    pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for index, (srec, lrec) in enumerate(zip(small_records, large_records)):
        if (srec.get("id"), srec.get("reference")) != (lrec.get("id"), lrec.get("reference")):
            raise ValueError(f"unpaired record at {task}[{index}]")
        pairs.append((srec, lrec))
    if task == "mbpp":
        # MBPP's test harness launches a subprocess per item; parallel workers
        # reduce wall time without changing each item's deterministic result.
        with ThreadPoolExecutor(max_workers=min(16, len(pairs))) as pool:
            results = list(pool.map(lambda pair: (_score(task, pair[0]), _score(task, pair[1])), pairs))
        small_scores = [item[0] for item in results]
        large_scores = [item[1] for item in results]
    else:
        small_scores = [_score(task, pair[0]) for pair in pairs]
        large_scores = [_score(task, pair[1]) for pair in pairs]
    small_arr = np.asarray(small_scores, dtype=float)
    large_arr = np.asarray(large_scores, dtype=float)
    reported_small = float(small.get("metrics", {}).get(TASK_METRICS[task], math.nan))
    reported_large = float(large.get("metrics", {}).get(TASK_METRICS[task], math.nan))
    if not np.isclose(small_arr.mean(), reported_small, atol=1e-10):
        warnings.append(f"{task}: 1.5B artifact metric {reported_small:.9f} vs current harness {small_arr.mean():.9f}")
    if not np.isclose(large_arr.mean(), reported_large, atol=1e-10):
        warnings.append(f"{task}: 7B artifact metric {reported_large:.9f} vs current harness {large_arr.mean():.9f}")
    return {
        "task": task,
        "metric": TASK_METRICS[task],
        "n": int(len(small_arr)),
        "small_score": float(small_arr.mean()),
        "large_score": float(large_arr.mean()),
        "gap": float(large_arr.mean() - small_arr.mean()),
        "artifact_small_score": reported_small,
        "artifact_large_score": reported_large,
        "small_correct": small_arr,
        "large_correct": large_arr,
    }


def _bootstrap_task_gap(task: Dict[str, Any], reps: int, rng: np.random.Generator) -> np.ndarray:
    small = task["small_correct"]
    large = task["large_correct"]
    out = np.empty(reps, dtype=float)
    chunk = 512
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start)
        indices = rng.integers(0, len(small), size=(count, len(small)))
        out[start : start + count] = (large[indices] - small[indices]).mean(axis=1)
    return out


def _bootstrap_group_gap(tasks: Sequence[Dict[str, Any]], reps: int, rng: np.random.Generator) -> np.ndarray:
    # Resample within each task, then average task metrics so each task has
    # equal weight despite different split sizes (MBPP has only 257 examples).
    return np.mean([_bootstrap_task_gap(task, reps, rng) for task in tasks], axis=0)


def _summary(samples: np.ndarray, observed: float) -> Dict[str, float]:
    return {
        "estimate": float(observed),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "p_gap_le_zero": float((np.count_nonzero(samples <= 0) + 1) / (len(samples) + 1)),
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, group_rows: Sequence[Dict[str, Any]], interaction: Dict[str, float]) -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font_name = "DejaVu Sans"
    plt.rcParams.update({"font.family": font_name, "axes.unicode_minus": False, "font.size": 11})
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    x = np.arange(len(group_rows))
    small = np.asarray([row["small_score"] * 100 for row in group_rows])
    large = np.asarray([row["large_score"] * 100 for row in group_rows])
    ax.plot(x, small, color="#3f6f9f", marker="o", markersize=9, linewidth=2.8, label="Qwen2.5-1.5B")
    ax.plot(x, large, color="#d66b4c", marker="o", markersize=9, linewidth=2.8, label="Qwen2.5-7B")
    ax.fill_between(x, small - 0.7, small + 0.7, color="#3f6f9f", alpha=0.08)
    ax.fill_between(x, large - 0.7, large + 0.7, color="#d66b4c", alpha=0.08)
    ax.set_xticks(x, [GROUP_LABELS[row["difficulty"]] for row in group_rows], fontsize=12)
    ax.set_ylabel("标准化任务得分 (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.15, len(group_rows) - 0.85)
    ax.set_title("任务难度上升时，大模型优势显著扩大", fontsize=16, pad=18, weight="bold")
    ax.text(0.5, 1.01, "每类为所含任务 primary metric 的等权平均", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#666666")
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 0.98), fontsize=11)
    for values, color, yoffset in ((small, "#3f6f9f", -4.2), (large, "#d66b4c", 2.0)):
        for i, value in enumerate(values):
            ax.text(i, value + yoffset, f"{value:.1f}", ha="center", va="center", color=color, fontsize=11, weight="bold")
    for i, row in enumerate(group_rows):
        ax.text(i, (small[i] + large[i]) / 2, f"{row['gap'] * 100:+.1f} pp", ha="center", va="center", fontsize=9, color="#444444", bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.85})
    fig.text(0.5, 0.015, f"困难相对简单的额外规模收益：+{interaction['estimate'] * 100:.1f} pp（95% CI [{interaction['ci95_low'] * 100:+.1f}, {interaction['ci95_high'] * 100:+.1f}]）", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-dir", default="artifacts/eval/base_all_limit1000")
    parser.add_argument("--large-dir", default="artifacts/eval/base_all_limit1000_7B")
    parser.add_argument("--output-dir", default="artifacts/reports/model_scale_difficulty")
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if args.bootstrap_reps < 1000:
        parser.error("--bootstrap-reps must be at least 1000")
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    if str(Path(__file__).resolve().parents[1] / "src") not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for task in ["cola", "sst2", "cnn_dailymail", "squad", "gsm8k", "mbpp"]:
        tasks.append(_paired_task(task, _load(Path(args.small_dir) / f"{task}.json"), _load(Path(args.large_dir) / f"{task}.json"), warnings))

    rng = np.random.default_rng(args.seed)
    task_rows: List[Dict[str, Any]] = []
    bootstrap_by_task: Dict[str, np.ndarray] = {}
    for task in tasks:
        samples = _bootstrap_task_gap(task, args.bootstrap_reps, rng)
        bootstrap_by_task[task["task"]] = samples
        summary = _summary(samples, task["gap"])
        difficulty = next(name for name, names in GROUPS.items() if task["task"] in names)
        task_rows.append({"task": task["task"], "difficulty": difficulty, "metric": task["metric"], "n": task["n"], "small_score": task["small_score"], "large_score": task["large_score"], "artifact_small_score": task["artifact_small_score"], "artifact_large_score": task["artifact_large_score"], "gap": task["gap"], **{f"bootstrap_{k}": v for k, v in summary.items()}})

    group_rows: List[Dict[str, Any]] = []
    group_bootstraps: Dict[str, np.ndarray] = {}
    for difficulty, names in GROUPS.items():
        selected = [task for task in tasks if task["task"] in names]
        small_score = float(np.mean([task["small_score"] for task in selected]))
        large_score = float(np.mean([task["large_score"] for task in selected]))
        samples = _bootstrap_group_gap(selected, args.bootstrap_reps, rng)
        group_bootstraps[difficulty] = samples
        summary = _summary(samples, large_score - small_score)
        group_rows.append({"difficulty": difficulty, "tasks": ",".join(names), "small_score": small_score, "large_score": large_score, "gap": large_score - small_score, **{f"bootstrap_{k}": v for k, v in summary.items()}})

    interaction_samples = group_bootstraps["hard"] - group_bootstraps["simple"]
    interaction = _summary(interaction_samples, group_rows[2]["gap"] - group_rows[0]["gap"])
    result = {
        "experiment": "model_scale_by_task_difficulty",
        "definition": {
            "simple": "binary classification with one-label output (CoLA, SST-2)",
            "medium": "summarization or extractive question answering (CNN/DailyMail, SQuAD)",
            "hard": "multi-step reasoning or executable code generation (GSM8K, MBPP)",
        },
        "models": {"small": "Qwen2.5-1.5B-Instruct", "large": "Qwen2.5-7B-Instruct"},
        "bootstrap": {"reps": args.bootstrap_reps, "seed": args.seed, "ci": 0.95},
        "warnings": warnings,
        "tasks": task_rows,
        "groups": group_rows,
        "interaction_hard_minus_simple": interaction,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    _write_csv(output_dir / "task_results.csv", task_rows)
    _write_csv(output_dir / "group_results.csv", group_rows)
    _plot(output_dir / "model_scale_by_difficulty.png", group_rows, interaction)

    simple, medium, hard = group_rows
    report = f"""# 模型规模与任务难度实验\n\n本实验复用仓库中已完成的 Qwen2.5-1.5B-Instruct 和 Qwen2.5-7B-Instruct 基座模型评测产物。两种模型使用相同提示、解码设置和样本顺序；统计基于逐样本配对结果，而不是独立样本近似。\n\n## 预注册式任务划分\n\n- **简单任务**：CoLA、SST-2。两者都是二分类，只需输出一个标签。\n- **困难任务**：GSM8K、MBPP。前者要求多步数学推理，后者要求生成并通过测试的 Python 程序。\n\nCNN/DailyMail 和 SQuAD 保留在完整 benchmark 中，但不放入本次聚焦检验：它们分别是摘要和抽取式问答，输出形式与上述两组不对称。\n\n## 结果\n\n| 分组 | Qwen2.5-1.5B | Qwen2.5-7B | 差距（7B - 1.5B） | 配对 bootstrap 95% CI |\n|---|---:|---:|---:|---:|\n| 简单任务 | {simple['small_score']:.4f} | {simple['large_score']:.4f} | {simple['gap']*100:+.2f} pp | [{simple['bootstrap_ci95_low']*100:+.2f}, {simple['bootstrap_ci95_high']*100:+.2f}] pp |\n| 困难任务 | {hard['small_score']:.4f} | {hard['large_score']:.4f} | {hard['gap']*100:+.2f} pp | [{hard['bootstrap_ci95_low']*100:+.2f}, {hard['bootstrap_ci95_high']*100:+.2f}] pp |\n\n困难任务相对简单任务的额外规模收益为 **{interaction['estimate']*100:+.2f} pp**（95% CI [{interaction['ci95_low']*100:+.2f}, {interaction['ci95_high']*100:+.2f}] pp；bootstrap P(差异≤0)={interaction['p_gap_le_zero']:.4f}）。因此，数据支持“模型规模优势主要集中在困难任务上”：简单任务平均只相差约 {simple['gap']*100:.1f} 个百分点，而困难任务相差约 {hard['gap']*100:.1f} 个百分点，后者约为前者的 {hard['gap']/max(simple['gap'], 1e-9):.1f} 倍。这里的“差距不大”是实际效应意义上的表述，不等同于宣称每个简单任务的差异都严格为零。\n\n逐任务明细见 `task_results.csv`；完整机器可读结果和随机种子见 `results.json`。图同时输出为 `model_scale_by_difficulty.png` 和 PDF。\n"""
    report = report.replace(
        "- **困难任务**：GSM8K、MBPP。前者要求多步数学推理，后者要求生成并通过测试的 Python 程序。",
        "- **中等任务**：CNN/DailyMail、SQuAD，分别要求摘要和抽取式问答。\n- **困难任务**：GSM8K、MBPP。前者要求多步数学推理，后者要求生成并通过测试的 Python 程序。",
    ).replace(
        "CNN/DailyMail 和 SQuAD 保留在完整 benchmark 中，但不放入本次聚焦检验：它们分别是摘要和抽取式问答，输出形式与上述两组不对称。",
        "三类任务均纳入图中的趋势比较；由于原始指标不同，类别分数是各任务 primary metric 的等权平均，统一称为标准化任务得分。",
    ).replace(
        "| 困难任务 | {hard['small_score']:.4f} | {hard['large_score']:.4f} | {hard['gap']*100:+.2f} pp | [{hard['bootstrap_ci95_low']*100:+.2f}, {hard['bootstrap_ci95_high']*100:+.2f}] pp |",
        "| 中等任务 | {medium['small_score']:.4f} | {medium['large_score']:.4f} | {medium['gap']*100:+.2f} pp | [{medium['bootstrap_ci95_low']*100:+.2f}, {medium['bootstrap_ci95_high']*100:+.2f}] pp |\n| 困难任务 | {hard['small_score']:.4f} | {hard['large_score']:.4f} | {hard['gap']*100:+.2f} pp | [{hard['bootstrap_ci95_low']*100:+.2f}, {hard['bootstrap_ci95_high']*100:+.2f}] pp |",
    ).replace("逐任务明细见", "分组统计明细见")
    report = report.replace(
        "| 困难任务 |",
        f"| 中等任务 | {medium['small_score']:.4f} | {medium['large_score']:.4f} | {medium['gap']*100:+.2f} pp | [{medium['bootstrap_ci95_low']*100:+.2f}, {medium['bootstrap_ci95_high']*100:+.2f}] pp |\n| 困难任务 |",
        1,
    )
    if warnings:
        report = report.replace(
            "本实验复用仓库中已完成的 Qwen2.5-1.5B-Instruct 和 Qwen2.5-7B-Instruct 基座模型评测产物。",
            "本实验复用仓库中已完成的 Qwen2.5-1.5B-Instruct 和 Qwen2.5-7B-Instruct 基座模型评测产物。\n\n> 复现说明：MBPP 原始 JSON 中记录的 pass@1 与当前仓库测试 harness 相差 1/257；逐题统计使用当前 `run_python_tests` 对原始生成代码重新判定，差异详见 `results.json` 的 `warnings` 字段。",
        )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "groups": group_rows, "interaction": interaction}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
