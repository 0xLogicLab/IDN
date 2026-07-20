#!/usr/bin/env python3
"""Build FFN-only carved models, evaluate SST-2, and plot the accuracy curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idn.config import load_machine_config, resolve_relative_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="h100")
    parser.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--config", default="configs/capabilities/sst2.yaml")
    parser.add_argument(
        "--scores",
        default=(
            "artifacts/locate/sst2/"
            "train_Qwen2.5-1.5B-Instruct_n32_bfloat16/scores.json"
        ),
    )
    parser.add_argument(
        "--ratios", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    )
    parser.add_argument("--carved-root", required=True)
    parser.add_argument(
        "--output-dir", default="artifacts/reports/carving_accuracy_sweep"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ratios = parse_ratios(args.ratios)
    machine = load_machine_config(args.machine)
    source_model = Path(
        resolve_relative_dir(
            base_dir=machine.model_dir,
            relative_dir=args.model,
            field="--model",
        )
    )
    scores_path = Path(args.scores).resolve()
    if not scores_path.exists():
        raise SystemExit("locate scores do not exist: {0}".format(scores_path))
    carved_root = Path(args.carved_root)
    output_dir = Path(args.output_dir)
    eval_dir = output_dir / "eval"
    carved_root.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    original_parameters = count_safetensor_parameters(
        source_model / "model.safetensors"
    )
    original_bytes = (source_model / "model.safetensors").stat().st_size
    rows = []
    for ratio in ratios:
        ratio_name = format_ratio(ratio)
        if ratio == 1.0:
            model_path = source_model
        else:
            model_path = carved_root / "ffn_r{0}".format(ratio_name)
            ensure_carved_model(
                machine=args.machine,
                model=args.model,
                ratio=ratio,
                scores_path=scores_path,
                output_path=model_path,
            )
        eval_path = eval_dir / "sst2_ffn_r{0}.json".format(ratio_name)
        ensure_evaluation(
            machine=args.machine,
            config=Path(args.config),
            model_path=model_path,
            output_path=eval_path,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
        with eval_path.open("r", encoding="utf-8") as handle:
            evaluation = json.load(handle)
        weight_path = model_path / "model.safetensors"
        parameters = count_safetensor_parameters(weight_path)
        weight_bytes = weight_path.stat().st_size
        rows.append(
            {
                "ffn_keep_ratio": ratio,
                "parameter_count": parameters,
                "parameter_retention": parameters / original_parameters,
                "weight_bytes": weight_bytes,
                "weight_size_retention": weight_bytes / original_bytes,
                "sst2_accuracy": float(evaluation["metrics"]["accuracy"]),
                "samples": int(evaluation["completed_examples"]),
                "model_path": str(model_path),
                "eval_artifact": str(eval_path),
            }
        )
        write_outputs(output_dir, rows, source_model, scores_path)
        print(
            "ratio={0:.1f} parameters={1:.2%} accuracy={2:.4f}".format(
                ratio, rows[-1]["parameter_retention"], rows[-1]["sst2_accuracy"]
            ),
            flush=True,
        )
    try:
        plot_results(output_dir / "carving_accuracy_curve.png", rows)
    except ImportError as exc:
        print("Plot skipped because a plotting dependency is missing: {0}".format(exc))
    print("summary: {0}".format(output_dir / "summary.json"))
    print("plot: {0}".format(output_dir / "carving_accuracy_curve.png"))
    return 0


def parse_ratios(value: str) -> List[float]:
    ratios = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not ratios or any(ratio <= 0.0 or ratio > 1.0 for ratio in ratios):
        raise ValueError("ratios must be comma-separated values in (0, 1]")
    if 1.0 not in ratios:
        ratios.append(1.0)
    return sorted(ratios)


def ensure_carved_model(
    *,
    machine: str,
    model: str,
    ratio: float,
    scores_path: Path,
    output_path: Path,
) -> None:
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists() and (output_path / "model.safetensors").exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        unit_ratios = manifest.get("unit_keep_ratios", {})
        if (
            float(unit_ratios.get("attention", -1.0)) == 1.0
            and float(unit_ratios.get("ffn", -1.0)) == ratio
        ):
            print("Reusing carved model: {0}".format(output_path), flush=True)
            return
        raise RuntimeError("existing carved model has incompatible ratios: {0}".format(output_path))
    command = [
        sys.executable,
        "-m",
        "idn.cli",
        "--machine",
        machine,
        "carve",
        "--model",
        model,
        "--keep-ratio",
        str(ratio),
        "--attention-keep-ratio",
        "1.0",
        "--ffn-keep-ratio",
        str(ratio),
        "--scores",
        str(scores_path),
        "--output",
        str(output_path),
    ]
    run(command)


def ensure_evaluation(
    *,
    machine: str,
    config: Path,
    model_path: Path,
    output_path: Path,
    batch_size: int,
    dtype: str,
) -> None:
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            evaluation = json.load(handle)
        if evaluation.get("status") == "complete":
            print("Reusing evaluation: {0}".format(output_path), flush=True)
            return
    command = [
        sys.executable,
        "-m",
        "idn.cli",
        "--machine",
        machine,
        "eval",
        "--config",
        str(config),
        "--model-dir",
        str(model_path.parent),
        "--model",
        model_path.name,
        "--limit",
        "0",
        "--batch-size",
        str(batch_size),
        "--dtype",
        dtype,
        "--output",
        str(output_path),
        "--override",
    ]
    run(command)


def run(command: Sequence[str]) -> None:
    print("Running: {0}".format(" ".join(command)), flush=True)
    subprocess.run(list(command), check=True)


def count_safetensor_parameters(path: Path) -> int:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return sum(
            math.prod(handle.get_slice(key).get_shape()) for key in handle.keys()
        )


def write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    source_model: Path,
    scores_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "artifact_version": 1,
        "experiment": "ffn_only_carving_accuracy_sweep",
        "capability": "sst2",
        "strategy": {
            "attention_keep_ratio": 1.0,
            "ffn_selection": "descending capability locate score per layer",
            "scores_path": str(scores_path),
            "source_model": str(source_model),
        },
        "results": list(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def plot_results(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    healing_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})
    ordered = sorted(rows, key=lambda row: float(row["parameter_retention"]))
    x = [100.0 * float(row["parameter_retention"]) for row in ordered]
    y = [100.0 * float(row["sst2_accuracy"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(6.2, 4.4))
    axis.plot(
        x,
        y,
        color="#2869A6",
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="仅 Carving",
    )
    if healing_rows:
        healing_by_ratio = {
            float(row["ffn_keep_ratio"]): row for row in healing_rows
        }
        matched = [
            healing_by_ratio[float(row["ffn_keep_ratio"])]
            for row in ordered
            if float(row["ffn_keep_ratio"]) in healing_by_ratio
        ]
        healed_x = [
            100.0
            * float(row.get("deployed_parameter_retention", row["parameter_retention"]))
            for row in matched
        ]
        healed_y = [100.0 * float(row["healed_accuracy"]) for row in matched]
        axis.plot(
            healed_x,
            healed_y,
            color="#D66A45",
            marker="s",
            linewidth=2.2,
            markersize=6,
            label="Carving + Healing（含 LoRA）",
        )
        for index, (x_value, y_value, row) in enumerate(
            zip(healed_x, healed_y, matched)
        ):
            if not bool(row.get("healing_applied", True)):
                continue
            axis.annotate(
                "{0:.1f}".format(y_value),
                (x_value, y_value),
                xytext=(0, 9 if index % 2 == 0 else 11),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#A9462C",
            )
    axis.axhline(
        y[-1], color="#BE4D3D", linestyle="--", linewidth=1.6, label="原始 1.5B"
    )
    for x_value, y_value, row in zip(x, y, ordered):
        axis.annotate(
            "{0:.1f}".format(y_value),
            (x_value, y_value),
            xytext=(0, 8 if healing_rows and y_value < 12 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#1F568A",
        )
    axis.set_xlabel("实际模型参数保留比例（%）", fontsize=13)
    axis.set_ylabel("SST-2 Accuracy（%）", fontsize=13)
    axis.set_ylim(-2, 100)
    axis.set_title(
        "FFN-only Carving 与 Healing 效果",
        fontsize=16,
        weight="bold",
        pad=12,
    )
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=11)
    axis.legend(frameon=False, fontsize=10, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def format_ratio(value: float) -> str:
    return ("{0:.3f}".format(value)).rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
