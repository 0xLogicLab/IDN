#!/usr/bin/env python3
"""Regenerate the carving accuracy plot from a completed sweep summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_carving_accuracy_sweep import plot_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="artifacts/reports/carving_accuracy_sweep/summary.json",
    )
    parser.add_argument(
        "--healing-summary",
        default="artifacts/reports/carving_accuracy_sweep/healing_summary.json",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary_path = Path(args.summary)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    rows = summary.get("results", [])
    if len(rows) < 2:
        raise SystemExit("carving summary needs at least two completed points")
    output = (
        Path(args.output)
        if args.output
        else summary_path.parent / "carving_accuracy_curve.png"
    )
    healing_path = Path(args.healing_summary)
    healing_rows = None
    if healing_path.exists():
        with healing_path.open("r", encoding="utf-8") as handle:
            healing_rows = json.load(handle).get("results", [])
    plot_results(output, rows, healing_rows)
    print("plot: {0}".format(output))
    print("pdf: {0}".format(output.with_suffix(".pdf")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
