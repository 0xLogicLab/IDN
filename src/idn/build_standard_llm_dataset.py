from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


LABEL_TEXT = {
    1: "acceptable",
    0: "unacceptable",
}


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_cola(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            if len(row) < 4:
                continue
            label = int(row[1])
            sentence = row[3].strip()
            rows.append(
                {
                    "id": f"cola-{Path(path).stem}-{idx:05d}",
                    "task": "grammatical_acceptability",
                    "source": "CoLA",
                    "instruction": "Classify whether the English sentence is grammatically acceptable. Answer with acceptable or unacceptable.",
                    "input": sentence,
                    "output": LABEL_TEXT[label],
                    "label": label,
                    "label_text": LABEL_TEXT[label],
                    "metadata": {
                        "original_source": row[0],
                        "acceptability_annotation": row[2],
                    },
                }
            )
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standard LLM instruction dataset from CoLA.")
    parser.add_argument("--train-tsv", default="data/cola/in_domain_train.tsv")
    parser.add_argument("--eval-tsv", default="data/cola/in_domain_dev.tsv")
    parser.add_argument("--train-output", default="data/standard_llm/cola_train.jsonl")
    parser.add_argument("--eval-output", default="data/standard_llm/cola_eval.jsonl")
    parser.add_argument("--combined-output", default="data/standard_llm/cola_all.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = load_cola(args.train_tsv)
    eval_rows = load_cola(args.eval_tsv)
    random.Random(args.seed).shuffle(train_rows)
    all_rows = train_rows + eval_rows

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    write_jsonl(args.combined_output, all_rows)

    stats = {
        "dataset": "CoLA grammatical acceptability",
        "task": "binary classification",
        "train": len(train_rows),
        "eval": len(eval_rows),
        "total": len(all_rows),
        "labels": {
            "acceptable": sum(row["label"] == 1 for row in all_rows),
            "unacceptable": sum(row["label"] == 0 for row in all_rows),
        },
        "files": {
            "train": args.train_output,
            "eval": args.eval_output,
            "combined": args.combined_output,
        },
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
