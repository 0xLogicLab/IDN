from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COLA_URL = "https://raw.githubusercontent.com/nyu-mll/CoLA-baselines/master/acceptability_corpus/cola_public/tokenized/in_domain_dev.tsv"
DEFAULT_PROMPT = """You are solving a linguistic acceptability classification task.
Return exactly one character:
A = the sentence is grammatically acceptable English.
B = the sentence is grammatically unacceptable English.

Sentence: {sentence}
Answer:"""


@dataclass(frozen=True)
class ClassificationExample:
    request_id: str
    sentence: str
    label: int

    @property
    def label_text(self) -> str:
        return "A" if self.label == 1 else "B"


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def download_cola(out_path: str | Path) -> None:
    ensure_parent(out_path)
    with urllib.request.urlopen(COLA_URL, timeout=60) as response:
        raw = response.read()
    Path(out_path).write_bytes(raw)


def load_cola(path: str | Path, num_samples: int = 0) -> list[ClassificationExample]:
    records: list[ClassificationExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            if len(row) < 4:
                continue
            records.append(
                ClassificationExample(
                    request_id=f"cola-dev-{idx:04d}",
                    sentence=row[3].strip(),
                    label=int(row[1]),
                )
            )
            if num_samples > 0 and len(records) >= num_samples:
                break
    return records


def normalize_prediction(text: str) -> str | None:
    cleaned = text.strip().upper()
    match = re.search(r"\b([AB])\b", cleaned)
    if match:
        return match.group(1)
    if cleaned.startswith("A"):
        return "A"
    if cleaned.startswith("B"):
        return "B"
    if "ACCEPTABLE" in cleaned and "UNACCEPTABLE" not in cleaned:
        return "A"
    if "UNACCEPTABLE" in cleaned:
        return "B"
    return None


def call_chat_completion(model: str, prompt: str, max_tokens: int = 4, temperature: float = 0.0) -> dict[str, Any]:
    provider, model_name = parse_model(model)
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openclaw.local/idn",
            "X-Title": "IDN Capability Evaluation",
        }
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    if not api_key:
        raise RuntimeError(f"{provider.upper()} API key is required. Start an interactive shell or export it before running.")

    body = json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - start) * 1000
    choice = payload["choices"][0]
    text = choice["message"].get("content", "")
    usage = payload.get("usage") or {}
    return {"text": text, "latency_ms": latency_ms, "usage": usage, "raw_model": payload.get("model")}


def parse_model(model: str) -> tuple[str, str]:
    if model.startswith("openrouter:"):
        return "openrouter", model.removeprefix("openrouter:")
    if model.startswith("openai:"):
        return "openai", model.removeprefix("openai:")
    if "/" in model:
        return "openrouter", model
    return "openai", model


def evaluate_model(model: str, examples: list[ClassificationExample], prompt_template: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ex in examples:
        prompt = prompt_template.format(sentence=ex.sentence)
        result = call_chat_completion(model, prompt)
        pred = normalize_prediction(result["text"])
        rows.append(
            {
                "request_id": ex.request_id,
                "sentence": ex.sentence,
                "label": ex.label_text,
                "prediction": pred,
                "correct": pred == ex.label_text,
                "model": model,
                "raw_model": result.get("raw_model"),
                "text": result["text"],
                "latency_ms": result["latency_ms"],
                "usage": result.get("usage", {}),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    invalid = sum(1 for row in rows if row["prediction"] is None)
    latencies = sorted(float(row["latency_ms"]) for row in rows)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, max(0, round((p / 100) * (len(latencies) - 1))))
        return latencies[idx]

    return {
        "num_examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "invalid_predictions": invalid,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50": pct(50),
            "p95": pct(95),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate big/small LLM classification accuracy on real CoLA dev data.")
    parser.add_argument("--dataset-path", default="data/cola/in_domain_dev.tsv")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--big-model", default="openai/gpt-4o")
    parser.add_argument("--small-model", default="openai/gpt-4o-mini")
    parser.add_argument("--output-path", default="artifacts/eval/cola_big_vs_small.jsonl")
    parser.add_argument("--summary-path", default="artifacts/eval/cola_big_vs_small_summary.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if args.download or not dataset_path.exists():
        download_cola(dataset_path)

    examples = load_cola(dataset_path, args.num_samples)
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {
        "dataset": "CoLA in-domain dev",
        "dataset_url": COLA_URL,
        "dataset_path": str(dataset_path),
        "num_samples": len(examples),
        "task": "binary grammatical acceptability classification; A=acceptable, B=unacceptable",
        "models": {},
    }

    for role, model in [("small", args.small_model), ("big", args.big_model)]:
        rows = evaluate_model(model, examples, DEFAULT_PROMPT)
        for row in rows:
            row["role"] = role
        all_rows.extend(rows)
        summaries["models"][role] = {"model": model, **summarize(rows)}

    ensure_parent(args.output_path)
    with Path(args.output_path).open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ensure_parent(args.summary_path)
    with Path(args.summary_path).open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
