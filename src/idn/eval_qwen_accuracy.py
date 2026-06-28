from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

COLA_DEV_URL = "https://raw.githubusercontent.com/nyu-mll/CoLA-baselines/master/acceptability_corpus/cola_public/tokenized/in_domain_dev.tsv"
COLA_TRAIN_URL = "https://raw.githubusercontent.com/nyu-mll/CoLA-baselines/master/acceptability_corpus/cola_public/tokenized/in_domain_train.tsv"
ZERO_SHOT_TEMPLATE = """You are solving a linguistic acceptability classification task.
Return exactly one character:
A = the sentence is grammatically acceptable English.
B = the sentence is grammatically unacceptable English.

Sentence: {sentence}
Answer:"""
FEW_SHOT_HEADER = """Classify whether each English sentence is grammatically acceptable.
Return exactly one character: A or B.
A = acceptable. B = unacceptable.

Examples:
"""


@dataclass(frozen=True)
class Example:
    request_id: str
    sentence: str
    label: str


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def download(url: str, path: str | Path) -> None:
    ensure_parent(path)
    with urllib.request.urlopen(url, timeout=60) as response:
        Path(path).write_bytes(response.read())


def load_cola(path: str | Path, num_samples: int = 0) -> list[Example]:
    rows: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            if len(row) < 4:
                continue
            rows.append(Example(f"{Path(path).stem}-{idx:04d}", row[3].strip(), "A" if int(row[1]) == 1 else "B"))
            if num_samples > 0 and len(rows) >= num_samples:
                break
    return rows


def balanced_few_shots(examples: list[Example], per_label: int) -> list[Example]:
    selected: list[Example] = []
    counts = {"A": 0, "B": 0}
    for ex in examples:
        if counts[ex.label] < per_label:
            selected.append(ex)
            counts[ex.label] += 1
        if counts["A"] >= per_label and counts["B"] >= per_label:
            break
    return selected


def build_prompt(sentence: str, few_shots: list[Example]) -> str:
    if not few_shots:
        return ZERO_SHOT_TEMPLATE.format(sentence=sentence)
    body = "".join(f"Sentence: {ex.sentence}\nAnswer: {ex.label}\n\n" for ex in few_shots)
    return FEW_SHOT_HEADER + body + f"Sentence: {sentence}\nAnswer:"


def normalize_prediction(text: str) -> str | None:
    cleaned = text.strip().upper()
    match = re.search(r"\b([AB])\b", cleaned)
    if match:
        return match.group(1)
    if cleaned.startswith("A"):
        return "A"
    if cleaned.startswith("B"):
        return "B"
    if "UNACCEPTABLE" in cleaned:
        return "B"
    if "ACCEPTABLE" in cleaned:
        return "A"
    return None


class LocalQwenClassifier:
    def __init__(self, model_path: str, torch_dtype: str = "float16", device_map: str = "auto") -> None:
        self.model_path = model_path
        dtype_map = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32, "auto": "auto"}
        dtype = dtype_map.get(torch_dtype.lower(), torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
        self.model.eval()

    def format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        start = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {"text": text, "latency_ms": latency_ms, "input_tokens": int(inputs["input_ids"].shape[-1]), "output_tokens": int(generated.shape[-1])}


def evaluate(role: str, model_path: str, examples: list[Example], few_shots: list[Example], max_new_tokens: int, torch_dtype: str, device_map: str) -> list[dict[str, Any]]:
    clf = LocalQwenClassifier(model_path, torch_dtype=torch_dtype, device_map=device_map)
    rows: list[dict[str, Any]] = []
    for ex in examples:
        prompt = build_prompt(ex.sentence, few_shots)
        result = clf.generate(prompt, max_new_tokens=max_new_tokens)
        pred = normalize_prediction(result["text"])
        rows.append({"request_id": ex.request_id, "sentence": ex.sentence, "label": ex.label, "prediction": pred, "correct": pred == ex.label, "role": role, "model": model_path, **result})
    del clf
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    return {"num_examples": total, "correct": correct, "accuracy": correct / total if total else 0.0, "invalid_predictions": invalid, "latency_ms": {"mean": sum(latencies) / len(latencies) if latencies else 0.0, "p50": pct(50), "p95": pct(95)}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local Qwen big/small models on CoLA accuracy.")
    parser.add_argument("--dataset-path", default="data/cola/in_domain_dev.tsv")
    parser.add_argument("--train-path", default="data/cola/in_domain_train.tsv")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--few-shot-per-label", type=int, default=3)
    parser.add_argument("--small-model", default="/data/huggingface/model/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--big-model", default="/data/huggingface/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output-path", default="artifacts/eval/cola_qwen25_7b_vs_15b_200.jsonl")
    parser.add_argument("--summary-path", default="artifacts/eval/cola_qwen25_7b_vs_15b_200_summary.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    train_path = Path(args.train_path)
    if args.download or not dataset_path.exists():
        download(COLA_DEV_URL, dataset_path)
    if args.few_shot_per_label > 0 and (args.download or not train_path.exists()):
        download(COLA_TRAIN_URL, train_path)

    examples = load_cola(dataset_path, args.num_samples)
    few_shots = balanced_few_shots(load_cola(train_path), args.few_shot_per_label) if args.few_shot_per_label > 0 else []

    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "dataset": "CoLA in-domain dev",
        "dataset_url": COLA_DEV_URL,
        "train_url": COLA_TRAIN_URL if few_shots else None,
        "dataset_path": str(dataset_path),
        "train_path": str(train_path) if few_shots else None,
        "num_samples": len(examples),
        "few_shot_per_label": args.few_shot_per_label,
        "few_shots": [ex.__dict__ for ex in few_shots],
        "task": "binary grammatical acceptability classification; A=acceptable, B=unacceptable",
        "generation": {"max_new_tokens": args.max_new_tokens, "temperature": 0.0},
        "models": {},
    }
    for role, model_path in [("small", args.small_model), ("big", args.big_model)]:
        rows = evaluate(role, model_path, examples, few_shots, args.max_new_tokens, args.torch_dtype, args.device_map)
        all_rows.extend(rows)
        summary["models"][role] = {"model": model_path, **summarize(rows)}

    ensure_parent(args.output_path)
    with Path(args.output_path).open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ensure_parent(args.summary_path)
    with Path(args.summary_path).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
