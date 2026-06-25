from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class EvalItem:
    id: str
    task_type: str
    capability: str
    prompt: str
    answer: dict[str, Any]


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            continue
    return None


KEY_FIELDS = {
    "capability_descriptor_extraction": [
        "capability",
        "task",
        "input_modalities",
        "output_modalities",
        "quality_tier",
        "latency_class",
        "privacy_class",
        "resource_requirement",
    ],
    "route_decision": ["route", "fallback_required"],
    "fallback_decision": ["accept_local", "fallback_required", "target"],
    "resource_fit": ["feasible", "route_tier"],
    "cache_decision": ["action", "cache_type"],
    "trust_policy": ["allowed"],
}


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower().replace("-", "_")
    if isinstance(value, list):
        return sorted(normalize_value(v) for v in value)
    return value


def score_prediction(task_type: str, expected: dict[str, Any], predicted: dict[str, Any] | None) -> dict[str, Any]:
    fields = KEY_FIELDS.get(task_type, list(expected.keys()))
    if predicted is None:
        return {"json_valid": False, "field_correct": 0, "field_total": len(fields), "exact_match": False, "field_scores": {field: False for field in fields}}
    field_scores = {}
    for field in fields:
        field_scores[field] = normalize_value(predicted.get(field)) == normalize_value(expected.get(field))
    correct = sum(1 for ok in field_scores.values() if ok)
    return {
        "json_valid": True,
        "field_correct": correct,
        "field_total": len(fields),
        "field_accuracy": correct / len(fields) if fields else 0.0,
        "exact_match": correct == len(fields),
        "field_scores": field_scores,
    }


class LocalChatModel:
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
        wrapped = (
            "You are evaluating an IDN model capability benchmark.\n"
            "Return exactly one valid JSON object.\n"
            "Use double-quoted JSON keys and values.\n"
            "Do not use markdown fences, comments, explanations, or trailing commas.\n\n"
            f"{prompt}\n\n"
            "Final answer: one valid JSON object only."
        )
        messages = [
            {
                "role": "system",
                "content": "You are a strict JSON API. Output only valid JSON and no prose.",
            },
            {"role": "user", "content": wrapped},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        start = time.perf_counter()
        outputs = self.model.generate(
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
        generated = outputs[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "text": text,
            "latency_ms": latency_ms,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": int(generated.shape[-1]),
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    json_valid = sum(1 for r in rows if r["score"]["json_valid"])
    exact = sum(1 for r in rows if r["score"].get("exact_match"))
    field_correct = sum(r["score"].get("field_correct", 0) for r in rows)
    field_total = sum(r["score"].get("field_total", 0) for r in rows)
    latencies = sorted(r["latency_ms"] for r in rows)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, max(0, round((p / 100) * (len(latencies) - 1))))
        return float(latencies[idx])

    by_task = {}
    for task in sorted({r["task_type"] for r in rows}):
        subset = [r for r in rows if r["task_type"] == task]
        fc = sum(r["score"].get("field_correct", 0) for r in subset)
        ft = sum(r["score"].get("field_total", 0) for r in subset)
        by_task[task] = {
            "num_examples": len(subset),
            "json_valid_rate": sum(1 for r in subset if r["score"]["json_valid"]) / len(subset),
            "exact_match_rate": sum(1 for r in subset if r["score"].get("exact_match")) / len(subset),
            "field_accuracy": fc / ft if ft else 0.0,
        }
    return {
        "num_examples": total,
        "json_valid_rate": json_valid / total if total else 0.0,
        "exact_match_rate": exact / total if total else 0.0,
        "field_accuracy": field_correct / field_total if field_total else 0.0,
        "latency_ms": {"mean": sum(latencies) / len(latencies) if latencies else 0.0, "p50": pct(50), "p95": pct(95)},
        "by_task": by_task,
    }


def evaluate_model(model_path: str, items: list[EvalItem], max_new_tokens: int, torch_dtype: str, device_map: str) -> list[dict[str, Any]]:
    model = LocalChatModel(model_path, torch_dtype=torch_dtype, device_map=device_map)
    rows = []
    for item in items:
        result = model.generate(item.prompt, max_new_tokens=max_new_tokens)
        parsed = extract_json(result["text"])
        score = score_prediction(item.task_type, item.answer, parsed)
        rows.append({
            "id": item.id,
            "task_type": item.task_type,
            "capability": item.capability,
            "model": model_path,
            "expected": item.answer,
            "prediction": parsed,
            "raw_text": result["text"],
            "latency_ms": result["latency_ms"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "score": score,
        })
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local Qwen models on IDN benchmark JSON tasks.")
    parser.add_argument("--dataset", default="data/benchmarks/idn_benchmark_v1_eval.jsonl")
    parser.add_argument("--small-model", default="/home/sig/models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--big-model", default="/home/sig/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output-path", default="artifacts/eval/idn_benchmark_qwen25_7b_vs_15b.jsonl")
    parser.add_argument("--summary-path", default="artifacts/eval/idn_benchmark_qwen25_7b_vs_15b_summary.json")
    args = parser.parse_args()

    raw = read_jsonl(args.dataset)
    items = [EvalItem(id=r["id"], task_type=r["task_type"], capability=r["capability"], prompt=r["prompt"], answer=r["answer"]) for r in raw]
    all_rows = []
    summary = {"dataset": args.dataset, "num_examples": len(items), "models": {}}
    for role, model_path in [("small", args.small_model), ("big", args.big_model)]:
        rows = evaluate_model(model_path, items, args.max_new_tokens, args.torch_dtype, args.device_map)
        for row in rows:
            row["role"] = role
        all_rows.extend(rows)
        summary["models"][role] = {"model": model_path, **summarize(rows)}
    write_jsonl(args.output_path, all_rows)
    ensure_parent(args.summary_path)
    with Path(args.summary_path).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
