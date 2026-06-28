from __future__ import annotations

import argparse
import gc
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class EvalItem:
    id: str
    task_type: str
    capability: str
    prompt: str
    answer: dict[str, Any]
    difficulty: str = "unknown"


KEY_FIELDS = {
    "capability_descriptor_extraction": ["capability", "task", "input_modalities", "output_modalities", "quality_tier", "latency_class", "privacy_class", "resource_requirement"],
    "route_decision": ["route", "fallback_required"],
    "fallback_decision": ["accept_local", "fallback_required", "target"],
    "resource_fit": ["feasible", "route_tier"],
    "cache_decision": ["action", "cache_type"],
    "trust_policy": ["allowed"],
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower().replace("-", "_")
    if isinstance(value, list):
        return sorted(normalize_value(v) for v in value)
    return value


def score_prediction(task_type: str, expected: dict[str, Any], predicted: dict[str, Any] | None) -> dict[str, Any]:
    fields = KEY_FIELDS.get(task_type, list(expected.keys()))
    if predicted is None:
        return {"json_valid": False, "field_correct": 0, "field_total": len(fields), "field_accuracy": 0.0, "exact_match": False, "field_scores": {field: False for field in fields}}
    field_scores = {field: normalize_value(predicted.get(field)) == normalize_value(expected.get(field)) for field in fields}
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
    def __init__(self, model_path: str, adapter_path: str | None = None, torch_dtype: str = "float16", device_map: str = "auto") -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        dtype_map = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32, "auto": "auto"}
        dtype = dtype_map.get(torch_dtype.lower(), torch.float16)
        tokenizer_path = adapter_path if adapter_path else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
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
            {"role": "system", "content": "You are a strict JSON API. Output only valid JSON and no prose."},
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
        return {"text": self.tokenizer.decode(generated, skip_special_tokens=True).strip(), "latency_ms": latency_ms, "input_tokens": int(inputs["input_ids"].shape[-1]), "output_tokens": int(generated.shape[-1])}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    latencies = sorted(r["latency_ms"] for r in rows)
    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, max(0, round((p / 100) * (len(latencies) - 1))))
        return float(latencies[idx])
    field_correct = sum(r["score"].get("field_correct", 0) for r in rows)
    field_total = sum(r["score"].get("field_total", 0) for r in rows)
    by_task = {}
    for task in sorted({r["task_type"] for r in rows}):
        subset = [r for r in rows if r["task_type"] == task]
        fc = sum(r["score"].get("field_correct", 0) for r in subset)
        ft = sum(r["score"].get("field_total", 0) for r in subset)
        by_task[task] = {"num_examples": len(subset), "json_valid_rate": sum(r["score"]["json_valid"] for r in subset) / len(subset), "exact_match_rate": sum(r["score"]["exact_match"] for r in subset) / len(subset), "field_accuracy": fc / ft if ft else 0.0}
    return {"num_examples": total, "json_valid_rate": sum(r["score"]["json_valid"] for r in rows) / total if total else 0.0, "exact_match_rate": sum(r["score"]["exact_match"] for r in rows) / total if total else 0.0, "field_accuracy": field_correct / field_total if field_total else 0.0, "latency_ms": {"mean": sum(latencies) / len(latencies) if latencies else 0.0, "p50": pct(50), "p95": pct(95)}, "by_task": by_task}


def validator_accept(score: dict[str, Any], item: EvalItem, prediction: dict[str, Any] | None, min_field_accuracy: float) -> bool:
    if not score.get("json_valid") or prediction is None:
        return False
    if score.get("field_accuracy", 0.0) < min_field_accuracy:
        return False
    if item.task_type in {"route_decision", "fallback_decision", "trust_policy", "resource_fit"}:
        return bool(score.get("exact_match"))
    if item.difficulty == "hard":
        return bool(score.get("exact_match"))
    return True


def evaluate(role: str, model_path: str, adapter_path: str | None, items: list[EvalItem], max_new_tokens: int, torch_dtype: str, device_map: str, min_field_accuracy: float) -> list[dict[str, Any]]:
    model = LocalChatModel(model_path, adapter_path, torch_dtype, device_map)
    rows = []
    for item in items:
        result = model.generate(item.prompt, max_new_tokens)
        parsed = extract_json(result["text"])
        score = score_prediction(item.task_type, item.answer, parsed)
        rows.append({"id": item.id, "task_type": item.task_type, "capability": item.capability, "difficulty": item.difficulty, "role": role, "model": model_path, "adapter": adapter_path, "expected": item.answer, "prediction": parsed, "raw_text": result["text"], "latency_ms": result["latency_ms"], "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"], "score": score, "validator_accept": validator_accept(score, item, parsed, min_field_accuracy)})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def build_router_validator(tuned_rows: list[dict[str, Any]], teacher_rows: list[dict[str, Any]]) -> dict[str, Any]:
    teacher_by_id = {r["id"]: r for r in teacher_rows}
    accepted = [r for r in tuned_rows if r["validator_accept"]]
    routed_rows = []
    for row in tuned_rows:
        if row["validator_accept"]:
            routed = {**row, "served_by": "small_tuned"}
        else:
            teacher = teacher_by_id[row["id"]]
            routed = {**teacher, "served_by": "teacher_fallback", "fallback_from": "small_tuned"}
        routed_rows.append(routed)
    summary = summarize(routed_rows)
    summary["coverage_local"] = len(accepted) / len(tuned_rows) if tuned_rows else 0.0
    summary["fallback_rate"] = 1.0 - summary["coverage_local"]
    return {"summary": summary, "rows": routed_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base, tuned LoRA, teacher, and Router+Validator fallback.")
    parser.add_argument("--dataset", default="data/benchmarks/idn_benchmark_v2_eval.jsonl")
    parser.add_argument("--small-model", default="/data/huggingface/model/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--teacher-model", default="/data/huggingface/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="artifacts/adapters/qwen25_15b_idn_router_validator_lora")
    parser.add_argument("--output-path", default="artifacts/eval/idn_router_validator_lora_eval.jsonl")
    parser.add_argument("--summary-path", default="artifacts/eval/idn_router_validator_lora_eval_summary.json")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--min-field-accuracy", type=float, default=1.0)
    parser.add_argument("--skip-teacher", action="store_true")
    args = parser.parse_args()

    raw = read_jsonl(args.dataset)
    if args.max_samples > 0:
        raw = raw[: args.max_samples]
    items = [EvalItem(id=r["id"], task_type=r["task_type"], capability=r["capability"], prompt=r["prompt"], answer=r["answer"], difficulty=r.get("difficulty", "unknown")) for r in raw]

    base_rows = evaluate("small_base", args.small_model, None, items, args.max_new_tokens, args.torch_dtype, args.device_map, args.min_field_accuracy)
    tuned_rows = evaluate("small_tuned_lora", args.small_model, args.adapter, items, args.max_new_tokens, args.torch_dtype, args.device_map, args.min_field_accuracy)
    teacher_rows = [] if args.skip_teacher else evaluate("teacher_7b", args.teacher_model, None, items, args.max_new_tokens, args.torch_dtype, args.device_map, args.min_field_accuracy)

    all_rows = base_rows + tuned_rows + teacher_rows
    summary: dict[str, Any] = {"dataset": args.dataset, "num_examples": len(items), "models": {"small_base": summarize(base_rows), "small_tuned_lora": summarize(tuned_rows)}}
    if teacher_rows:
        summary["models"]["teacher_7b"] = summarize(teacher_rows)
        routed = build_router_validator(tuned_rows, teacher_rows)
        for row in routed["rows"]:
            row["role"] = "router_validator"
        all_rows.extend(routed["rows"])
        summary["models"]["router_validator"] = routed["summary"]

    write_jsonl(args.output_path, all_rows)
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.summary_path).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
