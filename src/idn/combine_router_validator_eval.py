from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

KEY_FIELDS = {
    "capability_descriptor_extraction": ["capability", "task", "input_modalities", "output_modalities", "quality_tier", "latency_class", "privacy_class", "resource_requirement"],
    "route_decision": ["route", "fallback_required"],
    "fallback_decision": ["accept_local", "fallback_required", "target"],
    "resource_fit": ["feasible", "route_tier"],
    "cache_decision": ["action", "cache_type"],
    "trust_policy": ["allowed"],
}

STRICT_TASKS = {"route_decision", "fallback_decision", "trust_policy", "resource_fit"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    return {"json_valid": True, "field_correct": correct, "field_total": len(fields), "field_accuracy": correct / len(fields) if fields else 0.0, "exact_match": correct == len(fields), "field_scores": field_scores}


def validator_accept(row: dict[str, Any], min_field_accuracy: float) -> bool:
    score = row["score"]
    if not score.get("json_valid"):
        return False
    if score.get("field_accuracy", 0.0) < min_field_accuracy:
        return False
    if row["task_type"] in STRICT_TASKS:
        return bool(score.get("exact_match"))
    if row.get("difficulty") == "hard":
        return bool(score.get("exact_match"))
    return True


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    latencies = sorted(float(r.get("latency_ms", 0.0)) for r in rows)
    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, max(0, round((p / 100) * (len(latencies) - 1))))
        return latencies[idx]
    field_correct = sum(r["score"].get("field_correct", 0) for r in rows)
    field_total = sum(r["score"].get("field_total", 0) for r in rows)
    by_task = {}
    for task in sorted({r["task_type"] for r in rows}):
        subset = [r for r in rows if r["task_type"] == task]
        fc = sum(r["score"].get("field_correct", 0) for r in subset)
        ft = sum(r["score"].get("field_total", 0) for r in subset)
        by_task[task] = {"num_examples": len(subset), "json_valid_rate": sum(r["score"]["json_valid"] for r in subset) / len(subset), "exact_match_rate": sum(r["score"]["exact_match"] for r in subset) / len(subset), "field_accuracy": fc / ft if ft else 0.0}
    return {"num_examples": total, "json_valid_rate": sum(r["score"]["json_valid"] for r in rows) / total if total else 0.0, "exact_match_rate": sum(r["score"]["exact_match"] for r in rows) / total if total else 0.0, "field_accuracy": field_correct / field_total if field_total else 0.0, "latency_ms": {"mean": sum(latencies) / len(latencies) if latencies else 0.0, "p50": pct(50), "p95": pct(95)}, "by_task": by_task}


def keyed(rows: list[dict[str, Any]], role_pattern: str | None = None, model_contains: str | None = None) -> dict[str, dict[str, Any]]:
    out = {}
    for r in rows:
        if role_pattern and not re.search(role_pattern, str(r.get("role", ""))):
            continue
        if model_contains and model_contains not in str(r.get("model", "")):
            continue
        out[r["id"]] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Combine prior teacher/base eval rows with new tuned-student rows into Router+Validator metrics.")
    p.add_argument("--prior", default="artifacts/eval/idn_benchmark_v2_qwen25_7b_vs_15b_strict.jsonl")
    p.add_argument("--tuned", default="artifacts/eval/idn_router_validator_lora_fp32_eval_40_no_teacher.jsonl")
    p.add_argument("--summary", default="artifacts/eval/idn_router_validator_combined_summary.json")
    p.add_argument("--output", default="artifacts/eval/idn_router_validator_combined.jsonl")
    p.add_argument("--min-field-accuracy", type=float, default=1.0)
    args = p.parse_args()

    prior = read_jsonl(args.prior)
    tuned_eval = read_jsonl(args.tuned)
    tuned_by_id = keyed(tuned_eval, role_pattern="small_tuned")
    ids = set(tuned_by_id)
    base_by_id = {k: v for k, v in keyed(tuned_eval, role_pattern="small_base").items() if k in ids}
    if not base_by_id:
        base_by_id = {k: v for k, v in keyed(prior, role_pattern="small|base", model_contains="1.5B").items() if k in ids}
    teacher_by_id = {k: v for k, v in keyed(prior, role_pattern="big|teacher", model_contains="7B").items() if k in ids}

    for row in tuned_by_id.values():
        row["validator_accept"] = validator_accept(row, args.min_field_accuracy)

    routed = []
    for id_, tuned in tuned_by_id.items():
        if tuned["validator_accept"]:
            routed.append({**tuned, "role": "router_validator", "served_by": "small_tuned"})
        elif id_ in teacher_by_id:
            routed.append({**teacher_by_id[id_], "role": "router_validator", "served_by": "teacher_fallback", "fallback_from": "small_tuned"})

    all_rows = list(base_by_id.values()) + list(tuned_by_id.values()) + list(teacher_by_id.values()) + routed
    summary = {
        "num_examples": len(ids),
        "prior_teacher_source": args.prior,
        "tuned_source": args.tuned,
        "acceptance_policy": {"min_field_accuracy": args.min_field_accuracy, "strict_tasks_require_exact_match": sorted(STRICT_TASKS)},
        "models": {
            "small_base": summarize(list(base_by_id.values())),
            "small_tuned_lora": summarize(list(tuned_by_id.values())),
            "teacher_7b_prior": summarize(list(teacher_by_id.values())),
            "router_validator": {**summarize(routed), "coverage_local": sum(r.get("served_by") == "small_tuned" for r in routed) / len(routed) if routed else 0.0, "fallback_rate": sum(r.get("served_by") == "teacher_fallback" for r in routed) / len(routed) if routed else 0.0},
        },
    }
    write_jsonl(args.output, all_rows)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.summary).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
