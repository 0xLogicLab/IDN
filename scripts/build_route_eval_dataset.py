#!/usr/bin/env python3
"""Build a 1,000-example mixed holdout for end-to-end IDN evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idn.analysis.embedding_viz import normalize_request_text, request_text_from_row
from idn.capabilities.gsm8k import parse_gsm8k_answer
from idn.config import discover_capability_configs, load_resolved_config


TOTAL_SAMPLES = 1000
EVALUATED_PREFIX = 1000

# Sources are ordered from cleanest to fallback. No training split is ever allowed.
SOURCE_PLAN: Mapping[str, Sequence[Tuple[str, int, Optional[int], bool]]] = {
    "cnn_dailymail": (("test", 0, None, False),),
    "cola": (
        ("validation", EVALUATED_PREFIX, None, False),
        ("validation", 0, EVALUATED_PREFIX, True),
    ),
    "gsm8k": (("test", EVALUATED_PREFIX, None, False),),
    "mbpp": (
        ("validation", 0, None, False),
        ("prompt", 0, None, False),
        ("test", 0, None, True),
    ),
    "squad": (("validation", EVALUATED_PREFIX, None, False),),
    "sst2": (("validation", 0, None, True),),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a mixed end-to-end evaluation JSONL without using any capability "
            "training split."
        )
    )
    parser.add_argument("--machine", default="3090")
    parser.add_argument("--config-dir", default="configs/capabilities")
    parser.add_argument(
        "--output", default="artifacts/eval/idn_mixed_1000.jsonl"
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--override", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    manifest_path = output_path.with_suffix(".manifest.json")
    if not args.override and (output_path.exists() or manifest_path.exists()):
        raise SystemExit(
            "output already exists; pass --override: {0}".format(output_path)
        )

    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise SystemExit("datasets is required: {0}".format(exc))

    registrations = []
    for config_path in discover_capability_configs(Path(args.config_dir)):
        resolved = load_resolved_config(config_path, args.machine)
        registrations.append(resolved.capability)
    registrations.sort(key=lambda capability: capability.id)
    actual_ids = [capability.id for capability in registrations]
    expected_ids = sorted(SOURCE_PLAN)
    if actual_ids != expected_ids:
        raise SystemExit(
            "source plan does not match configured capabilities: expected {0}, got {1}".format(
                expected_ids, actual_ids
            )
        )

    quotas = balanced_quotas(actual_ids, TOTAL_SAMPLES)
    rng = random.Random(args.seed)
    records: List[Dict[str, Any]] = []
    seen_texts = set()

    for capability in registrations:
        dataset = load_from_disk(capability.dataset_path)
        if not isinstance(dataset, DatasetDict):
            raise SystemExit(
                "expected DatasetDict for {0}: {1}".format(
                    capability.id, capability.dataset_path
                )
            )
        selected = select_capability_records(
            capability=capability,
            dataset=dataset,
            quota=quotas[capability.id],
            rng=rng,
            max_chars=args.max_chars,
            seen_texts=seen_texts,
        )
        records.extend(selected)

    rng.shuffle(records)
    if len(records) != TOTAL_SAMPLES:
        raise SystemExit(
            "internal error: expected {0} records, built {1}".format(
                TOTAL_SAMPLES, len(records)
            )
        )
    for index, record in enumerate(records):
        record["mixed_index"] = index

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_path, records)
    manifest = build_manifest(
        output_path=output_path,
        records=records,
        quotas=quotas,
        seed=args.seed,
        max_chars=args.max_chars,
    )
    write_json_atomic(manifest_path, manifest)

    print("End-to-end IDN evaluation dataset created.")
    print("samples: {0}".format(len(records)))
    print("output: {0}".format(output_path))
    print("manifest: {0}".format(manifest_path))
    print("capabilities: {0}".format(json.dumps(quotas, sort_keys=True)))
    print(
        "possible prior-eval overlap: {0}".format(
            manifest["possible_prior_eval_overlap_samples"]
        )
    )
    return 0


def balanced_quotas(capabilities: Sequence[str], total: int) -> Dict[str, int]:
    base, remainder = divmod(total, len(capabilities))
    return {
        capability: base + int(index < remainder)
        for index, capability in enumerate(capabilities)
    }


def select_capability_records(
    *,
    capability: Any,
    dataset: Any,
    quota: int,
    rng: random.Random,
    max_chars: int,
    seen_texts: set,
) -> List[Dict[str, Any]]:
    selected = []
    for split, start_index, stop_index, possible_prior_eval in SOURCE_PLAN[capability.id]:
        if split == capability.train_split:
            raise SystemExit(
                "refusing to use training split for {0}: {1}".format(
                    capability.id, split
                )
            )
        if split not in dataset:
            raise SystemExit(
                "missing planned split for {0}: {1}".format(capability.id, split)
            )
        partition = dataset[split]
        stop = len(partition) if stop_index is None else min(stop_index, len(partition))
        indices = list(range(start_index, stop))
        rng.shuffle(indices)
        for source_index in indices:
            row = dict(partition[source_index])
            payload = normalize_request_text(
                request_text_from_row(capability.id, row), max_chars=max_chars
            )
            if not payload:
                continue
            text = normalize_request_text(
                "{0}\n\n{1}".format(capability.prompt_template, payload),
                max_chars=max_chars,
            )
            dedupe_key = text.casefold()
            if dedupe_key in seen_texts:
                continue
            task = task_fields(capability.id, row)
            if task is None:
                continue
            seen_texts.add(dedupe_key)
            selected.append(
                {
                    "capability": capability.id,
                    "example_id": source_example_id(row, source_index),
                    "text": text,
                    "payload": payload,
                    "prompt_template": capability.prompt_template,
                    "prompt_inputs": task["prompt_inputs"],
                    "reference": task["reference"],
                    "gold_target": task["gold_target"],
                    "source": {
                        "dataset_path": capability.dataset_path,
                        "split": split,
                        "index": source_index,
                        "example_id": source_example_id(row, source_index),
                    },
                    "possible_prior_eval_overlap": possible_prior_eval,
                }
            )
            if len(selected) == quota:
                return selected
    raise SystemExit(
        "not enough eligible samples for {0}: need {1}, found {2}".format(
            capability.id, quota, len(selected)
        )
    )


def task_fields(capability_id: str, row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return fields sufficient to rebuild CapabilitySpec.Example and score output."""

    if capability_id == "cnn_dailymail":
        article = str(row.get("article", ""))
        highlights = str(row.get("highlights", ""))
        if not article or not highlights:
            return None
        return {
            "prompt_inputs": {"article": article},
            "reference": highlights,
            "gold_target": highlights,
        }
    if capability_id in ("cola", "sst2"):
        sentence = str(row.get("sentence", ""))
        label = int(row.get("label", -1))
        if not sentence or label not in (0, 1):
            return None
        if capability_id == "cola":
            target = "yes" if label == 1 else "no"
        else:
            target = "positive" if label == 1 else "negative"
        return {
            "prompt_inputs": {"sentence": sentence},
            "reference": label,
            "gold_target": target,
        }
    if capability_id == "gsm8k":
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        reference = parse_gsm8k_answer(answer)
        if not question or reference is None:
            return None
        return {
            "prompt_inputs": {"question": question},
            "reference": reference,
            "gold_target": answer,
        }
    if capability_id == "mbpp":
        prompt = str(row.get("prompt", row.get("text", "")))
        tests = string_list(row.get("test_list", []))
        setup_code = str(row.get("test_setup_code", "") or "")
        code = str(row.get("code", "") or "")
        if not prompt or not tests or not code:
            return None
        return {
            "prompt_inputs": {
                "prompt": prompt,
                "tests": tests,
                "test_setup_code": setup_code,
            },
            "reference": {
                "code": code,
                "tests": tests,
                "test_setup_code": setup_code,
            },
            "gold_target": code,
        }
    if capability_id == "squad":
        context = str(row.get("context", ""))
        question = str(row.get("question", ""))
        answers = row.get("answers", {})
        if isinstance(answers, Mapping):
            references = [str(item) for item in answers.get("text", []) if str(item)]
        else:
            references = [str(item) for item in answers or [] if str(item)]
        if not context or not question or not references:
            return None
        return {
            "prompt_inputs": {
                "title": str(row.get("title", "")),
                "context": context,
                "question": question,
            },
            "reference": references,
            "gold_target": references[0],
        }
    raise ValueError("unsupported capability: {0}".format(capability_id))


def string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def source_example_id(row: Mapping[str, Any], fallback: int) -> str:
    for key in ("id", "idx", "task_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(fallback)


def build_manifest(
    *,
    output_path: Path,
    records: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, int],
    seed: int,
    max_chars: int,
) -> Dict[str, Any]:
    capability_counts = Counter(record["capability"] for record in records)
    source_counts = Counter(
        "{0}:{1}".format(record["capability"], record["source"]["split"])
        for record in records
    )
    overlap = sum(
        int(bool(record["possible_prior_eval_overlap"])) for record in records
    )
    return {
        "artifact_version": 1,
        "purpose": "end-to-end IDN evaluation across six capabilities",
        "output": str(output_path),
        "total_samples": len(records),
        "requested_quotas": dict(quotas),
        "capability_counts": dict(sorted(capability_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "seed": seed,
        "max_chars": max_chars,
        "excluded_training_splits": True,
        "evaluated_prefix_excluded": EVALUATED_PREFIX,
        "possible_prior_eval_overlap_samples": overlap,
        "possible_prior_eval_overlap_reason": (
            "CoLA and SST-2 have no unused public labeled test targets; MBPP has "
            "only 50 labeled non-training rows outside test. Labeled validation/test "
            "rows are reused only to fill those capability quotas."
        ),
        "task_references_included": True,
        "notes": (
            "Each record includes prompt_inputs, a metric-ready reference, and the "
            "gold target needed to evaluate routing and downstream generation."
        ),
    }


def write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(".{0}.tmp".format(path.name))
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(".{0}.tmp".format(path.name))
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
