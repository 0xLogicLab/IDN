from __future__ import annotations

from typing import Any

from idn.utils import EvalExample, read_jsonl, stable_id


def _alpaca_prompt(row: dict[str, Any]) -> tuple[str, str | None]:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    output = row.get("output")
    if input_text:
        prompt = f"{instruction}\n\nInput:\n{input_text}"
    else:
        prompt = instruction
    return prompt, str(output).strip() if output is not None else None


def _generic_prompt(row: dict[str, Any]) -> tuple[str, str | None]:
    for key in ("prompt", "question", "instruction", "query", "text"):
        value = row.get(key)
        if value:
            prompt = str(value).strip()
            break
    else:
        prompt = str(row).strip()

    reference = None
    for key in ("reference", "answer", "output", "response", "completion"):
        value = row.get(key)
        if value:
            reference = str(value).strip()
            break
    return prompt, reference


def load_eval_examples(dataset_name: str, split: str, num_samples: int) -> list[EvalExample]:
    if dataset_name.endswith(".jsonl") or "/" in dataset_name and dataset_name.startswith((".", "~", "/")):
        path = dataset_name.replace("~", str(__import__("pathlib").Path.home()), 1)
        rows = read_jsonl(path)
        if num_samples > 0:
            rows = rows[:num_samples]
        examples = []
        for row in rows:
            prompt, reference = _generic_prompt(row)
            request_id = str(row.get("request_id") or stable_id(path + prompt))
            examples.append(EvalExample(request_id=request_id, prompt=prompt, reference=reference, source=path))
        return examples

    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    examples: list[EvalExample] = []
    for row in ds:
        if dataset_name == "tatsu-lab/alpaca":
            prompt, reference = _alpaca_prompt(row)
        else:
            prompt, reference = _generic_prompt(row)
        if not prompt:
            continue
        examples.append(
            EvalExample(
                request_id=stable_id(dataset_name + split + prompt),
                prompt=prompt,
                reference=reference,
                source=dataset_name,
            )
        )
    return examples
