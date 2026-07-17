"""GSM8K arithmetic reasoning capability."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from idn.capabilities.base import Example
from idn.capabilities.classification import DatasetLoadError
from idn.config import CapabilityConfig


NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")
HASH_ANSWER_RE = re.compile(r"####\s*([-+]?(?:\d[\d,]*)(?:\.\d+)?)")
CALCULATOR_ANNOTATION_RE = re.compile(r"<<[^<>]*>>")


@dataclass
class GSM8KSpec:
    """CapabilitySpec implementation for GSM8K."""

    config: CapabilityConfig

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def metric_name(self) -> str:
        return self.config.metric_name

    @property
    def higher_is_better(self) -> bool:
        return self.config.higher_is_better

    def load_split(self, split: str, limit: Optional[int] = None) -> List[Example]:
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise DatasetLoadError(
                "datasets is required to load capability data"
            ) from exc

        dataset_dict = load_from_disk(self.config.dataset_path)
        dataset = dataset_dict[split]
        examples = []
        for index, row in enumerate(dataset):
            examples.append(
                Example(
                    id=str(row.get("idx", index)),
                    prompt_inputs={"question": row["question"]},
                    reference=parse_gsm8k_answer(row["answer"]),
                    gold_target=str(row["answer"]),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
        return examples

    def format_prompt(self, ex: Example) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You solve grade-school math problems carefully. "
                    "Show the reasoning, then finish with exactly '#### <answer>'."
                ),
            },
            {
                "role": "user",
                "content": "{0}\n\nQuestion: {1}".format(
                    self.config.prompt_template,
                    ex.prompt_inputs["question"],
                ),
            },
        ]

    def parse_prediction(self, raw_text: str) -> Optional[str]:
        return parse_gsm8k_answer(raw_text)

    def target_text(self, ex: Example, teacher_text: Optional[str] = None) -> str:
        if teacher_text:
            parsed = self.parse_prediction(teacher_text)
            if parsed is not None:
                return teacher_text.strip()
        if ex.gold_target:
            return clean_gsm8k_rationale(ex.gold_target)
        return "#### {0}".format(ex.reference)

    def metric(self, preds: List[Any], refs: List[Any]) -> Dict[str, float]:
        if len(preds) != len(refs):
            raise ValueError("preds and refs must have the same length")
        if not refs:
            return {"exact_match": 0.0}
        correct = 0
        for pred, ref in zip(preds, refs):
            correct += int(normalize_number(pred) == normalize_number(ref))
        return {"exact_match": correct / len(refs)}

    def verify(self, raw_text: str, parsed: Any) -> bool:
        return parsed is not None


def parse_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the final GSM8K numeric answer from text."""

    if text is None:
        return None
    hash_matches = HASH_ANSWER_RE.findall(text)
    if hash_matches:
        return normalize_number(hash_matches[-1])
    number_matches = NUMBER_RE.findall(text)
    if not number_matches:
        return None
    return normalize_number(number_matches[-1])


def clean_gsm8k_rationale(text: str) -> str:
    """Remove GSM8K calculator annotations while preserving readable reasoning."""

    return CALCULATOR_ANNOTATION_RE.sub("", str(text)).strip()


def normalize_number(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        normalized = "0"
    return normalized
