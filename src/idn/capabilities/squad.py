"""SQuAD extractive question answering capability."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from idn.capabilities.base import Example
from idn.capabilities.classification import DatasetLoadError
from idn.config import CapabilityConfig


@dataclass
class SQuADSpec:
    """CapabilitySpec implementation for SQuAD."""

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
                    id=str(row.get("id", index)),
                    prompt_inputs={
                        "title": row.get("title", ""),
                        "context": row["context"],
                        "question": row["question"],
                    },
                    reference=_answer_texts(row["answers"]),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
        return examples

    def format_prompt(self, ex: Example) -> List[Dict[str, str]]:
        content = (
            "{template}\n\nContext: {context}\n\nQuestion: {question}\n\n"
            "Return only the answer span."
        ).format(
            template=self.config.prompt_template,
            context=ex.prompt_inputs["context"],
            question=ex.prompt_inputs["question"],
        )
        return [
            {
                "role": "system",
                "content": "You answer questions using only the provided context.",
            },
            {"role": "user", "content": content},
        ]

    def parse_prediction(self, raw_text: str) -> str:
        return clean_answer_text(raw_text)

    def target_text(self, ex: Example, teacher_text: Optional[str] = None) -> str:
        if teacher_text:
            parsed = self.parse_prediction(teacher_text)
            if parsed:
                return parsed
        references = list(ex.reference)
        return references[0] if references else ""

    def metric(self, preds: List[Any], refs: List[Any]) -> Dict[str, float]:
        if len(preds) != len(refs):
            raise ValueError("preds and refs must have the same length")
        if not refs:
            return {"exact_match": 0.0, "f1": 0.0}
        exact = 0.0
        f1 = 0.0
        for pred, ref in zip(preds, refs):
            references = _normalize_references(ref)
            exact += max((squad_exact_match(pred, item) for item in references), default=0.0)
            f1 += max((squad_f1(pred, item) for item in references), default=0.0)
        return {"exact_match": exact / len(refs), "f1": f1 / len(refs)}

    def verify(self, raw_text: str, parsed: Any) -> bool:
        return bool(parsed)


def _answer_texts(raw_answers: Any) -> List[str]:
    if isinstance(raw_answers, dict):
        texts = raw_answers.get("text", [])
        return [str(item) for item in texts if str(item)]
    if isinstance(raw_answers, (list, tuple)):
        return [str(item) for item in raw_answers if str(item)]
    return [str(raw_answers)] if raw_answers else []


def _normalize_references(ref: Any) -> List[str]:
    if isinstance(ref, dict):
        return _answer_texts(ref)
    if isinstance(ref, (list, tuple)):
        return [str(item) for item in ref]
    return [str(ref)]


def clean_answer_text(text: Any) -> str:
    if text is None:
        return ""
    answer = str(text).strip()
    answer = re.sub(r"^(answer|final answer)\s*:\s*", "", answer, flags=re.I).strip()
    return answer.strip("\"'` \n\t")


def normalize_squad_text(text: Any) -> str:
    value = "" if text is None else str(text).lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def squad_exact_match(pred: Any, ref: Any) -> float:
    return float(normalize_squad_text(pred) == normalize_squad_text(ref))


def squad_f1(pred: Any, ref: Any) -> float:
    pred_tokens = normalize_squad_text(pred).split()
    ref_tokens = normalize_squad_text(ref).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
