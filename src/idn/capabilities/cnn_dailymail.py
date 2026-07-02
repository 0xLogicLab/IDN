"""CNN/DailyMail summarization capability."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from idn.capabilities.base import Example
from idn.capabilities.classification import DatasetLoadError
from idn.config import CapabilityConfig


@dataclass
class CNNDailyMailSpec:
    """CapabilitySpec implementation for CNN/DailyMail."""

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
                    prompt_inputs={"article": row["article"]},
                    reference=str(row["highlights"]),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
        return examples

    def format_prompt(self, ex: Example) -> List[Dict[str, str]]:
        content = "{0}\n\nArticle:\n{1}".format(
            self.config.prompt_template,
            ex.prompt_inputs["article"],
        )
        return [
            {
                "role": "system",
                "content": "You write faithful, concise news summaries.",
            },
            {"role": "user", "content": content},
        ]

    def parse_prediction(self, raw_text: str) -> str:
        return "" if raw_text is None else str(raw_text).strip()

    def target_text(self, ex: Example, teacher_text: Optional[str] = None) -> str:
        if teacher_text and self.parse_prediction(teacher_text):
            return self.parse_prediction(teacher_text)
        return str(ex.reference)

    def metric(self, preds: List[Any], refs: List[Any]) -> Dict[str, float]:
        if len(preds) != len(refs):
            raise ValueError("preds and refs must have the same length")
        if not refs:
            return {"rougeL": 0.0}
        scores = [rouge_l_f1(pred, ref) for pred, ref in zip(preds, refs)]
        return {"rougeL": sum(scores) / len(scores)}

    def verify(self, raw_text: str, parsed: Any) -> bool:
        return bool(parsed)


def rouge_l_f1(pred: Any, ref: Any) -> float:
    pred_tokens = _rouge_tokens(pred)
    ref_tokens = _rouge_tokens(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _rouge_tokens(text: Any) -> List[str]:
    value = "" if text is None else str(text).lower()
    value = value.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    return re.findall(r"\w+", value)


def _lcs_length(left: List[str], right: List[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]
