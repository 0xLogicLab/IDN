"""Shared helpers for simple classification capabilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from idn.capabilities.base import Example
from idn.config import CapabilityConfig
from idn.eval.metrics import accuracy_score


class DatasetLoadError(RuntimeError):
    """Raised when a configured local dataset cannot be loaded."""


@dataclass
class BinaryLabelCapability:
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
            label = int(row["label"])
            if label < 0:
                continue
            examples.append(self._row_to_example(index=index, row=row, label=label))
            if limit is not None and len(examples) >= limit:
                break
        return examples

    def format_prompt(self, ex: Example) -> List[Dict[str, str]]:
        body = self._prompt_body(ex)
        return [
            {
                "role": "system",
                "content": "You are a careful classifier. Return only the requested label.",
            },
            {"role": "user", "content": body},
        ]

    def target_text(self, ex: Example, teacher_text: Optional[str] = None) -> str:
        if teacher_text:
            parsed = self.parse_prediction(teacher_text)
            if parsed is not None:
                return self._label_text(parsed)
        return self._label_text(int(ex.reference))

    def verify(self, raw_text: str, parsed: Any) -> bool:
        return parsed in (0, 1)

    def _row_to_example(self, index: int, row: Dict[str, Any], label: int) -> Example:
        raise NotImplementedError

    def _prompt_body(self, ex: Example) -> str:
        raise NotImplementedError

    def _label_text(self, label: int) -> str:
        raise NotImplementedError

    def _parse_from_aliases(
        self,
        raw_text: str,
        positive_aliases: List[str],
        negative_aliases: List[str],
    ) -> Optional[int]:
        text = raw_text.strip().lower()
        if not text:
            return None

        tokens = re.findall(r"[a-z]+", text)
        for token in tokens:
            if token in positive_aliases:
                return 1
            if token in negative_aliases:
                return 0
        return None


def binary_accuracy(preds: List[Optional[int]], refs: List[int]) -> Dict[str, float]:
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    normalized = [
        pred if pred is not None else 1 - int(ref)
        for pred, ref in zip(preds, refs)
    ]
    return {"accuracy": accuracy_score([int(pred) for pred in normalized], refs)}
