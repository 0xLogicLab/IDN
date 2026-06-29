"""SST-2 sentiment classification capability."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from idn.capabilities.base import Example
from idn.capabilities.classification import BinaryLabelCapability, binary_accuracy


class SST2Spec(BinaryLabelCapability):
    """CapabilitySpec implementation for SST-2."""

    def _row_to_example(self, index: int, row: Dict[str, Any], label: int) -> Example:
        example_id = str(row.get("idx", index))
        return Example(
            id=example_id,
            prompt_inputs={"sentence": row["sentence"]},
            reference=label,
        )

    def _prompt_body(self, ex: Example) -> str:
        return (
            "{0}\n\nSentence: {1}\n\nAnswer positive or negative.".format(
                self.config.prompt_template,
                ex.prompt_inputs["sentence"],
            )
        )

    def parse_prediction(self, raw_text: str) -> Optional[int]:
        return self._parse_from_aliases(
            raw_text=raw_text,
            positive_aliases=["positive", "pos"],
            negative_aliases=["negative", "neg"],
        )

    def _label_text(self, label: int) -> str:
        return "positive" if int(label) == 1 else "negative"

    def metric(self, preds: List[Any], refs: List[Any]) -> Dict[str, float]:
        return binary_accuracy(
            [None if pred is None else int(pred) for pred in preds],
            [int(ref) for ref in refs],
        )

