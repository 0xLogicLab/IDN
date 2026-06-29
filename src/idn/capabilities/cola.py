"""CoLA grammatical acceptability capability."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from idn.capabilities.base import Example
from idn.capabilities.classification import BinaryLabelCapability, binary_accuracy
from idn.eval.metrics import matthews_corrcoef


class CoLASpec(BinaryLabelCapability):
    """CapabilitySpec implementation for CoLA."""

    def _row_to_example(self, index: int, row: Dict[str, Any], label: int) -> Example:
        example_id = str(row.get("idx", index))
        return Example(
            id=example_id,
            prompt_inputs={"sentence": row["sentence"]},
            reference=label,
        )

    def _prompt_body(self, ex: Example) -> str:
        return (
            "{0}\n\nSentence: {1}\n\nAnswer yes or no.".format(
                self.config.prompt_template,
                ex.prompt_inputs["sentence"],
            )
        )

    def parse_prediction(self, raw_text: str) -> Optional[int]:
        return self._parse_from_aliases(
            raw_text=raw_text,
            positive_aliases=["yes", "acceptable"],
            negative_aliases=["no", "unacceptable"],
        )

    def _label_text(self, label: int) -> str:
        return "yes" if int(label) == 1 else "no"

    def metric(self, preds: List[Any], refs: List[Any]) -> Dict[str, float]:
        normalized_preds = [None if pred is None else int(pred) for pred in preds]
        normalized_refs = [int(ref) for ref in refs]
        metrics = binary_accuracy(normalized_preds, normalized_refs)
        mcc_preds = [
            pred if pred is not None else 1 - ref
            for pred, ref in zip(normalized_preds, normalized_refs)
        ]
        metrics["mcc"] = matthews_corrcoef(mcc_preds, normalized_refs)
        return metrics
