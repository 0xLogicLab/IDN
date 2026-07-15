"""Shared capability protocol used by later implementation phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class Example:
    id: str
    prompt_inputs: Dict[str, Any]
    reference: Any
    gold_target: Optional[str] = None


class CapabilitySpec(Protocol):
    id: str
    metric_name: str
    higher_is_better: bool

    def load_split(self, split: str, limit: Optional[int] = None) -> List[Example]:
        ...

    def format_prompt(self, ex: Example) -> List[Dict[str, str]]:
        ...

    def parse_prediction(self, raw_text: str) -> Any:
        ...

    def target_text(self, ex: Example, teacher_text: Optional[str] = None) -> str:
        ...

    def metric(self, preds: List[Any], refs: List[Any]) -> Dict[str, float]:
        ...

    def verify(self, raw_text: str, parsed: Any) -> bool:
        ...
