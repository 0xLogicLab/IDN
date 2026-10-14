"""Dataset contract for the runtime benchmark."""

from __future__ import annotations

from typing import Optional, Protocol

from idn.capabilities.base import Example, PromptDataset


class BenchmarkDataset(PromptDataset, Protocol):
    """Prompt dataset with an optional gold capability label for route scoring.

    Each adapter owns its file format, split loading, prompt construction,
    target extraction, output parsing, and task-specific verification. Return
    ``None`` from ``capability_label`` when the dataset has no route labels.
    """

    def capability_label(self, example: Example) -> Optional[str]:
        ...
