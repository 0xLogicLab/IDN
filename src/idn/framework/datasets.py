"""Pluggable dataset registry for existing task specs and chat JSONL data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from idn.capabilities.base import PromptDataset


DatasetFactory = Callable[..., PromptDataset]


class DatasetRegistry:
    """Create dataset adapters by a stable name instead of hard-coding tasks."""

    def __init__(self) -> None:
        self._factories: Dict[str, DatasetFactory] = {}

    def register(self, name: str, factory: DatasetFactory) -> None:
        if not name.strip():
            raise ValueError("dataset name cannot be empty")
        if name in self._factories:
            raise ValueError("dataset factory already registered: {0}".format(name))
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> PromptDataset:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(
                "unknown dataset {0}; registered: {1}".format(
                    name, ", ".join(self.names())
                )
            ) from exc
        return factory(**kwargs)

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._factories))


def build_default_dataset_registry(
    *,
    config_dir: Optional[Path] = None,
    machine_path: Optional[Path] = None,
) -> DatasetRegistry:
    """Register the six current task specs plus a generic chat JSONL adapter."""

    from idn.capabilities.jsonl_chat import JsonlChatDataset
    from idn.config import DEFAULT_CAPABILITY_DIR, DEFAULT_MACHINE_CONFIG
    from idn.registry import SPEC_FACTORIES, get_spec

    resolved_config_dir = Path(config_dir or DEFAULT_CAPABILITY_DIR)
    resolved_machine_path = Path(machine_path or DEFAULT_MACHINE_CONFIG)
    registry = DatasetRegistry()
    registry.register("jsonl_chat", JsonlChatDataset)

    for capability_id in SPEC_FACTORIES:
        def legacy_factory(
            *,
            _capability_id: str = capability_id,
            **kwargs: Any
        ) -> PromptDataset:
            local_config_dir = Path(kwargs.pop("config_dir", resolved_config_dir))
            local_machine_path = Path(kwargs.pop("machine_path", resolved_machine_path))
            if kwargs:
                raise TypeError(
                    "unexpected dataset options: {0}".format(", ".join(sorted(kwargs)))
                )
            spec = get_spec(
                _capability_id,
                config_dir=local_config_dir,
                machine_path=local_machine_path,
            )
            if spec is None:
                raise ValueError(
                    "no capability implementation registered for {0}".format(
                        _capability_id
                    )
                )
            return spec

        registry.register(capability_id, legacy_factory)
    return registry
