"""Capability registry for Phase 0 config discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from idn.config import (
    DEFAULT_CAPABILITY_DIR,
    DEFAULT_MACHINE_CONFIG,
    CapabilityConfig,
    MachineConfig,
    discover_capability_configs,
    load_resolved_config,
)


@dataclass(frozen=True)
class CapabilityRegistration:
    id: str
    config_path: Path
    config: CapabilityConfig
    machine: MachineConfig


def load_registry(
    config_dir: Path = DEFAULT_CAPABILITY_DIR,
    machine_path: Path = DEFAULT_MACHINE_CONFIG,
) -> Dict[str, CapabilityRegistration]:
    registrations = {}
    for config_path in discover_capability_configs(config_dir):
        resolved = load_resolved_config(config_path, machine_path)
        capability_id = resolved.capability.id
        if capability_id in registrations:
            raise ValueError("Duplicate capability id: {0}".format(capability_id))
        registrations[capability_id] = CapabilityRegistration(
            id=capability_id,
            config_path=config_path,
            config=resolved.capability,
            machine=resolved.machine,
        )
    return dict(sorted(registrations.items()))


def get_capability(
    capability_id: str,
    config_dir: Path = DEFAULT_CAPABILITY_DIR,
    machine_path: Path = DEFAULT_MACHINE_CONFIG,
) -> Optional[CapabilityRegistration]:
    return load_registry(config_dir=config_dir, machine_path=machine_path).get(
        capability_id
    )


def iter_capabilities(
    config_dir: Path = DEFAULT_CAPABILITY_DIR,
    machine_path: Path = DEFAULT_MACHINE_CONFIG,
) -> Iterable[CapabilityRegistration]:
    return load_registry(config_dir=config_dir, machine_path=machine_path).values()

