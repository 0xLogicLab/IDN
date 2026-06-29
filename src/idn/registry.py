"""Capability registry and config discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from idn.capabilities.base import CapabilitySpec
from idn.capabilities.cola import CoLASpec
from idn.capabilities.gsm8k import GSM8KSpec
from idn.capabilities.sst2 import SST2Spec
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
    spec: Optional[CapabilitySpec] = None


SPEC_FACTORIES: Dict[str, Callable[[CapabilityConfig], CapabilitySpec]] = {
    "cola": CoLASpec,
    "gsm8k": GSM8KSpec,
    "sst2": SST2Spec,
}


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
            spec=_build_spec(resolved.capability),
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


def get_spec(
    capability_id: str,
    config_dir: Path = DEFAULT_CAPABILITY_DIR,
    machine_path: Path = DEFAULT_MACHINE_CONFIG,
) -> Optional[CapabilitySpec]:
    registration = get_capability(
        capability_id=capability_id,
        config_dir=config_dir,
        machine_path=machine_path,
    )
    if registration is None:
        return None
    return registration.spec


def spec_from_config(config: CapabilityConfig) -> Optional[CapabilitySpec]:
    return _build_spec(config)


def _build_spec(config: CapabilityConfig) -> Optional[CapabilitySpec]:
    factory = SPEC_FACTORIES.get(config.id)
    if factory is None:
        return None
    return factory(config)
