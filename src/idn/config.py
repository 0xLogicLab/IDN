"""Configuration loading and merge helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import yaml


DEFAULT_CAPABILITY_DIR = Path("configs/capabilities")
DEFAULT_MACHINE = "3090"
MACHINE_CONFIGS = {
    "3090": Path("configs/machine/3090.yaml"),
    "h100": Path("configs/machine/h100.yaml"),
}
DEFAULT_MACHINE_CONFIG = MACHINE_CONFIGS[DEFAULT_MACHINE]
DEFAULT_ARTIFACT_DIR = Path("artifacts")
MACHINE_FIELDS = {
    "name",
    "model_dir",
    "data_dir",
    "device",
    "train_batch_size",
    "eval_batch_size",
    "local_files_only",
}


@dataclass(frozen=True)
class CapabilityConfig:
    id: str
    name: str
    metric_name: str
    secondary_metrics: List[str]
    higher_is_better: bool
    dataset_path: str
    train_split: str
    eval_split: str
    eval_limit: Optional[int]
    max_new_tokens: int
    epsilon: float
    target_source: str
    prompt_template: str


@dataclass(frozen=True)
class MachineConfig:
    name: str
    model_dir: str
    data_dir: str
    device: str
    train_batch_size: int
    eval_batch_size: int
    local_files_only: bool


@dataclass(frozen=True)
class ResolvedConfig:
    capability: CapabilityConfig
    machine: MachineConfig
    raw: Dict[str, Any]


class ConfigError(ValueError):
    """Raised when a config file is missing required fields or is invalid."""


PathLike = Union[str, Path]


def load_yaml(path: PathLike) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError("Config file does not exist: {0}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigError("Config file must contain a YAML mapping: {0}".format(path))
    return loaded


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without mutating either input."""

    result = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_resolved_config(
    capability_path: PathLike,
    machine_path: Optional[PathLike] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ResolvedConfig:
    """Load capability YAML and merge machine settings over it."""

    capability_path = Path(capability_path)
    machine_path = resolve_machine_config_path(machine_path)
    raw = deep_merge(load_yaml(capability_path), load_machine_yaml(machine_path))
    if overrides:
        raw = deep_merge(raw, overrides)

    capability_raw = _required_mapping(raw, "capability", capability_path)
    machine_raw = _required_mapping(raw, "machine", machine_path)
    machine = _machine_config_from_raw(machine_raw)
    dataset_path = resolve_relative_dir(
        base_dir=machine.data_dir,
        relative_dir=_required_str(capability_raw, "dataset_path"),
        field="capability.dataset_path",
    )

    capability = CapabilityConfig(
        id=_required_str(capability_raw, "id"),
        name=_required_str(capability_raw, "name"),
        metric_name=_required_str(capability_raw, "metric_name"),
        secondary_metrics=_optional_str_list(capability_raw, "secondary_metrics"),
        higher_is_better=_required_bool(capability_raw, "higher_is_better"),
        dataset_path=dataset_path,
        train_split=_required_str(capability_raw, "train_split"),
        eval_split=_required_str(capability_raw, "eval_split"),
        eval_limit=_optional_int(capability_raw, "eval_limit"),
        max_new_tokens=_required_int(capability_raw, "max_new_tokens"),
        epsilon=float(capability_raw.get("epsilon", 0.0)),
        target_source=_required_str(capability_raw, "target_source"),
        prompt_template=_prompt_template(capability_raw),
    )
    return ResolvedConfig(capability=capability, machine=machine, raw=raw)


def load_machine_yaml(path: PathLike) -> Dict[str, Any]:
    raw = load_yaml(path)
    if "machine" in raw:
        machine = _required_mapping(raw, "machine", Path(path))
        return {"machine": dict(machine)}
    return {"machine": raw}


def load_machine_config(machine: Optional[PathLike] = None) -> MachineConfig:
    """Load and validate a machine config without requiring a capability config."""

    machine_path = resolve_machine_config_path(machine)
    raw = load_machine_yaml(machine_path)
    machine_raw = _required_mapping(raw, "machine", machine_path)
    return _machine_config_from_raw(machine_raw)


def resolve_machine_config_path(machine: Optional[PathLike] = None) -> Path:
    if machine is None:
        return DEFAULT_MACHINE_CONFIG
    machine_value = str(machine)
    if machine_value in MACHINE_CONFIGS:
        return MACHINE_CONFIGS[machine_value]
    return Path(machine)


def resolve_relative_dir(base_dir: str, relative_dir: str, field: str) -> str:
    relative_path = Path(relative_dir)
    if relative_path.is_absolute():
        raise ConfigError("{0} must be relative, got: {1}".format(field, relative_dir))
    if ".." in relative_path.parts:
        raise ConfigError("{0} must not contain '..': {1}".format(field, relative_dir))
    return str(Path(base_dir) / relative_path)


def discover_capability_configs(
    config_dir: Path = DEFAULT_CAPABILITY_DIR,
) -> Iterable[Path]:
    if not config_dir.exists():
        return []
    return sorted(config_dir.glob("*.yaml"))


def _required_mapping(raw: Mapping[str, Any], key: str, source: Path) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError("Missing mapping '{0}' in {1}".format(key, source))
    return value


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError("Missing string field: {0}".format(key))
    return value


def _required_int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ConfigError("Missing integer field: {0}".format(key))
    return value


def _optional_int(raw: Mapping[str, Any], key: str) -> Optional[int]:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ConfigError("Field must be an integer or null: {0}".format(key))
    return value


def _required_bool(raw: Mapping[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ConfigError("Missing boolean field: {0}".format(key))
    return value


def _machine_config_from_raw(raw: Mapping[str, Any]) -> MachineConfig:
    _reject_unknown_fields(raw, MACHINE_FIELDS, "machine")
    return MachineConfig(
        name=_required_str(raw, "name"),
        model_dir=_required_str(raw, "model_dir"),
        data_dir=_required_str(raw, "data_dir"),
        device=_required_str(raw, "device"),
        train_batch_size=_required_int(raw, "train_batch_size"),
        eval_batch_size=_required_int(raw, "eval_batch_size"),
        local_files_only=_required_bool(raw, "local_files_only"),
    )


def _optional_str_list(raw: Mapping[str, Any], key: str) -> List[str]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError("Field must be a list of strings: {0}".format(key))
    return list(value)


def _reject_unknown_fields(
    raw: Mapping[str, Any],
    allowed: Iterable[str],
    label: str,
) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ConfigError(
            "Unknown {0} field(s): {1}".format(label, ", ".join(unknown))
        )


def _prompt_template(raw: Mapping[str, Any]) -> str:
    prompt = raw.get("prompt")
    if not isinstance(prompt, Mapping):
        return ""
    template = prompt.get("template", "")
    if not isinstance(template, str):
        raise ConfigError("Field must be a string: prompt.template")
    return template
