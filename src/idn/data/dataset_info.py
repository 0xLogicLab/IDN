"""Dataset inventory helpers for configured capabilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from idn.registry import CapabilityRegistration


DATASET_METADATA: Dict[str, Dict[str, str]] = {
    "cnn_dailymail": {
        "about": (
            "News articles paired with human-written highlights from CNN and "
            "Daily Mail."
        ),
        "used_for": "Abstractive summarization and ROUGE-L evaluation.",
    },
    "cola": {
        "about": (
            "English sentences annotated for grammatical acceptability from "
            "the Corpus of Linguistic Acceptability."
        ),
        "used_for": "Binary grammatical acceptability classification.",
    },
    "gsm8k": {
        "about": (
            "Grade-school math word problems with step-by-step solutions and "
            "numeric final answers."
        ),
        "used_for": "Arithmetic reasoning and exact-match answer evaluation.",
    },
    "mbpp": {
        "about": (
            "Introductory Python programming prompts with reference code and "
            "assertion-based tests."
        ),
        "used_for": "Code generation evaluated by unit-test pass rate.",
    },
    "squad": {
        "about": (
            "Wikipedia passages with questions and extractive answer spans "
            "from the Stanford Question Answering Dataset."
        ),
        "used_for": "Extractive question answering with exact-match and F1 metrics.",
    },
    "sst2": {
        "about": (
            "Movie-review sentences labeled with positive or negative "
            "sentiment from the GLUE SST-2 task."
        ),
        "used_for": "Binary sentiment classification.",
    },
}


@dataclass(frozen=True)
class PartitionInfo:
    name: str
    num_samples: Optional[int]


@dataclass(frozen=True)
class LocalDatasetInfo:
    status: str
    partitions: List[PartitionInfo]
    error: Optional[str] = None

    @property
    def num_partitions(self) -> Optional[int]:
        if self.status != "ok":
            return None
        return len(self.partitions)

    @property
    def total_samples(self) -> Optional[int]:
        if self.status != "ok":
            return None
        values = [partition.num_samples for partition in self.partitions]
        if any(value is None for value in values):
            return None
        return sum(int(value) for value in values)


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    name: str
    about: str
    used_for: str
    dataset_path: str
    train_split: str
    eval_split: str
    eval_limit: Optional[int]
    metric_name: str
    secondary_metrics: List[str]
    higher_is_better: bool
    target_source: str
    prompt_template: str
    local: LocalDatasetInfo


def collect_dataset_infos(
    registrations: Iterable[CapabilityRegistration],
    inspect_local: bool = True,
) -> List[DatasetInfo]:
    return [
        dataset_info_for_registration(row, inspect_local=inspect_local)
        for row in registrations
    ]


def dataset_info_for_registration(
    registration: CapabilityRegistration,
    inspect_local: bool = True,
) -> DatasetInfo:
    config = registration.config
    metadata = DATASET_METADATA.get(
        config.id,
        {
            "about": "No static description is registered for this dataset.",
            "used_for": "Capability evaluation.",
        },
    )
    if inspect_local:
        local = inspect_local_dataset(config.dataset_path)
    else:
        local = LocalDatasetInfo(
            status="not inspected",
            partitions=[],
        )
    return DatasetInfo(
        id=config.id,
        name=config.name,
        about=metadata["about"],
        used_for=metadata["used_for"],
        dataset_path=config.dataset_path,
        train_split=config.train_split,
        eval_split=config.eval_split,
        eval_limit=config.eval_limit,
        metric_name=config.metric_name,
        secondary_metrics=list(config.secondary_metrics),
        higher_is_better=config.higher_is_better,
        target_source=config.target_source,
        prompt_template=config.prompt_template,
        local=local,
    )


def inspect_local_dataset(dataset_path: str) -> LocalDatasetInfo:
    path = Path(dataset_path)
    if not path.exists():
        return LocalDatasetInfo(
            status="missing",
            partitions=[],
            error="path does not exist",
        )

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        return LocalDatasetInfo(
            status="unavailable",
            partitions=[],
            error="datasets is not installed: {0}".format(exc),
        )

    try:
        loaded = load_from_disk(str(path))
    except Exception as exc:  # pragma: no cover - exact dataset errors vary.
        return LocalDatasetInfo(
            status="error",
            partitions=[],
            error="{0}: {1}".format(type(exc).__name__, exc),
        )

    return LocalDatasetInfo(status="ok", partitions=_partitions_from_loaded(loaded))


def _partitions_from_loaded(loaded: Any) -> List[PartitionInfo]:
    if isinstance(loaded, Mapping):
        return [
            PartitionInfo(name=str(name), num_samples=_safe_len(dataset))
            for name, dataset in loaded.items()
        ]
    if hasattr(loaded, "keys") and hasattr(loaded, "__getitem__"):
        return [
            PartitionInfo(name=str(name), num_samples=_safe_len(loaded[name]))
            for name in loaded.keys()
        ]
    return [PartitionInfo(name="dataset", num_samples=_safe_len(loaded))]


def _safe_len(value: Any) -> Optional[int]:
    try:
        return int(len(value))
    except TypeError:
        return None


def format_dataset_infos(infos: Sequence[DatasetInfo]) -> str:
    if not infos:
        return "No datasets found."

    sections = []
    for info in infos:
        sections.append(_format_dataset_info(info))
    return "\n\n".join(sections)


def _format_dataset_info(info: DatasetInfo) -> str:
    metrics = [info.metric_name] + list(info.secondary_metrics)
    lines = [
        "{0} - {1}".format(info.id, info.name),
        "  about: {0}".format(info.about),
        "  used_for: {0}".format(info.used_for),
        "  dataset_path: {0}".format(info.dataset_path),
        "  configured_splits: train={0}, eval={1}".format(
            info.train_split,
            info.eval_split,
        ),
        "  partitions: {0}".format(_format_partitions(info.local)),
        "  samples: {0}".format(_format_samples(info.local)),
        "  metrics: {0} ({1})".format(
            ", ".join(metrics),
            "higher is better" if info.higher_is_better else "lower is better",
        ),
        "  eval_limit: {0}".format(
            "all" if info.eval_limit is None else str(info.eval_limit)
        ),
        "  target_source: {0}".format(info.target_source),
    ]
    if info.local.error:
        lines.append(
            "  local_status: {0} ({1})".format(info.local.status, info.local.error)
        )
    else:
        lines.append("  local_status: {0}".format(info.local.status))
    return "\n".join(lines)


def _format_partitions(local: LocalDatasetInfo) -> str:
    if local.status != "ok":
        return "unavailable"
    if not local.partitions:
        return "0"
    return "{0} ({1})".format(
        len(local.partitions),
        ", ".join(partition.name for partition in local.partitions),
    )


def _format_samples(local: LocalDatasetInfo) -> str:
    if local.status != "ok":
        return "unavailable"
    if not local.partitions:
        return "0"
    parts = [
        "{0}={1}".format(
            partition.name,
            "unknown" if partition.num_samples is None else partition.num_samples,
        )
        for partition in local.partitions
    ]
    total = local.total_samples
    if total is not None:
        parts.append("total={0}".format(total))
    return ", ".join(parts)


def dataset_infos_to_json(infos: Sequence[DatasetInfo]) -> str:
    return json.dumps([_dataset_info_to_dict(info) for info in infos], indent=2)


def _dataset_info_to_dict(info: DatasetInfo) -> Dict[str, Any]:
    payload = asdict(info)
    payload["local"]["num_partitions"] = info.local.num_partitions
    payload["local"]["total_samples"] = info.local.total_samples
    return payload
