"""Carve artifact creation for Qwen2 capability extraction.

The first supported path is the rho=1.0 invariant artifact: copy the teacher
checkpoint without changing weights and write a manifest describing the full
unit keep-set. This gives the repacking pipeline a testable anchor before
introducing heterogeneous per-layer shapes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from idn.extraction.units import (
    FFN_CHANNEL,
    KV_HEAD_GROUP,
    LAYER,
    QUERY_HEAD,
    PrunableUnit,
    UnitMap,
    enumerate_qwen2_units,
)


MANIFEST_NAME = "manifest.json"
SUPPORTED_FULL_KEEP_RATIOS = (1.0,)


class CarveError(RuntimeError):
    """Raised when a carve artifact cannot be created."""


@dataclass(frozen=True)
class CarveManifest:
    artifact_version: int
    architecture: str
    carve_type: str
    keep_ratio: float
    teacher_path: str
    teacher_hash: str
    num_units: int
    kept_units: Mapping[str, List[str]]
    dimensions: Mapping[str, int]
    scores_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "artifact_version": self.artifact_version,
            "architecture": self.architecture,
            "carve_type": self.carve_type,
            "keep_ratio": self.keep_ratio,
            "teacher_path": self.teacher_path,
            "teacher_hash": self.teacher_hash,
            "num_units": self.num_units,
            "kept_units": dict(self.kept_units),
            "dimensions": dict(self.dimensions),
        }
        if self.scores_path is not None:
            payload["scores_path"] = self.scores_path
        return payload


def carve_qwen2_full_keep(
    model_path: Path,
    output_dir: Path,
    *,
    scores_path: Optional[Path] = None,
    override: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> CarveManifest:
    """Create a rho=1.0 carve artifact by copying an HF Qwen2 checkpoint."""

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    if not model_path.exists() or not model_path.is_dir():
        raise CarveError("Model directory does not exist: {0}".format(model_path))
    
    if output_dir.exists():
        if not override:
            raise CarveError("Output directory already exists: {0}".format(output_dir))
        _report(progress, "removing existing output directory: {0}".format(output_dir))
        shutil.rmtree(output_dir)

    _report(progress, "reading model config: {0}".format(model_path / "config.json"))
    config = _read_config(model_path)
    _report(progress, "enumerating Qwen2 prunable units")
    unit_map = enumerate_qwen2_units(config)
    _report(progress, "hashing source model files")
    teacher_hash = hash_model_directory(model_path)
    manifest = build_full_keep_manifest(
        model_path=model_path,
        unit_map=unit_map,
        teacher_hash=teacher_hash,
        scores_path=scores_path,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(MANIFEST_NAME)
    _report(progress, "copying model files to: {0}".format(output_dir))
    shutil.copytree(model_path, output_dir, ignore=ignore)
    _report(progress, "writing manifest: {0}".format(output_dir / MANIFEST_NAME))
    write_manifest(output_dir / MANIFEST_NAME, manifest)
    return manifest


def build_full_keep_manifest(
    *,
    model_path: Path,
    unit_map: UnitMap,
    teacher_hash: str,
    scores_path: Optional[Path] = None,
) -> CarveManifest:
    dims = unit_map.dimensions
    return CarveManifest(
        artifact_version=1,
        architecture=unit_map.architecture,
        carve_type="full_keep_copy",
        keep_ratio=1.0,
        teacher_path=str(model_path),
        teacher_hash=teacher_hash,
        num_units=len(unit_map.units),
        kept_units=kept_unit_keys_by_kind(unit_map.units),
        dimensions={
            "num_layers": dims.num_layers,
            "hidden_size": dims.hidden_size,
            "intermediate_size": dims.intermediate_size,
            "num_attention_heads": dims.num_attention_heads,
            "num_key_value_heads": dims.num_key_value_heads,
            "head_dim": dims.head_dim,
            "query_heads_per_kv": dims.query_heads_per_kv,
        },
        scores_path=str(scores_path) if scores_path is not None else None,
    )


def kept_unit_keys_by_kind(units: Iterable[PrunableUnit]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {
        LAYER: [],
        QUERY_HEAD: [],
        KV_HEAD_GROUP: [],
        FFN_CHANNEL: [],
    }
    for unit in units:
        result.setdefault(unit.id.kind, []).append(unit.id.key())
    return result


def write_manifest(path: Path, manifest: CarveManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def hash_model_directory(model_path: Path) -> str:
    """Hash config/tokenizer/weight files that define a local HF model artifact."""

    digest = hashlib.sha256()
    for file_path in _iter_hashable_model_files(model_path):
        relative = file_path.relative_to(model_path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return "sha256:{0}".format(digest.hexdigest())


def validate_keep_ratio(keep_ratio: float) -> float:
    keep_ratio = float(keep_ratio)
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise CarveError("keep_ratio must be in (0, 1]")
    if keep_ratio not in SUPPORTED_FULL_KEEP_RATIOS:
        raise CarveError(
            "Only keep_ratio=1.0 is implemented; heterogeneous repacking comes next"
        )
    return keep_ratio


def _read_config(model_path: Path) -> Mapping[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise CarveError("Missing config.json in model directory: {0}".format(model_path))
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise CarveError("Could not read model config: {0}".format(config_path)) from exc
    if not isinstance(payload, MappingABC):
        raise CarveError("Model config must be a JSON object: {0}".format(config_path))
    return payload


def _iter_hashable_model_files(model_path: Path) -> Sequence[Path]:
    suffixes = {
        ".json",
        ".model",
        ".safetensors",
        ".bin",
        ".txt",
        ".py",
    }
    files = [
        path
        for path in model_path.rglob("*")
        if path.is_file()
        and path.name != MANIFEST_NAME
        and not any(part.startswith(".") for part in path.relative_to(model_path).parts)
        and path.suffix in suffixes
    ]
    return sorted(files, key=lambda path: path.relative_to(model_path).as_posix())


def _report(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)
