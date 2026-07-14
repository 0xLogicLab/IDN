"""Carve artifact creation for Qwen2 capability extraction."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from idn.extraction.units import (
    FFN_CHANNEL,
    KV_HEAD_GROUP,
    LAYER,
    QUERY_HEAD,
    PrunableUnit,
    UnitMap,
    enumerate_qwen2_units,
    qwen2_dimensions,
)


MANIFEST_NAME = "manifest.json"
PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
SAFETENSORS_WEIGHTS_NAME = "model.safetensors"


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
    unit_keep_ratios: Optional[Mapping[str, float]] = None

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
        if self.unit_keep_ratios is not None:
            payload["unit_keep_ratios"] = dict(self.unit_keep_ratios)
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


def carve_qwen2_homogeneous(
    model_path: Path,
    output_dir: Path,
    *,
    keep_ratio: float,
    attention_keep_ratio: Optional[float] = None,
    ffn_keep_ratio: Optional[float] = None,
    scores_path: Optional[Path] = None,
    override: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> CarveManifest:
    """Create a standard-HF-loadable Qwen2 carve with uniform reduced shapes."""

    keep_ratio = validate_keep_ratio(keep_ratio)
    attention_keep_ratio = validate_keep_ratio(
        1.0 if attention_keep_ratio is None else attention_keep_ratio
    )
    ffn_keep_ratio = validate_keep_ratio(
        keep_ratio if ffn_keep_ratio is None else ffn_keep_ratio
    )
    if attention_keep_ratio == 1.0 and ffn_keep_ratio == 1.0:
        return carve_qwen2_full_keep(
            model_path=model_path,
            output_dir=output_dir,
            scores_path=scores_path,
            override=override,
            progress=progress,
        )

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
    config = dict(_read_config(model_path))
    dims = qwen2_dimensions(config)
    _report(progress, "loading carve scores")
    scores = load_unit_scores(scores_path)
    keep_set = select_homogeneous_keep_set(
        dims,
        keep_ratio,
        scores,
        attention_keep_ratio=attention_keep_ratio,
        ffn_keep_ratio=ffn_keep_ratio,
    )
    _report(progress, "loading source model weights")
    state_dict, weight_format = load_model_state_dict(model_path)
    _report(progress, "slicing Qwen2 weights")
    carved_state = slice_qwen2_state_dict(state_dict, dims, keep_set)
    carved_config = build_carved_qwen2_config(config, dims, keep_set)
    unit_map = enumerate_qwen2_units(config)
    _report(progress, "hashing source model files")
    teacher_hash = hash_model_directory(model_path)
    manifest = build_carve_manifest(
        model_path=model_path,
        unit_map=unit_map,
        teacher_hash=teacher_hash,
        keep_ratio=keep_ratio,
        keep_set=keep_set,
        dimensions={
            "num_layers": dims.num_layers,
            "hidden_size": dims.hidden_size,
            "intermediate_size": keep_set.intermediate_size,
            "num_attention_heads": keep_set.num_query_heads,
            "num_key_value_heads": keep_set.num_kv_groups,
            "head_dim": dims.head_dim,
            "query_heads_per_kv": dims.query_heads_per_kv,
        },
        carve_type="homogeneous_repack",
        scores_path=scores_path,
        unit_keep_ratios={
            "attention": attention_keep_ratio,
            "ffn": ffn_keep_ratio,
        },
    )

    _report(
        progress,
        "copying tokenizer and auxiliary files to: {0}".format(output_dir),
    )
    copy_model_auxiliary_files(model_path, output_dir)
    _report(progress, "writing carved config: {0}".format(output_dir / "config.json"))
    write_json(output_dir / "config.json", carved_config)
    _report(progress, "writing carved weights")
    save_model_state_dict(output_dir, carved_state, weight_format)
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


def build_carve_manifest(
    *,
    model_path: Path,
    unit_map: UnitMap,
    teacher_hash: str,
    keep_ratio: float,
    keep_set: "HomogeneousKeepSet",
    dimensions: Mapping[str, int],
    carve_type: str,
    scores_path: Optional[Path] = None,
    unit_keep_ratios: Optional[Mapping[str, float]] = None,
) -> CarveManifest:
    kept = {
        LAYER: [unit.id.key() for unit in unit_map.by_kind(LAYER)],
        QUERY_HEAD: [
            unit.id.key()
            for unit in unit_map.by_kind(QUERY_HEAD)
            if unit.id.index in keep_set.layers[unit.id.layer].query_heads
        ],
        KV_HEAD_GROUP: [
            unit.id.key()
            for unit in unit_map.by_kind(KV_HEAD_GROUP)
            if unit.id.index in keep_set.layers[unit.id.layer].kv_groups
        ],
        FFN_CHANNEL: [
            unit.id.key()
            for unit in unit_map.by_kind(FFN_CHANNEL)
            if unit.id.index in keep_set.layers[unit.id.layer].ffn_channels
        ],
    }
    return CarveManifest(
        artifact_version=2,
        architecture=unit_map.architecture,
        carve_type=carve_type,
        keep_ratio=keep_ratio,
        teacher_path=str(model_path),
        teacher_hash=teacher_hash,
        num_units=sum(len(items) for items in kept.values()),
        kept_units=kept,
        dimensions=dimensions,
        scores_path=str(scores_path) if scores_path is not None else None,
        unit_keep_ratios=unit_keep_ratios,
    )


@dataclass(frozen=True)
class LayerKeepSet:
    query_heads: Tuple[int, ...]
    kv_groups: Tuple[int, ...]
    ffn_channels: Tuple[int, ...]


@dataclass(frozen=True)
class HomogeneousKeepSet:
    layers: Mapping[int, LayerKeepSet]
    num_query_heads: int
    num_kv_groups: int
    intermediate_size: int


def load_unit_scores(scores_path: Optional[Path]) -> Mapping[str, float]:
    if scores_path is None:
        return {}
    scores_path = Path(scores_path)
    if not scores_path.exists():
        raise CarveError("Scores path does not exist: {0}".format(scores_path))
    if scores_path.suffix == ".json":
        try:
            payload = json.loads(scores_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CarveError("Could not read scores JSON: {0}".format(scores_path)) from exc
        if not isinstance(payload, MappingABC):
            raise CarveError("Scores JSON must be an object: {0}".format(scores_path))
        raw_scores = payload.get("scores", payload)
        if not isinstance(raw_scores, MappingABC):
            raise CarveError("Scores JSON field 'scores' must be an object")
        return {str(key): float(value) for key, value in raw_scores.items()}
    if scores_path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise CarveError("safetensors is required to load score files") from exc
        result: Dict[str, float] = {}
        with safe_open(str(scores_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if tensor.numel() == 1:
                    result[key] = float(tensor.item())
                elif tensor.ndim == 1:
                    for index, value in enumerate(tensor.tolist()):
                        result["{0}:{1}".format(key, index)] = float(value)
                else:
                    raise CarveError(
                        "Unsupported score tensor shape for key: {0}".format(key)
                    )
        return result
    raise CarveError("Unsupported scores file type: {0}".format(scores_path))


def select_homogeneous_keep_set(
    dims: Any,
    keep_ratio: float,
    scores: Mapping[str, float],
    *,
    attention_keep_ratio: Optional[float] = None,
    ffn_keep_ratio: Optional[float] = None,
) -> HomogeneousKeepSet:
    attention_keep_ratio = validate_keep_ratio(
        1.0 if attention_keep_ratio is None else attention_keep_ratio
    )
    ffn_keep_ratio = validate_keep_ratio(
        keep_ratio if ffn_keep_ratio is None else ffn_keep_ratio
    )
    keep_query_per_kv = _kept_count(
        dims.query_heads_per_kv,
        attention_keep_ratio,
    )
    keep_ffn = _kept_count(dims.intermediate_size, ffn_keep_ratio)
    layers: Dict[int, LayerKeepSet] = {}
    for layer in range(dims.num_layers):
        kv_groups = tuple(range(dims.num_key_value_heads))
        query_heads = []
        for kv_group in kv_groups:
            query_start = kv_group * dims.query_heads_per_kv
            selected_offsets = _select_indices(
                dims.query_heads_per_kv,
                keep_query_per_kv,
                lambda offset, layer=layer, query_start=query_start: _score_index(
                    scores,
                    QUERY_HEAD,
                    layer,
                    query_start + offset,
                ),
            )
            query_heads.extend(query_start + offset for offset in selected_offsets)
        ffn_channels = _select_indices(
            dims.intermediate_size,
            keep_ffn,
            lambda index, layer=layer: _score_index(
                scores,
                FFN_CHANNEL,
                layer,
                index,
            ),
        )
        layers[layer] = LayerKeepSet(
            query_heads=tuple(query_heads),
            kv_groups=tuple(sorted(kv_groups)),
            ffn_channels=tuple(sorted(ffn_channels)),
        )
    return HomogeneousKeepSet(
        layers=layers,
        num_query_heads=dims.num_key_value_heads * keep_query_per_kv,
        num_kv_groups=dims.num_key_value_heads,
        intermediate_size=keep_ffn,
    )


def slice_qwen2_state_dict(
    state_dict: Mapping[str, Any],
    dims: Any,
    keep_set: HomogeneousKeepSet,
) -> Dict[str, Any]:
    import torch

    layer_indices = {}
    for layer, layer_keep in keep_set.layers.items():
        query_rows = torch.tensor(
            [
                dim
                for head in layer_keep.query_heads
                for dim in range(head * dims.head_dim, (head + 1) * dims.head_dim)
            ],
            dtype=torch.long,
        )
        kv_rows = torch.tensor(
            [
                dim
                for kv_group in layer_keep.kv_groups
                for dim in range(
                    kv_group * dims.head_dim,
                    (kv_group + 1) * dims.head_dim,
                )
            ],
            dtype=torch.long,
        )
        ffn = torch.tensor(layer_keep.ffn_channels, dtype=torch.long)
        layer_indices[layer] = (query_rows, kv_rows, ffn)

    carved: Dict[str, Any] = {}
    for name, tensor in state_dict.items():
        value = tensor.detach().cpu()
        layer = _layer_from_name(name)
        if layer is None:
            carved[name] = value
            continue
        query_rows, kv_rows, ffn = layer_indices[layer]
        if name.endswith(".self_attn.q_proj.weight"):
            carved[name] = _index_select(value, 0, query_rows)
        elif name.endswith(".self_attn.q_proj.bias"):
            carved[name] = _index_select(value, 0, query_rows)
        elif name.endswith(".self_attn.k_proj.weight") or name.endswith(
            ".self_attn.v_proj.weight"
        ):
            carved[name] = _index_select(value, 0, kv_rows)
        elif name.endswith(".self_attn.k_proj.bias") or name.endswith(
            ".self_attn.v_proj.bias"
        ):
            carved[name] = _index_select(value, 0, kv_rows)
        elif name.endswith(".self_attn.o_proj.weight"):
            carved[name] = _index_select(value, 1, query_rows)
        elif name.endswith(".mlp.gate_proj.weight") or name.endswith(
            ".mlp.up_proj.weight"
        ):
            carved[name] = _index_select(value, 0, ffn)
        elif name.endswith(".mlp.down_proj.weight"):
            carved[name] = _index_select(value, 1, ffn)
        else:
            carved[name] = value
    return carved


def build_carved_qwen2_config(
    config: Mapping[str, Any],
    dims: Any,
    keep_set: HomogeneousKeepSet,
) -> Dict[str, Any]:
    carved = dict(config)
    carved["hidden_size"] = dims.hidden_size
    carved["intermediate_size"] = keep_set.intermediate_size
    carved["num_attention_heads"] = keep_set.num_query_heads
    carved["num_key_value_heads"] = keep_set.num_kv_groups
    carved["head_dim"] = dims.head_dim
    carved["architectures"] = ["Qwen2ForCausalLM"]
    carved["idn_carve"] = {
        "version": 2,
        "mode": "homogeneous",
        "source_dimensions": {
            "num_layers": dims.num_layers,
            "hidden_size": dims.hidden_size,
            "intermediate_size": dims.intermediate_size,
            "num_attention_heads": dims.num_attention_heads,
            "num_key_value_heads": dims.num_key_value_heads,
            "head_dim": dims.head_dim,
            "query_heads_per_kv": dims.query_heads_per_kv,
        },
        "kept_query_heads_by_layer": {
            str(layer): list(layer_keep.query_heads)
            for layer, layer_keep in keep_set.layers.items()
        },
        "kept_kv_groups_by_layer": {
            str(layer): list(layer_keep.kv_groups)
            for layer, layer_keep in keep_set.layers.items()
        },
        "kept_ffn_channels_by_layer": {
            str(layer): list(layer_keep.ffn_channels)
            for layer, layer_keep in keep_set.layers.items()
        },
    }
    if carved["num_attention_heads"] % carved["num_key_value_heads"] != 0:
        raise CarveError(
            "Internal carve error: query heads are not divisible by KV groups"
        )
    return carved


def load_model_state_dict(model_path: Path) -> Tuple[Mapping[str, Any], str]:
    safetensors_path = model_path / SAFETENSORS_WEIGHTS_NAME
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise CarveError("safetensors is required to load model.safetensors") from exc
        return load_file(str(safetensors_path), device="cpu"), "safetensors"
    bin_path = model_path / PYTORCH_WEIGHTS_NAME
    if bin_path.exists():
        try:
            import torch
        except ImportError as exc:
            raise CarveError("torch is required to load pytorch_model.bin") from exc
        return torch.load(str(bin_path), map_location="cpu"), "bin"
    raise CarveError(
        "Expected {0} or {1} in model directory: {2}".format(
            SAFETENSORS_WEIGHTS_NAME,
            PYTORCH_WEIGHTS_NAME,
            model_path,
        )
    )


def save_model_state_dict(
    output_dir: Path,
    state_dict: Mapping[str, Any],
    weight_format: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if weight_format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise CarveError("safetensors is required to save model.safetensors") from exc
        save_file(
            dict(state_dict),
            str(output_dir / SAFETENSORS_WEIGHTS_NAME),
            metadata={"format": "pt"},
        )
        return
    if weight_format == "bin":
        import torch

        torch.save(dict(state_dict), str(output_dir / PYTORCH_WEIGHTS_NAME))
        return
    raise CarveError("Unsupported weight format: {0}".format(weight_format))


def copy_model_auxiliary_files(model_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_names = {
        MANIFEST_NAME,
        SAFETENSORS_WEIGHTS_NAME,
        PYTORCH_WEIGHTS_NAME,
        "config.json",
    }
    for source in model_path.iterdir():
        if source.name in skip_names:
            continue
        destination = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.is_file():
            shutil.copy2(source, destination)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    return keep_ratio


def _kept_count(size: int, keep_ratio: float) -> int:
    return max(1, min(size, int(math.ceil(size * keep_ratio))))


def _select_indices(
    size: int,
    count: int,
    score_fn: Callable[[int], float],
) -> Tuple[int, ...]:
    ranked = sorted(
        range(size),
        key=lambda index: (score_fn(index), -index),
        reverse=True,
    )
    return tuple(sorted(ranked[:count]))


def _score_index(
    scores: Mapping[str, float],
    kind: str,
    layer: int,
    index: int,
) -> float:
    key = "{0}:{1}:{2}".format(kind, layer, index)
    return float(scores.get(key, 0.0))


def _index_select(tensor: Any, dim: int, index: Any) -> Any:
    if tensor.ndim <= dim:
        raise CarveError(
            "Cannot slice tensor with shape {0} on dim {1}".format(
                tuple(tensor.shape),
                dim,
            )
        )
    return tensor.index_select(dim, index.to(tensor.device))


def _layer_from_name(name: str) -> Optional[int]:
    match = re.match(r"model\.layers\.(\d+)\.", name)
    if match is None:
        return None
    return int(match.group(1))


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
