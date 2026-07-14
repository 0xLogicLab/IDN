"""Prunable-unit enumeration for physically repackable Qwen2 carving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


QUERY_HEAD = "query_head"
KV_HEAD_GROUP = "kv_head_group"
FFN_CHANNEL = "ffn_channel"
LAYER = "layer"


@dataclass(frozen=True)
class Qwen2Dimensions:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    query_heads_per_kv: int


@dataclass(frozen=True)
class TensorSlice:
    """A contiguous slice in one model tensor touched by a prunable unit."""

    tensor: str
    dim: int
    start: int
    stop: int

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class UnitId:
    kind: str
    layer: int
    index: int

    def key(self) -> str:
        return "{0}:{1}:{2}".format(self.kind, self.layer, self.index)


@dataclass(frozen=True)
class PrunableUnit:
    id: UnitId
    slices: Tuple[TensorSlice, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class UnitMap:
    architecture: str
    dimensions: Qwen2Dimensions
    units: Tuple[PrunableUnit, ...]

    def by_kind(self, kind: str) -> List[PrunableUnit]:
        return [unit for unit in self.units if unit.id.kind == kind]

    def by_key(self) -> Dict[str, PrunableUnit]:
        return {unit.id.key(): unit for unit in self.units}


def qwen2_dimensions(config: Any) -> Qwen2Dimensions:
    """Return validated structured-pruning dimensions from a Qwen2 config."""

    model_type = _config_value(config, "model_type", None)
    if model_type not in (None, "qwen2"):
        raise ValueError("Expected a qwen2 config, got model_type={0!r}".format(model_type))

    num_layers = _required_positive_int(config, "num_hidden_layers")
    hidden_size = _required_positive_int(config, "hidden_size")
    intermediate_size = _required_positive_int(config, "intermediate_size")
    num_heads = _required_positive_int(config, "num_attention_heads")
    num_kv_heads = _required_positive_int(config, "num_key_value_heads")
    carve_metadata = _config_value(config, "idn_carve", None)
    head_dim = _config_value(config, "head_dim", None)
    if head_dim is None and isinstance(carve_metadata, Mapping):
        source_dimensions = carve_metadata.get("source_dimensions", {})
        if isinstance(source_dimensions, Mapping):
            head_dim = source_dimensions.get("head_dim")
    if head_dim is None:
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        head_dim = hidden_size // num_heads
    else:
        head_dim = int(head_dim)
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if carve_metadata is None and head_dim * num_heads != hidden_size:
            raise ValueError("head_dim * num_attention_heads must equal hidden_size")

    if num_heads % num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    return Qwen2Dimensions(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        query_heads_per_kv=num_heads // num_kv_heads,
    )


def enumerate_qwen2_units(config: Any) -> UnitMap:
    """Enumerate Qwen2 structured units and the tensor slices they own."""

    dims = qwen2_dimensions(config)
    units: List[PrunableUnit] = []
    for layer in range(dims.num_layers):
        units.append(_layer_unit(layer))
        for head in range(dims.num_attention_heads):
            units.append(_query_head_unit(dims, layer, head))
        for kv_head in range(dims.num_key_value_heads):
            units.append(_kv_group_unit(dims, layer, kv_head))
        for channel in range(dims.intermediate_size):
            units.append(_ffn_channel_unit(layer, channel))
    return UnitMap(architecture="qwen2", dimensions=dims, units=tuple(units))


def q_head_to_kv_group(config_or_dims: Any, query_head: int) -> int:
    dims = _as_dims(config_or_dims)
    _validate_index(query_head, dims.num_attention_heads, "query_head")
    return query_head // dims.query_heads_per_kv


def kv_group_query_heads(config_or_dims: Any, kv_group: int) -> Tuple[int, ...]:
    dims = _as_dims(config_or_dims)
    _validate_index(kv_group, dims.num_key_value_heads, "kv_group")
    start = kv_group * dims.query_heads_per_kv
    return tuple(range(start, start + dims.query_heads_per_kv))


def validate_gqa_keep_sets(
    config_or_dims: Any,
    kept_query_heads: Mapping[int, Iterable[int]],
    kept_kv_groups: Mapping[int, Iterable[int]],
) -> None:
    """Validate that kept query heads never reference a pruned KV group."""

    dims = _as_dims(config_or_dims)
    for layer in range(dims.num_layers):
        q_heads = set(kept_query_heads.get(layer, ()))
        kv_groups = set(kept_kv_groups.get(layer, ()))
        for query_head in q_heads:
            _validate_index(query_head, dims.num_attention_heads, "query_head")
            required_kv = q_head_to_kv_group(dims, query_head)
            if required_kv not in kv_groups:
                raise ValueError(
                    "layer {0} keeps query head {1}, but prunes required KV group {2}".format(
                        layer, query_head, required_kv
                    )
                )
        for kv_group in kv_groups:
            _validate_index(kv_group, dims.num_key_value_heads, "kv_group")


def layer_prefix(layer: int) -> str:
    if layer < 0:
        raise ValueError("layer must be non-negative")
    return "model.layers.{0}".format(layer)


def _layer_unit(layer: int) -> PrunableUnit:
    return PrunableUnit(
        id=UnitId(kind=LAYER, layer=layer, index=layer),
        slices=(),
        metadata={"module_prefix": layer_prefix(layer)},
    )


def _query_head_unit(dims: Qwen2Dimensions, layer: int, head: int) -> PrunableUnit:
    start = head * dims.head_dim
    stop = start + dims.head_dim
    prefix = layer_prefix(layer)
    return PrunableUnit(
        id=UnitId(kind=QUERY_HEAD, layer=layer, index=head),
        slices=(
            TensorSlice("{0}.self_attn.q_proj.weight".format(prefix), 0, start, stop),
            TensorSlice("{0}.self_attn.q_proj.bias".format(prefix), 0, start, stop),
            TensorSlice("{0}.self_attn.o_proj.weight".format(prefix), 1, start, stop),
        ),
        metadata={
            "head_dim": dims.head_dim,
            "kv_group": q_head_to_kv_group(dims, head),
        },
    )


def _kv_group_unit(dims: Qwen2Dimensions, layer: int, kv_head: int) -> PrunableUnit:
    start = kv_head * dims.head_dim
    stop = start + dims.head_dim
    prefix = layer_prefix(layer)
    return PrunableUnit(
        id=UnitId(kind=KV_HEAD_GROUP, layer=layer, index=kv_head),
        slices=(
            TensorSlice("{0}.self_attn.k_proj.weight".format(prefix), 0, start, stop),
            TensorSlice("{0}.self_attn.k_proj.bias".format(prefix), 0, start, stop),
            TensorSlice("{0}.self_attn.v_proj.weight".format(prefix), 0, start, stop),
            TensorSlice("{0}.self_attn.v_proj.bias".format(prefix), 0, start, stop),
        ),
        metadata={
            "head_dim": dims.head_dim,
            "query_heads": kv_group_query_heads(dims, kv_head),
        },
    )


def _ffn_channel_unit(layer: int, channel: int) -> PrunableUnit:
    prefix = layer_prefix(layer)
    return PrunableUnit(
        id=UnitId(kind=FFN_CHANNEL, layer=layer, index=channel),
        slices=(
            TensorSlice("{0}.mlp.gate_proj.weight".format(prefix), 0, channel, channel + 1),
            TensorSlice("{0}.mlp.up_proj.weight".format(prefix), 0, channel, channel + 1),
            TensorSlice("{0}.mlp.down_proj.weight".format(prefix), 1, channel, channel + 1),
        ),
        metadata={},
    )


def _as_dims(config_or_dims: Any) -> Qwen2Dimensions:
    if isinstance(config_or_dims, Qwen2Dimensions):
        return config_or_dims
    return qwen2_dimensions(config_or_dims)


def _config_value(config: Any, key: str, default: Optional[Any] = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _required_positive_int(config: Any, key: str) -> int:
    value = _config_value(config, key, None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("Missing positive integer config field: {0}".format(key))
    return value


def _validate_index(index: int, size: int, label: str) -> None:
    if not isinstance(index, int) or index < 0 or index >= size:
        raise ValueError("{0} index out of range: {1}".format(label, index))
