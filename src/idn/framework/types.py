"""Stable data contracts shared by IDN runtime components."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class AdaptationSpec:
    """Describe how a model package adapts its base model.

    ``strategy_id`` selects a registered loader strategy. ``artifact_path`` is
    a LoRA directory for ``lora`` or a shared-basis ``manifest.json`` for
    ``shared_basis``. For shared bases, ``coefficients`` may contain weights
    over capabilities (default: one-hot for the package capability), or direct
    basis coefficients when ``coefficient_space="basis"``.

    ``options`` is reserved for strategy-specific settings so new adaptation
    methods can be added without changing ``ModelPackage`` again.
    """

    strategy_id: str
    artifact_path: Optional[str] = None
    coefficients: Mapping[str, float] = field(default_factory=dict)
    coefficient_space: str = "capability"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("adaptation strategy_id cannot be empty")
        if self.artifact_path is not None and not self.artifact_path.strip():
            raise ValueError("adaptation artifact_path cannot be empty")
        if self.coefficient_space not in ("capability", "basis"):
            raise ValueError(
                "coefficient_space must be 'capability' or 'basis'"
            )
        coefficients = dict(self.coefficients)
        for name, value in coefficients.items():
            if not str(name).strip():
                raise ValueError("coefficient names cannot be empty")
            if not math.isfinite(float(value)):
                raise ValueError("coefficient values must be finite")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("message role cannot be empty")
        if not isinstance(self.content, str):
            raise TypeError("message content must be text")


@dataclass(frozen=True)
class InferenceRequest:
    """A normalized request; all messages participate in cache identity."""

    messages: Tuple[Message, ...]
    max_new_tokens: int = 128
    tenant_id: str = "default"

    def __post_init__(self) -> None:
        normalized = tuple(_as_message(message) for message in self.messages)
        if not normalized:
            raise ValueError("an inference request needs at least one message")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not self.tenant_id:
            raise ValueError("tenant_id cannot be empty")
        object.__setattr__(self, "messages", normalized)

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Union[Mapping[str, Any], Message]],
        *,
        max_new_tokens: int = 128,
        tenant_id: str = "default",
    ) -> "InferenceRequest":
        return cls(
            messages=tuple(_as_message(message) for message in messages),
            max_new_tokens=max_new_tokens,
            tenant_id=tenant_id,
        )

    @property
    def latest_user_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        raise ValueError("routing requires a user message")

    def as_dict_messages(self) -> Sequence[Dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]


@dataclass(frozen=True)
class ModelPackage:
    """One deployable capability/model/adapter combination."""

    package_id: str
    capability: Optional[str]
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    model_size_b: Optional[float] = None
    estimated_memory_gib: Optional[float] = None
    estimated_latency_ms: Optional[float] = None
    quality_lower_bound: Optional[float] = None
    certified: bool = False
    is_fallback: bool = False
    provider_id: str = "default"
    backend_model_id: Optional[str] = None
    is_remote: bool = False
    adaptation: Optional[AdaptationSpec] = None

    def __post_init__(self) -> None:
        if not self.package_id:
            raise ValueError("package_id cannot be empty")
        if not self.model_path and not self.backend_model_id:
            raise ValueError("model_path or backend_model_id must be provided")
        if self.model_path is not None and not self.model_path.strip():
            raise ValueError("model_path cannot be empty")
        if not self.provider_id.strip():
            raise ValueError("provider_id cannot be empty")
        if self.adapter_path and self.adaptation is not None:
            raise ValueError(
                "use either legacy adapter_path or adaptation, not both"
            )
        if self.capability is None and not self.is_fallback:
            raise ValueError("a package without a capability must be a fallback")
        for name in (
            "model_size_b",
            "estimated_memory_gib",
            "estimated_latency_ms",
            "quality_lower_bound",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("{0} must be finite when provided".format(name))
            if value is not None and name != "quality_lower_bound" and value < 0:
                raise ValueError("{0} cannot be negative".format(name))


@dataclass(frozen=True)
class ResourceBudget:
    """Optional placement limits used to adapt model size per request."""

    max_memory_gib: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_quality_lower_bound: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("max_memory_gib", "max_latency_ms", "min_quality_lower_bound"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("{0} must be finite when provided".format(name))
            if value is not None and value < 0:
                raise ValueError("{0} cannot be negative".format(name))


@dataclass(frozen=True)
class InferenceResponse:
    text: str
    capability: str
    package_id: str
    route_confidence: float
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    cache_hit: bool = False


def _as_message(value: Union[Mapping[str, Any], Message]) -> Message:
    if isinstance(value, Message):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("messages must contain Message objects or mappings")
    if "role" not in value or "content" not in value:
        raise ValueError("each message needs role and content")
    return Message(role=str(value["role"]), content=value["content"])
