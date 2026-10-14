"""Extensible task-metric registration and dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


MetricFunction = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class MetricValue:
    value: Optional[float]
    count: int
    status: str = "ok"
    higher_is_better: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("metric count cannot be negative")
        if self.status not in ("ok", "not_computed", "not_applicable"):
            raise ValueError("invalid metric status: {0}".format(self.status))
        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError("metric values must be finite")
        if self.status == "ok" and self.value is None:
            raise ValueError("an ok metric result needs a value")


@dataclass(frozen=True)
class MetricSpec:
    """A configured metric calculator selected by a capability."""

    metric_id: str
    options: Mapping[str, Any] = field(default_factory=dict)
    alias: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.metric_id.strip():
            raise ValueError("metric_id cannot be empty")
        object.__setattr__(self, "options", dict(self.options))
        if self.alias is not None and not self.alias.strip():
            raise ValueError("metric alias cannot be empty")


class MetricRegistry:
    """Map stable metric IDs to developer-supplied calculator functions."""

    def __init__(self) -> None:
        self._functions: Dict[str, MetricFunction] = {}

    def register(
        self,
        metric_id: str,
        function: Optional[MetricFunction] = None,
    ):
        if function is None:
            def decorator(candidate: MetricFunction) -> MetricFunction:
                return self.register(metric_id, candidate)

            return decorator
        metric_id = str(metric_id).strip()
        if not metric_id:
            raise ValueError("metric_id cannot be empty")
        if not callable(function):
            raise TypeError("metric calculator must be callable")
        if metric_id in self._functions:
            raise ValueError("metric already registered: {0}".format(metric_id))
        self._functions[metric_id] = function
        return function

    def get(self, metric_id: str) -> MetricFunction:
        try:
            return self._functions[metric_id]
        except KeyError as exc:
            raise KeyError(
                "unknown metric {0}; registered: {1}".format(
                    metric_id, ", ".join(self.names())
                )
            ) from exc

    def names(self) -> Sequence[str]:
        return tuple(sorted(self._functions))

    def load_modules(self, module_names: Sequence[str]) -> None:
        """Import modules whose decorators register task-specific calculators."""

        for module_name in module_names:
            if not str(module_name).strip():
                raise ValueError("metric module names cannot be empty")
            importlib.import_module(str(module_name))

    def compute(
        self,
        spec: MetricSpec,
        *,
        predictions: Sequence[Any],
        references: Sequence[Any],
        examples: Sequence[Any],
    ) -> Dict[str, Dict[str, Any]]:
        function = self.get(spec.metric_id)
        values = function(
            predictions=predictions,
            references=references,
            examples=examples,
            options=dict(spec.options),
        )
        if not isinstance(values, Mapping):
            raise TypeError("metric {0} must return a mapping".format(spec.metric_id))
        result: Dict[str, Dict[str, Any]] = {}
        use_alias_prefix = spec.alias is not None and len(values) > 1
        for name, value in values.items():
            if not str(name).strip():
                raise ValueError("metric result names cannot be empty")
            result_name = str(name)
            if spec.alias is not None:
                result_name = (
                    "{0}.{1}".format(spec.alias, result_name)
                    if use_alias_prefix
                    else spec.alias
                )
            if result_name in result:
                raise ValueError("duplicate metric result name: {0}".format(result_name))
            if isinstance(value, MetricValue):
                metric_value = value
            else:
                numeric = float(value)
                metric_value = MetricValue(value=numeric, count=len(predictions))
            result[result_name] = {
                "value": metric_value.value,
                "count": metric_value.count,
                "status": metric_value.status,
                "higher_is_better": metric_value.higher_is_better,
                "metric_id": spec.metric_id,
                "options": dict(spec.options),
            }
        return result


DEFAULT_METRICS = MetricRegistry()


def register_metric(metric_id: str, *, registry: MetricRegistry = DEFAULT_METRICS):
    """Register a calculator with a stable, explicit metric ID."""

    def decorator(function: MetricFunction) -> MetricFunction:
        return registry.register(metric_id, function)

    return decorator


def metric_specs_from_config(config: Any) -> Sequence[MetricSpec]:
    """Normalize new and legacy capability metric configuration."""

    configured = getattr(config, "metrics", None)
    if configured:
        return tuple(
            item
            if isinstance(item, MetricSpec)
            else MetricSpec(
                metric_id=str(item.get("id", item.get("metric_id", "")))
                if isinstance(item, Mapping)
                else str(item),
                options=item.get("options", {}) if isinstance(item, Mapping) else {},
                alias=item.get("alias") if isinstance(item, Mapping) else None,
            )
            for item in configured
        )
    metric_name = getattr(config, "metric_name", None)
    if not metric_name:
        return ()
    secondary = getattr(config, "secondary_metrics", ()) or ()
    return tuple([MetricSpec(str(metric_name))] + [MetricSpec(str(item)) for item in secondary])
