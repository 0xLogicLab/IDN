"""Composable routing, model-scale selection, fallback, and cache runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from idn.framework.cache import ResponseCache, response_cache_key
from idn.framework.types import (
    InferenceRequest,
    InferenceResponse,
    ModelPackage,
    ResourceBudget,
)


class FrameworkError(RuntimeError):
    """Raised when IDN cannot safely route or serve a request."""


class Router(Protocol):
    def route(self, text: str) -> Any:
        ...


class ModelProvider(Protocol):
    def generate(self, package: ModelPackage, request: InferenceRequest) -> str:
        ...


@dataclass(frozen=True)
class RuntimePolicy:
    min_route_confidence: float = 0.0
    require_in_scope: bool = True
    require_certification: bool = True
    require_certified_fallback: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_route_confidence <= 1.0:
            raise ValueError("min_route_confidence must be between zero and one")


class ModelPackageRegistry:
    """Resolve certified capability packages against per-request budgets."""

    def __init__(self, packages: Iterable[ModelPackage]) -> None:
        self._packages: Dict[str, ModelPackage] = {}
        self._by_capability: Dict[str, List[ModelPackage]] = {}
        self._fallbacks: List[ModelPackage] = []
        for package in packages:
            if package.package_id in self._packages:
                raise ValueError("duplicate package_id: {0}".format(package.package_id))
            self._packages[package.package_id] = package
            if package.is_fallback:
                self._fallbacks.append(package)
            elif package.capability is not None:
                self._by_capability.setdefault(package.capability, []).append(package)

    def select(
        self,
        capability: str,
        budget: ResourceBudget,
        *,
        require_certified: bool = True,
    ) -> Optional[ModelPackage]:
        candidates = self._feasible(
            self._by_capability.get(capability, ()),
            budget,
            require_certified=require_certified,
        )
        return min(candidates, key=_resource_order) if candidates else None

    def select_fallback(
        self,
        budget: ResourceBudget,
        *,
        require_certified: bool = False,
    ) -> Optional[ModelPackage]:
        candidates = self._feasible(
            self._fallbacks,
            budget,
            require_certified=require_certified,
        )
        # Prefer a measured quality bound; unknown quality must not rank first.
        return min(candidates, key=_fallback_order) if candidates else None

    @staticmethod
    def _feasible(
        packages: Sequence[ModelPackage],
        budget: ResourceBudget,
        *,
        require_certified: bool,
    ) -> List[ModelPackage]:
        feasible = []
        for package in packages:
            if require_certified and not package.certified:
                continue
            if budget.min_quality_lower_bound is not None:
                score = package.quality_lower_bound
                if score is None or score < budget.min_quality_lower_bound:
                    continue
            if budget.max_memory_gib is not None and not package.is_remote:
                memory = package.estimated_memory_gib
                if memory is None or memory > budget.max_memory_gib:
                    continue
            if budget.max_latency_ms is not None:
                latency = package.estimated_latency_ms
                if latency is None or latency > budget.max_latency_ms:
                    continue
            feasible.append(package)
        return feasible


class IDNRuntime:
    """Library-level request path that composes IDN's existing components.

    The router can be the existing ``SemanticRouter``. The model provider can
    load local Transformers models, call a vLLM endpoint, or use another backend.
    """

    def __init__(
        self,
        *,
        router: Router,
        packages: ModelPackageRegistry,
        provider: ModelProvider,
        cache: Optional[ResponseCache] = None,
        policy: RuntimePolicy = RuntimePolicy(),
        deployment_version: str = "dev",
    ) -> None:
        if not deployment_version:
            raise ValueError("deployment_version cannot be empty")
        self.router = router
        self.packages = packages
        self.provider = provider
        self.cache = cache
        self.policy = policy
        self.deployment_version = deployment_version

    def infer(
        self,
        request: InferenceRequest,
        budget: ResourceBudget = ResourceBudget(),
    ) -> InferenceResponse:
        cache_key = response_cache_key(
            request, budget, deployment_version=self.deployment_version
        )
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return InferenceResponse(
                    text=cached.text,
                    capability=cached.capability,
                    package_id=cached.package_id,
                    route_confidence=cached.route_confidence,
                    used_fallback=cached.used_fallback,
                    fallback_reason=cached.fallback_reason,
                    cache_hit=True,
                )

        route = self.router.route(request.latest_user_text)
        capability = str(_field(route, "capability"))
        confidence = float(_field(route, "confidence", 1.0))
        in_scope = bool(_field(route, "in_scope", True))
        fallback_reason = None

        if confidence < self.policy.min_route_confidence:
            fallback_reason = "route_confidence_below_threshold"
        elif self.policy.require_in_scope and not in_scope:
            fallback_reason = "request_out_of_scope"

        package = None
        if fallback_reason is None:
            package = self.packages.select(
                capability,
                budget,
                require_certified=self.policy.require_certification,
            )
            if package is None:
                fallback_reason = "no_certified_package_within_budget"

        used_fallback = fallback_reason is not None
        if used_fallback:
            package = self.packages.select_fallback(
                budget,
                require_certified=self.policy.require_certified_fallback,
            )
            if package is None:
                raise FrameworkError(
                    "request needs fallback ({0}), but no fallback package is available".format(
                        fallback_reason
                    )
                )

        text = self.provider.generate(package, request)
        response = InferenceResponse(
            text=text,
            capability=capability,
            package_id=package.package_id,
            route_confidence=confidence,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
        )
        if self.cache is not None:
            self.cache.put(cache_key, response)
        return response


def _field(value: Any, name: str, default: Any = None) -> Any:
    if not hasattr(value, name):
        if default is not None:
            return default
        raise FrameworkError("router result is missing {0}".format(name))
    return getattr(value, name)


def _known(value: Optional[float]) -> float:
    return float(value) if value is not None else float("inf")


def _resource_order(package: ModelPackage) -> tuple[float, float, float, float]:
    memory = _known(package.estimated_memory_gib)
    if package.estimated_memory_gib is None:
        memory = _known(package.model_size_b)
    latency = _known(package.estimated_latency_ms)
    quality = package.quality_lower_bound
    quality_order = -float(quality) if quality is not None else float("inf")
    # Prefer the smallest package meeting the quality/resource constraints.
    return (memory, latency, quality_order, package.model_size_b or float("inf"))


def _fallback_order(package: ModelPackage) -> tuple[float, float, float]:
    quality = package.quality_lower_bound
    # Unknown quality sorts after every measured quality bound.
    quality_order = -float(quality) if quality is not None else float("inf")
    return (
        quality_order,
        _known(package.estimated_latency_ms),
        _known(package.estimated_memory_gib),
    )
