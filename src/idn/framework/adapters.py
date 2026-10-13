"""Pluggable adaptation strategies for local model loading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from idn.framework.types import AdaptationSpec


class AdaptationError(RuntimeError):
    """Raised when an adaptation artifact cannot be resolved or attached."""


@dataclass(frozen=True)
class ResolvedAdaptation:
    """Backend-ready adapter paths, activation order, and per-adapter scales."""

    adapter_paths: Mapping[str, str]
    active_adapters: Tuple[str, ...]
    scales: Mapping[str, float]


class AdaptationStrategy(Protocol):
    """A strategy that attaches one adaptation to an already-loaded model."""

    strategy_id: str

    def apply(
        self,
        model: Any,
        spec: AdaptationSpec,
        *,
        capability: Optional[str],
        local_files_only: bool,
    ) -> Any:
        ...


class AdaptationStrategyRegistry:
    """Resolve strategy IDs to implementations; custom strategies are pluggable."""

    def __init__(self, strategies: Sequence[AdaptationStrategy] = ()) -> None:
        self._strategies: Dict[str, AdaptationStrategy] = {}
        for strategy in strategies:
            self.register(strategy)

    def register(self, strategy: AdaptationStrategy) -> None:
        strategy_id = getattr(strategy, "strategy_id", None)
        if not isinstance(strategy_id, str) or not strategy_id.strip():
            raise ValueError("adaptation strategy must have a non-empty strategy_id")
        if strategy_id in self._strategies:
            raise ValueError("adaptation strategy already registered: {0}".format(strategy_id))
        if not callable(getattr(strategy, "apply", None)):
            raise TypeError("adaptation strategy must implement apply(model, spec, ...)")
        self._strategies[strategy_id] = strategy

    def apply(
        self,
        model: Any,
        spec: AdaptationSpec,
        *,
        capability: Optional[str],
        local_files_only: bool,
    ) -> Any:
        strategy = self._strategies.get(spec.strategy_id)
        if strategy is None:
            raise AdaptationError(
                "no adaptation strategy registered for {0}; registered: {1}".format(
                    spec.strategy_id, ", ".join(sorted(self._strategies))
                )
            )
        return strategy.apply(
            model,
            spec,
            capability=capability,
            local_files_only=local_files_only,
        )

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._strategies))


class LoRAAdapterStrategy:
    """Attach one standard PEFT LoRA adapter."""

    strategy_id = "lora"

    def apply(
        self,
        model: Any,
        spec: AdaptationSpec,
        *,
        capability: Optional[str],
        local_files_only: bool,
    ) -> Any:
        del capability
        if not spec.artifact_path:
            raise AdaptationError("lora strategy requires artifact_path")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise AdaptationError("peft is required to load LoRA adapters") from exc
        return PeftModel.from_pretrained(
            model,
            spec.artifact_path,
            local_files_only=local_files_only,
        )


class SharedBasisAdapterStrategy:
    """Load materialized basis adapters and activate a capability combination."""

    strategy_id = "shared_basis"

    def resolve(
        self,
        spec: AdaptationSpec,
        *,
        capability: Optional[str],
    ) -> ResolvedAdaptation:
        if not spec.artifact_path:
            raise AdaptationError("shared_basis strategy requires a manifest path")
        try:
            from idn.compose.shared_basis import (
                SharedBasisError,
                compile_shared_scales,
                load_shared_basis_manifest,
                resolve_adapter_paths,
            )
        except ImportError as exc:
            raise AdaptationError(
                "shared-basis support needs the IDN composition dependencies"
            ) from exc
        try:
            manifest_path = Path(spec.artifact_path)
            manifest = load_shared_basis_manifest(manifest_path)
            path_map = resolve_adapter_paths(manifest, manifest_path)
        except (OSError, ValueError, SharedBasisError) as exc:
            raise AdaptationError("could not read shared-basis manifest: {0}".format(exc)) from exc

        capabilities = [str(value) for value in manifest["capabilities"]]
        bases = [str(item["name"]) for item in manifest["bases"]]
        residuals = {
            str(name): str(item["name"])
            for name, item in manifest["residuals"].items()
        }

        if spec.coefficient_space == "capability":
            weights = {name: 0.0 for name in capabilities}
            if spec.coefficients:
                unknown = set(spec.coefficients) - set(capabilities)
                if unknown:
                    raise AdaptationError(
                        "unknown shared-basis capability coefficient(s): {0}".format(
                            ", ".join(sorted(unknown))
                        )
                    )
                weights.update(
                    {str(name): float(value) for name, value in spec.coefficients.items()}
                )
            else:
                if capability is None or capability not in weights:
                    raise AdaptationError(
                        "package capability must match the shared-basis manifest "
                        "or explicit capability coefficients must be provided"
                    )
                weights[capability] = 1.0

            try:
                compiled = compile_shared_scales(manifest, weights)
            except SharedBasisError as exc:
                raise AdaptationError("invalid capability coefficients: {0}".format(exc)) from exc
            active = tuple(str(name) for name in compiled["active_adapters"])
            scales = {
                str(name): float(value)
                for name, value in compiled["adapter_scales"].items()
                if name in active
            }
        else:
            unknown = set(spec.coefficients) - set(bases)
            if unknown:
                raise AdaptationError(
                    "unknown shared-basis component coefficient(s): {0}".format(
                        ", ".join(sorted(unknown))
                    )
                )
            if not spec.coefficients:
                raise AdaptationError(
                    "coefficient_space='basis' requires explicit basis coefficients"
                )
            scales = {name: 0.0 for name in bases}
            scales.update(
                {str(name): float(value) for name, value in spec.coefficients.items()}
            )
            residual_capability = str(
                spec.options.get("residual_capability", capability or "")
            )
            if residual_capability not in residuals:
                raise AdaptationError(
                    "residual_capability must match the shared-basis manifest"
                )
            try:
                residual_scale = float(spec.options.get("residual_scale", 1.0))
            except (TypeError, ValueError) as exc:
                raise AdaptationError("residual_scale must be numeric") from exc
            if not math.isfinite(residual_scale):
                raise AdaptationError("residual_scale must be finite")
            residual_name = residuals[residual_capability]
            active = tuple(bases + [residual_name])
            scales[residual_name] = residual_scale

        return ResolvedAdaptation(
            adapter_paths={str(name): str(path) for name, path in path_map.items()},
            active_adapters=active,
            scales=scales,
        )

    def apply(
        self,
        model: Any,
        spec: AdaptationSpec,
        *,
        capability: Optional[str],
        local_files_only: bool,
    ) -> Any:
        resolved = self.resolve(spec, capability=capability)
        if not resolved.active_adapters:
            raise AdaptationError("shared-basis plan has no active adapters")
        try:
            from peft import PeftModel
            from peft.tuners.lora import LoraLayer
        except ImportError as exc:
            raise AdaptationError("peft with LoRA support is required for shared bases") from exc

        first = resolved.active_adapters[0]
        adapted = PeftModel.from_pretrained(
            model,
            resolved.adapter_paths[first],
            adapter_name=first,
            is_trainable=False,
            local_files_only=local_files_only,
        )
        for adapter_name in resolved.active_adapters[1:]:
            adapted.load_adapter(
                resolved.adapter_paths[adapter_name],
                adapter_name=adapter_name,
                is_trainable=False,
                local_files_only=local_files_only,
            )
        adapted.base_model.set_adapter(list(resolved.active_adapters))

        updated_layers = 0
        for module in adapted.modules():
            if not isinstance(module, LoraLayer):
                continue
            for adapter_name, scale in resolved.scales.items():
                module.set_scale(adapter_name, float(scale))
            updated_layers += 1
        if updated_layers == 0:
            raise AdaptationError("loaded shared-basis model contains no LoRA layers")
        adapted.eval()
        return adapted


def build_default_adaptation_registry() -> AdaptationStrategyRegistry:
    return AdaptationStrategyRegistry(
        [LoRAAdapterStrategy(), SharedBasisAdapterStrategy()]
    )
