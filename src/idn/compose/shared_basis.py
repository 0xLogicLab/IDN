"""Decompose capability LoRAs into global shared bases and exact residuals."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import load_file, save_file


WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"


class SharedBasisError(RuntimeError):
    """Raised when shared-basis artifacts cannot be built or loaded."""


def build_shared_basis(
    *,
    adapter_paths: Mapping[str, Path],
    output_dir: Path,
    components: int = 4,
    output_dtype: str = "bfloat16",
    override: bool = False,
) -> Dict[str, Any]:
    """Build materialized PEFT basis and residual adapters.

    The decomposition is performed over the complete scaled delta operator of
    each adapter. Dense model-sized delta matrices are never constructed.
    """

    capabilities = sorted(str(name) for name in adapter_paths)
    if len(capabilities) < 2:
        raise SharedBasisError("at least two source adapters are required")
    if components <= 0 or components >= len(capabilities):
        raise SharedBasisError(
            "components must be between 1 and {0}".format(len(capabilities) - 1)
        )
    dtype = _parse_dtype(output_dtype)
    output_dir = Path(output_dir)
    resolved_output = output_dir.resolve()
    for capability, adapter_path in adapter_paths.items():
        resolved_adapter = Path(adapter_path).resolve()
        if resolved_output == resolved_adapter or resolved_output in resolved_adapter.parents:
            raise SharedBasisError(
                "output directory cannot contain source adapter {0}: {1}".format(
                    capability, adapter_path
                )
            )
    if output_dir.exists() and not override:
        raise SharedBasisError(
            "output directory already exists: {0}; pass --override to replace it".format(
                output_dir
            )
        )

    sources = [
        _load_source(capability, Path(adapter_paths[capability]))
        for capability in capabilities
    ]
    _align_source_modules(sources)
    _validate_sources(sources)
    print("Computing global LoRA Gram matrix", flush=True)
    gram = compute_adapter_gram(
        [source["state"] for source in sources],
        [float(source["scaling"]) for source in sources],
    )
    eigenvalues, eigenvectors = decompose_gram(gram, components)
    projection = eigenvectors @ eigenvectors.T
    residual_coefficients = np.eye(len(capabilities), dtype=np.float64) - projection

    temporary_parent = output_dir.parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".{0}.tmp-".format(output_dir.name), dir=temporary_parent)
    )
    try:
        basis_entries = []
        materialized_rank = sum(int(source["rank"]) for source in sources)
        output_config = _output_config(
            sources[0]["config"],
            materialized_rank,
            target_modules=sorted(
                {
                    str(target)
                    for source in sources
                    for target in source["config"].get("target_modules") or []
                }
            ),
        )
        for component in range(components):
            name = "basis_{0:02d}".format(component)
            relative_path = Path("bases") / name
            coefficients = eigenvectors[:, component]
            print("Materializing {0}".format(name), flush=True)
            _write_adapter(
                temporary_dir / relative_path,
                sources,
                coefficients,
                output_config,
                dtype,
            )
            basis_entries.append(
                {
                    "name": name,
                    "path": str(relative_path),
                    "source_coefficients": {
                        capability: float(coefficients[index])
                        for index, capability in enumerate(capabilities)
                    },
                }
            )

        residual_entries = {}
        for index, capability in enumerate(capabilities):
            name = "residual_{0}".format(capability)
            relative_path = Path("residuals") / capability
            coefficients = residual_coefficients[index]
            print("Materializing {0}".format(name), flush=True)
            _write_adapter(
                temporary_dir / relative_path,
                sources,
                coefficients,
                output_config,
                dtype,
            )
            residual_entries[capability] = {
                "name": name,
                "path": str(relative_path),
                "source_coefficients": {
                    source_capability: float(coefficients[source_index])
                    for source_index, source_capability in enumerate(capabilities)
                },
            }

        total_energy = float(np.trace(gram))
        retained_energy = float(eigenvalues[:components].sum())
        all_eigenvalues = np.linalg.eigvalsh(gram)[::-1]
        coordinates = {
            capability: [
                float(eigenvectors[index, component])
                for component in range(components)
            ]
            for index, capability in enumerate(capabilities)
        }
        residual_energy = {}
        reconstruction_error = {}
        for index, capability in enumerate(capabilities):
            residual_row = residual_coefficients[index]
            energy = float(residual_row @ gram @ residual_row)
            source_energy = float(gram[index, index])
            residual_energy[capability] = {
                "squared_norm": max(0.0, energy),
                "relative_squared_norm": (
                    max(0.0, energy) / source_energy if source_energy > 0.0 else 0.0
                ),
            }
            reconstructed = projection[index] + residual_row
            error_row = np.eye(len(capabilities), dtype=np.float64)[index] - reconstructed
            error_energy = float(error_row @ gram @ error_row)
            reconstruction_error[capability] = math.sqrt(max(0.0, error_energy))

        manifest = {
            "artifact_version": 1,
            "method": "global_delta_gram_eigendecomposition",
            "capabilities": capabilities,
            "components": int(components),
            "source_rank": int(sources[0]["rank"]),
            "materialized_rank": int(materialized_rank),
            "output_dtype": output_dtype,
            "coordinates": coordinates,
            "bases": basis_entries,
            "residuals": residual_entries,
            "decomposition": {
                "gram_matrix": gram.tolist(),
                "eigenvalues": [float(value) for value in all_eigenvalues],
                "retained_energy_ratio": (
                    retained_energy / total_energy if total_energy > 0.0 else 0.0
                ),
                "residual_energy": residual_energy,
                "algebraic_reconstruction_error": reconstruction_error,
            },
            "source_adapters": {
                source["capability"]: {
                    "path": str(source["path"]),
                    "rank": int(source["rank"]),
                    "alpha": float(source["alpha"]),
                    "scaling": float(source["scaling"]),
                    "target_modules": sorted(
                        str(target)
                        for target in source["config"].get("target_modules") or []
                    ),
                    "zero_filled_lora_modules": int(
                        len(_factor_keys(source["state"]))
                        - len(source["original_factor_keys"])
                    ),
                    "config_sha256": _sha256(source["path"] / CONFIG_NAME),
                    "weights_sha256": _sha256(source["path"] / WEIGHTS_NAME),
                }
                for source in sources
            },
            "lineage": {"created_at": datetime.now(timezone.utc).isoformat()},
        }
        _write_json(temporary_dir / "manifest.json", manifest)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(str(temporary_dir), str(output_dir))
        return manifest
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def compute_adapter_gram(
    states: Sequence[Mapping[str, torch.Tensor]], scalings: Sequence[float]
) -> np.ndarray:
    """Compute Frobenius inner products of scaled LoRA deltas."""

    if not states or len(states) != len(scalings):
        raise SharedBasisError("states and scalings must have the same non-zero length")
    a_keys = _factor_keys(states[0])
    for state in states[1:]:
        if _factor_keys(state) != a_keys:
            raise SharedBasisError("source adapters do not contain identical LoRA keys")

    count = len(states)
    gram = np.zeros((count, count), dtype=np.float64)
    with torch.no_grad():
        for key_index, a_key in enumerate(a_keys, start=1):
            b_key = _b_key(a_key)
            a_factors = [state[a_key].detach().to(torch.float32) for state in states]
            b_factors = [state[b_key].detach().to(torch.float32) for state in states]
            for left in range(count):
                for right in range(left, count):
                    a_cross = a_factors[left] @ a_factors[right].T
                    b_cross = b_factors[left].T @ b_factors[right]
                    value = float((a_cross * b_cross).sum().item())
                    value *= float(scalings[left]) * float(scalings[right])
                    gram[left, right] += value
                    if left != right:
                        gram[right, left] += value
            if key_index % 32 == 0 or key_index == len(a_keys):
                print(
                    "  Gram modules: {0}/{1}".format(key_index, len(a_keys)),
                    flush=True,
                )
    return (gram + gram.T) * 0.5


def decompose_gram(gram: np.ndarray, components: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return descending eigenvalues and deterministic leading eigenvectors."""

    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise SharedBasisError("Gram matrix must be square")
    if components <= 0 or components > matrix.shape[0]:
        raise SharedBasisError("invalid number of components: {0}".format(components))
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    tolerance = max(1.0, abs(float(values[0]))) * 1.0e-10
    if float(values[components - 1]) <= tolerance:
        raise SharedBasisError(
            "source adapters have fewer than {0} independent delta directions".format(
                components
            )
        )
    leading = vectors[:, :components].copy()
    for component in range(components):
        pivot = int(np.argmax(np.abs(leading[:, component])))
        if leading[pivot, component] < 0.0:
            leading[:, component] *= -1.0
    return np.maximum(values[:components], 0.0), leading


def load_shared_basis_manifest(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise SharedBasisError("shared-basis manifest does not exist: {0}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("artifact_version") != 1:
        raise SharedBasisError("unsupported shared-basis artifact version")
    capabilities = manifest.get("capabilities")
    components = manifest.get("components")
    bases = manifest.get("bases")
    residuals = manifest.get("residuals")
    coordinates = manifest.get("coordinates")
    if not isinstance(capabilities, list) or not capabilities:
        raise SharedBasisError("manifest has no capabilities")
    if not isinstance(components, int) or components <= 0:
        raise SharedBasisError("manifest has an invalid component count")
    if not isinstance(bases, list) or len(bases) != components:
        raise SharedBasisError("manifest basis count does not match components")
    if not isinstance(residuals, dict) or set(residuals) != set(capabilities):
        raise SharedBasisError("manifest residuals do not match capabilities")
    if not isinstance(coordinates, dict) or set(coordinates) != set(capabilities):
        raise SharedBasisError("manifest coordinates do not match capabilities")
    for capability in capabilities:
        values = coordinates[capability]
        if not isinstance(values, list) or len(values) != components:
            raise SharedBasisError(
                "invalid coordinates for capability: {0}".format(capability)
            )
    return manifest


def resolve_adapter_paths(
    manifest: Mapping[str, Any], manifest_path: Path
) -> Dict[str, Path]:
    """Resolve every materialized adapter path using stable runtime names."""

    root = Path(manifest_path).resolve().parent
    paths = {}
    for basis in manifest["bases"]:
        paths[str(basis["name"])] = _resolve_manifest_path(root, basis["path"])
    for capability in manifest["capabilities"]:
        residual = manifest["residuals"][capability]
        paths[str(residual["name"])] = _resolve_manifest_path(root, residual["path"])
    return paths


def compile_shared_scales(
    manifest: Mapping[str, Any], route_weights: Mapping[str, float]
) -> Dict[str, Any]:
    """Compile ROUTE weights into basis scales plus one top-1 residual scale."""

    capabilities = [str(value) for value in manifest["capabilities"]]
    if set(route_weights) != set(capabilities):
        raise SharedBasisError("routing weights do not match basis capabilities")
    weights = np.asarray(
        [float(route_weights[capability]) for capability in capabilities],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise SharedBasisError("routing weights must be finite and non-negative")
    if abs(float(weights.sum()) - 1.0) > 1.0e-5:
        raise SharedBasisError("routing weights must sum to one")
    coordinate_matrix = np.asarray(
        [manifest["coordinates"][capability] for capability in capabilities],
        dtype=np.float64,
    )
    shared_coordinates = weights @ coordinate_matrix
    top_index = int(np.argmax(weights))
    selected_capability = capabilities[top_index]
    adapter_scales = {
        str(basis["name"]): float(shared_coordinates[index])
        for index, basis in enumerate(manifest["bases"])
    }
    for capability in capabilities:
        residual_name = str(manifest["residuals"][capability]["name"])
        adapter_scales[residual_name] = (
            float(weights[top_index]) if capability == selected_capability else 0.0
        )
    active_adapters = [str(basis["name"]) for basis in manifest["bases"]]
    active_adapters.append(str(manifest["residuals"][selected_capability]["name"]))
    return {
        "coordinates": {
            str(basis["name"]): float(shared_coordinates[index])
            for index, basis in enumerate(manifest["bases"])
        },
        "selected_residual": selected_capability,
        "residual_scale": float(weights[top_index]),
        "active_adapters": active_adapters,
        "adapter_scales": adapter_scales,
    }


def _load_source(capability: str, path: Path) -> Dict[str, Any]:
    config_path = path / CONFIG_NAME
    weights_path = path / WEIGHTS_NAME
    if not config_path.exists() or not weights_path.exists():
        raise SharedBasisError("incomplete adapter for {0}: {1}".format(capability, path))
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("peft_type") != "LORA":
        raise SharedBasisError("adapter is not LoRA: {0}".format(path))
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise SharedBasisError("per-module rank or alpha patterns are not supported")
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    scaling = alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)
    state = load_file(str(weights_path), device="cpu")
    original_factor_keys = _factor_keys(state)
    return {
        "capability": capability,
        "path": path,
        "config": config,
        "state": state,
        "original_factor_keys": original_factor_keys,
        "rank": rank,
        "alpha": alpha,
        "scaling": scaling,
    }


def _validate_sources(sources: Sequence[Mapping[str, Any]]) -> None:
    first = sources[0]
    first_keys = _factor_keys(first["state"])
    for source in sources:
        if int(source["rank"]) != int(first["rank"]):
            raise SharedBasisError("source adapters must use the same rank")
        if _factor_keys(source["state"]) != first_keys:
            raise SharedBasisError("source adapters have different LoRA modules")
        for a_key in first_keys:
            b_key = _b_key(a_key)
            if source["state"][a_key].shape[0] != int(source["rank"]):
                raise SharedBasisError("unexpected A rank in {0}".format(a_key))
            if source["state"][b_key].shape[1] != int(source["rank"]):
                raise SharedBasisError("unexpected B rank in {0}".format(b_key))


def _align_source_modules(sources: Sequence[Dict[str, Any]]) -> None:
    """Represent missing adapter modules as exact zero LoRA factors."""

    all_keys = sorted(
        {
            key
            for source in sources
            for key in _factor_keys(source["state"])
        }
    )
    for a_key in all_keys:
        b_key = _b_key(a_key)
        present = [source for source in sources if a_key in source["state"]]
        reference_a = present[0]["state"][a_key]
        reference_b = present[0]["state"][b_key]
        in_features = int(reference_a.shape[1])
        out_features = int(reference_b.shape[0])
        for source in present:
            a_factor = source["state"][a_key]
            b_factor = source["state"][b_key]
            if int(a_factor.shape[1]) != in_features or int(b_factor.shape[0]) != out_features:
                raise SharedBasisError(
                    "source adapters have incompatible shapes for {0}".format(a_key)
                )
        for source in sources:
            if a_key in source["state"]:
                continue
            rank = int(source["rank"])
            dtype = next(iter(source["state"].values())).dtype
            source["state"][a_key] = torch.zeros(
                (rank, in_features), dtype=dtype
            )
            source["state"][b_key] = torch.zeros(
                (out_features, rank), dtype=dtype
            )


def _factor_keys(state: Mapping[str, torch.Tensor]) -> List[str]:
    a_keys = sorted(key for key in state if key.endswith(".lora_A.weight"))
    expected = set(a_keys) | {_b_key(key) for key in a_keys}
    if not a_keys or set(state) != expected:
        unsupported = sorted(set(state) - expected)
        raise SharedBasisError(
            "adapter must contain only paired LoRA A/B weights; unsupported keys: {0}".format(
                unsupported[:3]
            )
        )
    for key in a_keys:
        if _b_key(key) not in state:
            raise SharedBasisError("missing paired B factor for {0}".format(key))
    return a_keys


def _b_key(a_key: str) -> str:
    return a_key[: -len(".lora_A.weight")] + ".lora_B.weight"


def _write_adapter(
    path: Path,
    sources: Sequence[Mapping[str, Any]],
    coefficients: Sequence[float],
    config: Mapping[str, Any],
    dtype: torch.dtype,
) -> None:
    path.mkdir(parents=True, exist_ok=False)
    output_state = {}
    for a_key in _factor_keys(sources[0]["state"]):
        b_key = _b_key(a_key)
        output_state[a_key] = torch.cat(
            [source["state"][a_key] for source in sources], dim=0
        ).to(dtype=dtype).contiguous()
        scaled_b = [
            source["state"][b_key].to(torch.float32)
            * (float(coefficient) * float(source["scaling"]))
            for source, coefficient in zip(sources, coefficients)
        ]
        output_state[b_key] = torch.cat(scaled_b, dim=1).to(dtype=dtype).contiguous()
    _write_json(path / CONFIG_NAME, config)
    save_file(output_state, str(path / WEIGHTS_NAME), metadata={"format": "pt"})


def _output_config(
    source_config: Mapping[str, Any],
    rank: int,
    *,
    target_modules: Sequence[str],
) -> Dict[str, Any]:
    config = dict(source_config)
    config.update(
        {
            "alpha_pattern": {},
            "inference_mode": True,
            "init_lora_weights": True,
            "lora_alpha": int(rank),
            "lora_dropout": 0.0,
            "r": int(rank),
            "rank_pattern": {},
            "use_rslora": False,
        }
    )
    config["target_modules"] = list(target_modules)
    return config


def _parse_dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise SharedBasisError("unsupported output dtype: {0}".format(name)) from exc


def _resolve_manifest_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
