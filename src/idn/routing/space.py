"""Fit and persist a semantic capability routing space."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class RoutingError(RuntimeError):
    """Raised when a routing space cannot be fitted, loaded, or saved."""


@dataclass(frozen=True)
class RoutingSpace:
    capabilities: Tuple[str, ...]
    embedding_mean: np.ndarray
    projection: np.ndarray
    centroids: np.ndarray
    precisions: np.ndarray
    temperature: float
    ood_threshold: float
    metadata: Mapping[str, Any]

    @property
    def embedding_size(self) -> int:
        return int(self.projection.shape[1])

    @property
    def dimensions(self) -> int:
        return int(self.projection.shape[0])

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.embedding_size:
            raise RoutingError(
                "expected embeddings with shape (n, {0}), got {1}".format(
                    self.embedding_size, tuple(values.shape)
                )
            )
        return (values - self.embedding_mean) @ self.projection.T

    def save(self, path: Path, override: bool = False) -> None:
        path = Path(path)
        if path.exists() and not override:
            raise RoutingError("routing space already exists: {0}".format(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata.update(
            {
                "artifact_version": 1,
                "representation": "base_model_mean_pooled_hidden_state",
                "embedding_size": self.embedding_size,
                "dimensions": self.dimensions,
                "temperature": self.temperature,
                "ood_threshold": self.ood_threshold,
            }
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".{0}.".format(path.name), suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    capabilities=np.asarray(self.capabilities, dtype=np.str_),
                    embedding_mean=np.asarray(self.embedding_mean, dtype=np.float64),
                    projection=np.asarray(self.projection, dtype=np.float64),
                    centroids=np.asarray(self.centroids, dtype=np.float64),
                    precisions=np.asarray(self.precisions, dtype=np.float64),
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path) -> "RoutingSpace":
        path = Path(path)
        if not path.exists():
            raise RoutingError("routing space does not exist: {0}".format(path))
        try:
            with np.load(str(path), allow_pickle=False) as artifact:
                capabilities = tuple(str(item) for item in artifact["capabilities"])
                embedding_mean = np.asarray(
                    artifact["embedding_mean"], dtype=np.float64
                )
                projection = np.asarray(artifact["projection"], dtype=np.float64)
                centroids = np.asarray(artifact["centroids"], dtype=np.float64)
                precisions = np.asarray(artifact["precisions"], dtype=np.float64)
                metadata = json.loads(str(artifact["metadata"].item()))
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise RoutingError("invalid routing space {0}: {1}".format(path, exc)) from exc
        _validate_shapes(
            capabilities, embedding_mean, projection, centroids, precisions
        )
        return cls(
            capabilities=capabilities,
            embedding_mean=embedding_mean,
            projection=projection,
            centroids=centroids,
            precisions=precisions,
            temperature=float(metadata["temperature"]),
            ood_threshold=float(metadata["ood_threshold"]),
            metadata=metadata,
        )


def fit_semantic_routing_space(
    embeddings: np.ndarray,
    labels: Sequence[str],
    *,
    components: int = 64,
    shrinkage: float = 0.1,
    temperature: float = 1.0,
    ood_percentile: float = 95.0,
    seed: int = 13,
    metadata: Optional[Mapping[str, Any]] = None,
) -> RoutingSpace:
    """Fit PCA and per-capability shrinkage Gaussian descriptors."""

    values = np.asarray(embeddings, dtype=np.float64)
    _validate_fit_inputs(
        embeddings=values,
        labels=labels,
        components=components,
        shrinkage=shrinkage,
        temperature=temperature,
        ood_percentile=ood_percentile,
    )
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise RoutingError("scikit-learn is required to fit routing PCA") from exc

    dimensions = min(components, values.shape[0] - 1, values.shape[1])
    reducer = PCA(
        n_components=dimensions,
        svd_solver="randomized" if dimensions < min(values.shape) else "full",
        random_state=seed,
    )
    projected = np.asarray(reducer.fit_transform(values), dtype=np.float64)
    capabilities = tuple(sorted(set(str(label) for label in labels)))
    label_array = np.asarray([str(label) for label in labels])
    centroids = []
    precisions = []
    counts: Dict[str, int] = {}
    for capability in capabilities:
        rows = projected[label_array == capability]
        if len(rows) < 2:
            raise RoutingError(
                "capability {0} needs at least two training samples".format(capability)
            )
        counts[capability] = int(len(rows))
        centroid = rows.mean(axis=0)
        covariance = np.atleast_2d(np.cov(rows, rowvar=False, ddof=1))
        scale = max(float(np.trace(covariance)) / dimensions, 1.0e-8)
        covariance = (
            (1.0 - shrinkage) * covariance
            + shrinkage * scale * np.eye(dimensions)
            + 1.0e-8 * np.eye(dimensions)
        )
        centroids.append(centroid)
        precisions.append(np.linalg.pinv(covariance, hermitian=True))

    centroids_array = np.asarray(centroids, dtype=np.float64)
    precisions_array = np.asarray(precisions, dtype=np.float64)
    distances = _minimum_distances(projected, centroids_array, precisions_array)
    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        {
            "seed": int(seed),
            "training_samples": int(len(values)),
            "training_samples_by_capability": counts,
            "shrinkage": float(shrinkage),
            "ood_percentile": float(ood_percentile),
            "pca_explained_variance_ratio": [
                float(value) for value in reducer.explained_variance_ratio_
            ],
        }
    )
    return RoutingSpace(
        capabilities=capabilities,
        embedding_mean=np.asarray(reducer.mean_, dtype=np.float64),
        projection=np.asarray(reducer.components_, dtype=np.float64),
        centroids=centroids_array,
        precisions=precisions_array,
        temperature=float(temperature),
        ood_threshold=float(np.percentile(distances, ood_percentile)),
        metadata=artifact_metadata,
    )


def fit_routing_from_configs(
    *,
    config_dir: Path,
    machine_path: str,
    model_path: str,
    output_path: Path,
    report_path: Path,
    dtype: str,
    device: str,
    local_files_only: bool,
    samples_per_capability: int = 500,
    eval_samples_per_capability: int = 200,
    components: int = 64,
    shrinkage: float = 0.1,
    temperature: float = 1.0,
    ood_percentile: float = 95.0,
    max_chars: int = 2000,
    max_length: int = 512,
    batch_size: int = 4,
    seed: int = 13,
    override: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Fit on train prompts and report health on configured eval prompts."""

    from idn.routing.router import SemanticEmbedder, SemanticRouter, evaluate_router

    train_texts, train_labels, train_counts = _collect_configured_texts(
        config_dir=config_dir,
        machine_path=machine_path,
        split_kind="train",
        samples_per_capability=samples_per_capability,
        seed=seed,
        max_chars=max_chars,
    )
    eval_texts, eval_labels, eval_counts = _collect_configured_texts(
        config_dir=config_dir,
        machine_path=machine_path,
        split_kind="eval",
        samples_per_capability=eval_samples_per_capability,
        seed=seed + 1,
        max_chars=max_chars,
    )
    _report(progress, "loading base model")
    embedder = SemanticEmbedder(
        model_path=model_path,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
        max_length=max_length,
        batch_size=batch_size,
    ).load()
    _report(progress, "embedding {0} training requests".format(len(train_texts)))
    train_embeddings = embedder.embed(train_texts, progress=progress)
    space = fit_semantic_routing_space(
        train_embeddings,
        train_labels,
        components=components,
        shrinkage=shrinkage,
        temperature=temperature,
        ood_percentile=ood_percentile,
        seed=seed,
        metadata={
            "config_dir": str(config_dir),
            "machine": str(machine_path),
            "model_path": str(model_path),
            "dtype": dtype,
            "max_length": int(max_length),
            "pooling": "attention_mask_mean",
            "hidden_layer": -1,
            "training_samples_by_capability": train_counts,
        },
    )
    space.save(output_path, override=override)

    _report(progress, "embedding {0} held-out requests".format(len(eval_texts)))
    eval_embeddings = embedder.embed(eval_texts, progress=progress)
    ood_texts = _unrelated_ood_probes(limit=min(200, len(eval_texts)))
    _report(progress, "embedding {0} OOD probes".format(len(ood_texts)))
    ood_embeddings = embedder.embed(ood_texts, progress=progress)
    router = SemanticRouter(space=space, embedder=embedder)
    health = evaluate_router(
        router,
        eval_embeddings,
        eval_labels,
        ood_embeddings=ood_embeddings,
    )
    report = {
        "artifact_version": 1,
        "representation": "base_model_mean_pooled_hidden_state",
        "model_path": str(model_path),
        "routing_space": str(output_path),
        "training_samples": len(train_texts),
        "training_samples_by_capability": train_counts,
        "eval_samples_by_capability": eval_counts,
        "embedding_size": space.embedding_size,
        "dimensions": space.dimensions,
        "temperature": space.temperature,
        "ood_threshold": space.ood_threshold,
        "health": health,
        "notes": {
            "ood_probe": "unrelated synthetic requests outside the six capabilities",
            "gate_accuracy": 0.9,
            "gate_passed": bool(
                health["accuracy"] is not None and health["accuracy"] >= 0.9
            ),
        },
    }
    _write_json_atomic(report_path, report, override=override)
    report["report_path"] = str(report_path)
    return report


def _collect_configured_texts(
    *,
    config_dir: Path,
    machine_path: str,
    split_kind: str,
    samples_per_capability: int,
    seed: int,
    max_chars: int,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    from idn.analysis.embedding_viz import (
        load_dataset_rows,
        normalize_request_text,
        request_text_from_row,
    )
    from idn.config import discover_capability_configs, load_resolved_config

    if samples_per_capability <= 0:
        raise RoutingError("samples_per_capability must be positive")
    if split_kind not in ("train", "eval"):
        raise RoutingError("split_kind must be train or eval")
    texts: List[str] = []
    labels: List[str] = []
    counts: Dict[str, int] = {}
    for config_path in discover_capability_configs(config_dir):
        resolved = load_resolved_config(config_path, machine_path)
        capability = resolved.capability
        split = (
            capability.train_split if split_kind == "train" else capability.eval_split
        )
        rows = load_dataset_rows(
            capability=capability,
            split=split,
            limit=samples_per_capability,
            seed=seed,
        )
        count = 0
        for row in rows:
            request = "{0}\n\n{1}".format(
                capability.prompt_template,
                request_text_from_row(capability.id, row),
            )
            text = normalize_request_text(request, max_chars=max_chars)
            if not text:
                continue
            texts.append(text)
            labels.append(capability.id)
            count += 1
        counts[capability.id] = count
    missing = [capability for capability, count in counts.items() if count < 2]
    if missing:
        raise RoutingError(
            "fewer than two {0} samples for: {1}".format(
                split_kind, ", ".join(sorted(missing))
            )
        )
    return texts, labels, counts


def _minimum_distances(
    projected: np.ndarray,
    centroids: np.ndarray,
    precisions: np.ndarray,
) -> np.ndarray:
    columns = []
    for centroid, precision in zip(centroids, precisions):
        differences = projected - centroid
        squared = np.einsum("ni,ij,nj->n", differences, precision, differences)
        columns.append(np.sqrt(np.maximum(squared, 0.0)))
    return np.min(np.stack(columns, axis=1), axis=1)


def _unrelated_ood_probes(limit: int) -> List[str]:
    templates = (
        "Translate this paragraph from French into Japanese.",
        "What is the weather forecast for Shanghai this weekend?",
        "Plan a five day travel itinerary with hotels and restaurants.",
        "Give general advice about choosing a mortgage interest rate.",
        "Compose a short poem about mountains at sunrise.",
        "Describe the objects visible in the attached photograph.",
        "Recommend a strength training routine for a beginner.",
        "Explain how photosynthesis works to a ten year old student.",
        "Find recent football scores and summarize the league standings.",
        "Draft a polite email asking to reschedule tomorrow's meeting.",
    )
    return [templates[index % len(templates)] for index in range(max(0, limit))]


def _write_json_atomic(path: Path, payload: Mapping[str, Any], override: bool) -> None:
    path = Path(path)
    if path.exists() and not override:
        raise RoutingError("routing report already exists: {0}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{0}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _validate_fit_inputs(**values: Any) -> None:
    embeddings = values["embeddings"]
    labels = values["labels"]
    if embeddings.ndim != 2 or len(embeddings) != len(labels) or len(labels) < 4:
        raise RoutingError("embeddings must be a 2D array matching at least four labels")
    if len(set(labels)) < 2:
        raise RoutingError("routing fit needs at least two capabilities")
    if int(values["components"]) <= 0:
        raise RoutingError("components must be positive")
    if not 0.0 <= float(values["shrinkage"]) <= 1.0:
        raise RoutingError("shrinkage must be in [0, 1]")
    if float(values["temperature"]) <= 0.0:
        raise RoutingError("temperature must be positive")
    if not 0.0 < float(values["ood_percentile"]) < 100.0:
        raise RoutingError("ood_percentile must be in (0, 100)")


def _validate_shapes(
    capabilities: Sequence[str],
    embedding_mean: np.ndarray,
    projection: np.ndarray,
    centroids: np.ndarray,
    precisions: np.ndarray,
) -> None:
    classes = len(capabilities)
    if not classes or projection.ndim != 2:
        raise RoutingError("routing space has invalid capabilities or projection")
    dimensions, embedding_size = projection.shape
    if embedding_mean.shape != (embedding_size,):
        raise RoutingError("routing space has an invalid embedding mean")
    if centroids.shape != (classes, dimensions):
        raise RoutingError("routing space has invalid centroids")
    if precisions.shape != (classes, dimensions, dimensions):
        raise RoutingError("routing space has invalid precision matrices")


def _report(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)
