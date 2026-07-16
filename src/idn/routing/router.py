"""Semantic embedding inference and capability routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from idn.models.loader import ChatModel, ModelLoadError
from idn.routing.space import RoutingError, RoutingSpace


@dataclass(frozen=True)
class RouteDecision:
    capability: str
    weights: Mapping[str, float]
    distances: Mapping[str, float]
    confidence: float
    entropy: float
    in_scope: bool
    ood_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "weights": dict(self.weights),
            "distances": dict(self.distances),
            "confidence": self.confidence,
            "entropy": self.entropy,
            "in_scope": self.in_scope,
            "ood_threshold": self.ood_threshold,
        }


@dataclass
class SemanticEmbedder:
    model_path: str
    dtype: str = "bfloat16"
    device: str = "cuda"
    local_files_only: bool = True
    max_length: int = 512
    batch_size: int = 4

    def __post_init__(self) -> None:
        self.chat_model: Optional[ChatModel] = None

    def load(self) -> "SemanticEmbedder":
        if self.batch_size <= 0:
            raise RoutingError("embedding batch_size must be positive")
        if self.max_length <= 0:
            raise RoutingError("embedding max_length must be positive")
        try:
            self.chat_model = ChatModel(
                model_path=self.model_path,
                dtype=self.dtype,
                device=self.device,
                local_files_only=self.local_files_only,
            ).load()
        except ModelLoadError as exc:
            raise RoutingError("could not load routing model: {0}".format(exc)) from exc
        return self

    def embed(
        self,
        texts: Sequence[str],
        progress: Optional[Callable[[str], None]] = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        if self.chat_model is None:
            self.load()
        if self.chat_model is None:
            raise RoutingError("semantic embedder is not loaded")
        try:
            import torch
        except ImportError as exc:
            raise RoutingError("torch is required for semantic routing") from exc

        model = self.chat_model.model
        tokenizer = self.chat_model.tokenizer
        if model is None or tokenizer is None:
            raise RoutingError("routing model or tokenizer is unavailable")
        backbone = getattr(model, "model", None)
        if backbone is None:
            raise RoutingError("routing model does not expose its transformer backbone")
        input_device = _model_input_device(model)
        batches = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for batch_index, offset in enumerate(
            range(0, len(texts), self.batch_size), start=1
        ):
            batch_texts = texts[offset : offset + self.batch_size]
            prompts = [
                self.chat_model.render_prompt([{"role": "user", "content": text}])
                for text in batch_texts
            ]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = backbone(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RoutingError("routing backbone did not return last_hidden_state")
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones(
                    hidden.shape[:2], device=hidden.device, dtype=torch.long
                )
            mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            batches.append(pooled.float().cpu().numpy())
            if progress is not None and (
                batch_index == total_batches or batch_index % 10 == 0
            ):
                progress(
                    "embedded batch {0}/{1}".format(batch_index, total_batches)
                )
        return np.asarray(np.concatenate(batches, axis=0), dtype=np.float64)


class SemanticRouter:
    def __init__(
        self, space: RoutingSpace, embedder: Optional[SemanticEmbedder] = None
    ) -> None:
        self.space = space
        self.embedder = embedder

    def route(self, text: str) -> RouteDecision:
        if self.embedder is None:
            raise RoutingError("text routing requires a semantic embedder")
        return self.route_embeddings(self.embedder.embed([text]))[0]

    def route_many(self, texts: Sequence[str]) -> List[RouteDecision]:
        if not texts:
            return []
        if self.embedder is None:
            raise RoutingError("text routing requires a semantic embedder")
        return self.route_embeddings(self.embedder.embed(texts))

    def route_embeddings(self, embeddings: np.ndarray) -> List[RouteDecision]:
        values = np.asarray(embeddings)
        if values.size == 0:
            return []
        projected = self.space.project(values)
        distance_columns = []
        for centroid, precision in zip(self.space.centroids, self.space.precisions):
            differences = projected - centroid
            squared = np.einsum("ni,ij,nj->n", differences, precision, differences)
            distance_columns.append(np.sqrt(np.maximum(squared, 0.0)))
        distances = np.stack(distance_columns, axis=1)
        logits = -distances / self.space.temperature
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)

        decisions = []
        for row_distances, row_weights in zip(distances, weights):
            selected = int(np.argmax(row_weights))
            entropy = -sum(
                float(weight) * math.log(max(float(weight), 1.0e-12))
                for weight in row_weights
            )
            decisions.append(
                RouteDecision(
                    capability=self.space.capabilities[selected],
                    weights={
                        capability: float(row_weights[index])
                        for index, capability in enumerate(self.space.capabilities)
                    },
                    distances={
                        capability: float(row_distances[index])
                        for index, capability in enumerate(self.space.capabilities)
                    },
                    confidence=float(row_weights[selected]),
                    entropy=float(entropy),
                    in_scope=float(row_distances.min()) <= self.space.ood_threshold,
                    ood_threshold=self.space.ood_threshold,
                )
            )
        return decisions


def evaluate_router(
    router: SemanticRouter,
    embeddings: np.ndarray,
    labels: Sequence[str],
    ood_embeddings: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and labels must have the same length")
    decisions = router.route_embeddings(embeddings)
    capabilities = router.space.capabilities
    confusion = {
        expected: {predicted: 0 for predicted in capabilities}
        for expected in capabilities
    }
    correct = 0
    covered = 0
    covered_correct = 0
    for label, decision in zip(labels, decisions):
        if label not in confusion:
            raise ValueError("unknown evaluation capability: {0}".format(label))
        confusion[label][decision.capability] += 1
        is_correct = decision.capability == label
        correct += int(is_correct)
        covered += int(decision.in_scope)
        covered_correct += int(decision.in_scope and is_correct)
    per_capability = {}
    for capability in capabilities:
        total = sum(confusion[capability].values())
        per_capability[capability] = {
            "samples": total,
            "accuracy": confusion[capability][capability] / total if total else None,
        }
    ood_values = (
        np.asarray(ood_embeddings)
        if ood_embeddings is not None
        else np.empty((0, router.space.embedding_size))
    )
    ood_decisions = router.route_embeddings(ood_values)
    rejected = sum(not decision.in_scope for decision in ood_decisions)
    return {
        "samples": len(labels),
        "accuracy": correct / len(labels) if labels else None,
        "coverage": covered / len(labels) if labels else None,
        "selective_accuracy": covered_correct / covered if covered else None,
        "per_capability": per_capability,
        "confusion_matrix": confusion,
        "ood_samples": len(ood_decisions),
        "ood_rejection_rate": rejected / len(ood_decisions) if ood_decisions else None,
    }


def _model_input_device(model: Any) -> Any:
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is not None:
        return weight.device
    return next(model.parameters()).device
