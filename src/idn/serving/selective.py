"""Calibration and evaluation helpers for selective IDN fallback."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class GatePolicy:
    """A query-only gate over semantic routing diagnostics."""

    name: str
    max_distance: float = math.inf
    min_confidence: float = 0.0
    min_margin: float = 0.0
    allowed_capabilities: Tuple[str, ...] = ()
    max_distance_by_capability: Tuple[Tuple[str, float], ...] = ()

    def accepts(self, example: Mapping[str, Any]) -> bool:
        predicted = str(example["predicted_capability"])
        if self.allowed_capabilities and predicted not in self.allowed_capabilities:
            return False
        capability_thresholds = dict(self.max_distance_by_capability)
        max_distance = capability_thresholds.get(predicted, self.max_distance)
        return bool(
            float(example["min_distance"]) <= max_distance
            and float(example["confidence"]) >= self.min_confidence
            and float(example["distance_margin"]) >= self.min_margin
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["allowed_capabilities"] = list(self.allowed_capabilities)
        payload["max_distance_by_capability"] = {
            capability: threshold
            for capability, threshold in self.max_distance_by_capability
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GatePolicy":
        per_capability = payload.get("max_distance_by_capability", {})
        if isinstance(per_capability, Mapping):
            thresholds = tuple(
                (str(capability), float(threshold))
                for capability, threshold in sorted(per_capability.items())
            )
        else:
            thresholds = tuple(
                (str(capability), float(threshold))
                for capability, threshold in per_capability
            )
        return cls(
            name=str(payload["name"]),
            max_distance=float(payload.get("max_distance", math.inf)),
            min_confidence=float(payload.get("min_confidence", 0.0)),
            min_margin=float(payload.get("min_margin", 0.0)),
            allowed_capabilities=tuple(payload.get("allowed_capabilities", ())),
            max_distance_by_capability=thresholds,
        )


def route_features(record: Mapping[str, Any]) -> Dict[str, Any]:
    routing = record.get("routing")
    if not isinstance(routing, Mapping):
        raise ValueError("student prediction is missing routing diagnostics")
    distances = routing.get("distances")
    if not isinstance(distances, Mapping) or len(distances) < 2:
        raise ValueError("routing diagnostics need at least two distances")
    ordered = sorted(float(value) for value in distances.values())
    return {
        "predicted_capability": str(routing["capability"]),
        "confidence": float(routing["confidence"]),
        "entropy": float(routing["entropy"]),
        "min_distance": ordered[0],
        "distance_margin": ordered[1] - ordered[0],
        "in_scope": bool(routing.get("in_scope", False)),
    }


def stratified_calibration_indices(
    examples: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Split by true capability while remaining deterministic across input order."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration fraction must be between zero and one")
    by_capability: Dict[str, List[int]] = {}
    for position, example in enumerate(examples):
        by_capability.setdefault(str(example["capability"]), []).append(position)

    calibration = []
    test = []
    for capability, positions in sorted(by_capability.items()):
        capability_seed = int.from_bytes(
            hashlib.sha256(capability.encode("utf-8")).digest()[:4], "big"
        )
        shuffled = list(positions)
        random.Random(seed + capability_seed).shuffle(shuffled)
        count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * fraction))))
        calibration.extend(shuffled[:count])
        test.extend(shuffled[count:])
    return sorted(calibration), sorted(test)


def evaluate_policy(
    examples: Sequence[Mapping[str, Any]],
    policy: GatePolicy,
    indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    selected = list(indices) if indices is not None else list(range(len(examples)))
    if not selected:
        raise ValueError("policy evaluation needs at least one example")

    per_capability: Dict[str, Dict[str, Any]] = {}
    accepted = 0
    accepted_route_correct = 0
    decisions = []
    for position in selected:
        example = examples[position]
        capability = str(example["capability"])
        use_local = policy.accepts(example)
        local_score = float(example["local_score"])
        teacher_score = float(example["teacher_score"])
        hybrid_score = local_score if use_local else teacher_score
        bucket = per_capability.setdefault(
            capability,
            {
                "samples": 0,
                "accepted": 0,
                "local_score_sum": 0.0,
                "teacher_score_sum": 0.0,
                "hybrid_score_sum": 0.0,
            },
        )
        bucket["samples"] += 1
        bucket["accepted"] += int(use_local)
        bucket["local_score_sum"] += local_score
        bucket["teacher_score_sum"] += teacher_score
        bucket["hybrid_score_sum"] += hybrid_score
        accepted += int(use_local)
        accepted_route_correct += int(
            use_local
            and str(example["predicted_capability"]) == capability
        )
        decisions.append(
            {
                "mixed_index": int(example["mixed_index"]),
                "capability": capability,
                "predicted_capability": str(example["predicted_capability"]),
                "target": "local" if use_local else "teacher",
                "local_score": local_score,
                "teacher_score": teacher_score,
                "hybrid_score": hybrid_score,
                "min_distance": float(example["min_distance"]),
                "confidence": float(example["confidence"]),
                "distance_margin": float(example["distance_margin"]),
            }
        )

    local_scores = []
    teacher_scores = []
    hybrid_scores = []
    normalized = {}
    for capability, bucket in sorted(per_capability.items()):
        count = int(bucket["samples"])
        values = {
            "samples": count,
            "accepted": int(bucket["accepted"]),
            "coverage": float(bucket["accepted"]) / count,
            "local_score": float(bucket["local_score_sum"]) / count,
            "teacher_score": float(bucket["teacher_score_sum"]) / count,
            "hybrid_score": float(bucket["hybrid_score_sum"]) / count,
        }
        normalized[capability] = values
        local_scores.append(values["local_score"])
        teacher_scores.append(values["teacher_score"])
        hybrid_scores.append(values["hybrid_score"])

    total = len(selected)
    return {
        "samples": total,
        "accepted": accepted,
        "fallback": total - accepted,
        "coverage": accepted / total,
        "fallback_ratio": 1.0 - accepted / total,
        "accepted_route_accuracy": (
            accepted_route_correct / accepted if accepted else None
        ),
        "macro_local_score": sum(local_scores) / len(local_scores),
        "macro_teacher_score": sum(teacher_scores) / len(teacher_scores),
        "macro_hybrid_score": sum(hybrid_scores) / len(hybrid_scores),
        "macro_gap_to_teacher": (
            sum(hybrid_scores) / len(hybrid_scores)
            - sum(teacher_scores) / len(teacher_scores)
        ),
        "per_capability": normalized,
        "decisions": decisions,
    }


def calibrate_policy(
    examples: Sequence[Mapping[str, Any]],
    calibration_indices: Sequence[int],
    *,
    quality_tolerance: float,
    grid_size: int = 21,
    confidence_z: float = 1.645,
    eligible_capabilities: Optional[Sequence[str]] = None,
) -> Tuple[GatePolicy, List[Dict[str, Any]]]:
    """Maximize coverage subject to a calibration-set teacher quality constraint."""

    if quality_tolerance < 0.0:
        raise ValueError("quality tolerance must be non-negative")
    if grid_size < 3:
        raise ValueError("grid size must be at least three")
    if confidence_z < 0.0:
        raise ValueError("confidence z-value must be non-negative")
    calibration_examples = [examples[index] for index in calibration_indices]
    if not calibration_examples:
        raise ValueError("calibration set is empty")

    distances = [float(example["min_distance"]) for example in calibration_examples]
    confidences = [float(example["confidence"]) for example in calibration_examples]
    margins = [float(example["distance_margin"]) for example in calibration_examples]
    distance_grid = _quantile_grid(distances, grid_size)
    accept_all_distance = max(distances) + max(1.0, abs(max(distances))) * 1.0e-6
    distance_grid.append(accept_all_distance)
    confidence_grid = _quantile_grid(confidences, max(5, grid_size // 2))
    margin_grid = _quantile_grid(margins, max(5, grid_size // 2))

    capabilities = tuple(
        sorted({str(example["predicted_capability"]) for example in calibration_examples})
    )
    eligible = (
        tuple(sorted(str(capability) for capability in eligible_capabilities))
        if eligible_capabilities is not None
        else capabilities
    )
    unknown = sorted(set(eligible) - set(capabilities))
    if unknown:
        raise ValueError("unknown eligible capabilities: {0}".format(unknown))
    positive_capabilities = []
    for capability in capabilities:
        group = [
            example
            for example in calibration_examples
            if str(example["predicted_capability"]) == capability
        ]
        advantage = sum(
            float(example["local_score"]) - float(example["teacher_score"])
            for example in group
        ) / len(group)
        if advantage >= 0.0:
            positive_capabilities.append(capability)
    allowlists = [eligible]
    positive_eligible = tuple(
        capability for capability in positive_capabilities if capability in eligible
    )
    if positive_eligible != eligible:
        allowlists.append(positive_eligible)

    reject_all_distance = min(distances) - max(1.0, abs(min(distances))) * 1.0e-6
    policies = [GatePolicy(name="fallback_all", max_distance=reject_all_distance)]
    for allowed in allowlists:
        suffix = "all_caps" if not allowed else "positive_caps"
        for distance in distance_grid:
            policies.append(
                GatePolicy(
                    name="distance_{0}".format(suffix),
                    max_distance=distance,
                    allowed_capabilities=allowed,
                )
            )
        for confidence in confidence_grid:
            policies.append(
                GatePolicy(
                    name="confidence_{0}".format(suffix),
                    max_distance=accept_all_distance,
                    min_confidence=confidence,
                    allowed_capabilities=allowed,
                )
            )
        for distance in distance_grid:
            for confidence in confidence_grid:
                policies.append(
                    GatePolicy(
                        name="distance_confidence_{0}".format(suffix),
                        max_distance=distance,
                        min_confidence=confidence,
                        allowed_capabilities=allowed,
                    )
                )
        for distance in distance_grid:
            for margin in margin_grid:
                policies.append(
                    GatePolicy(
                        name="distance_margin_{0}".format(suffix),
                        max_distance=distance,
                        min_margin=margin,
                        allowed_capabilities=allowed,
                    )
                )

    unique: Dict[Tuple[Any, ...], GatePolicy] = {}
    for policy in policies:
        key = (
            policy.max_distance,
            policy.min_confidence,
            policy.min_margin,
            policy.allowed_capabilities,
            policy.max_distance_by_capability,
        )
        unique[key] = policy

    candidates = []
    for policy in unique.values():
        result = evaluate_policy(examples, policy, calibration_indices)
        candidates.append(
            {
                "policy": policy.to_dict(),
                "coverage": result["coverage"],
                "macro_hybrid_score": result["macro_hybrid_score"],
                "macro_teacher_score": result["macro_teacher_score"],
                "macro_gap_to_teacher": result["macro_gap_to_teacher"],
                "accepted_route_accuracy": result["accepted_route_accuracy"],
            }
        )

    candidates.extend(
        _capability_distance_candidates(
            calibration_examples,
            teacher_macro=float(candidates[0]["macro_teacher_score"]),
            eligible_capabilities=eligible,
        )
    )
    for candidate in candidates:
        candidate_policy = GatePolicy.from_dict(candidate["policy"])
        standard_error = _macro_gap_standard_error(
            calibration_examples, candidate_policy
        )
        candidate["macro_gap_standard_error"] = standard_error
        candidate["quality_lower_bound"] = (
            float(candidate["macro_gap_to_teacher"])
            - confidence_z * standard_error
        )

    feasible = [
        row
        for row in candidates
        if row["quality_lower_bound"] >= -quality_tolerance
    ]
    if not feasible:
        raise RuntimeError("no fallback policy satisfies the quality constraint")
    best = max(
        feasible,
        key=lambda row: (
            float(row["coverage"]),
            float(row["macro_hybrid_score"]),
            float(row["accepted_route_accuracy"] or 0.0),
        ),
    )
    policy = GatePolicy.from_dict(best["policy"])
    return policy, sorted(candidates, key=lambda row: float(row["coverage"]))


def _macro_gap_standard_error(
    examples: Sequence[Mapping[str, Any]], policy: GatePolicy
) -> float:
    groups: Dict[str, List[float]] = {}
    for example in examples:
        difference = (
            float(example["local_score"]) - float(example["teacher_score"])
            if policy.accepts(example)
            else 0.0
        )
        groups.setdefault(str(example["capability"]), []).append(difference)
    variance = 0.0
    for values in groups.values():
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        sample_variance = sum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        variance += sample_variance / len(values)
    return math.sqrt(variance) / len(groups)


def _capability_distance_candidates(
    calibration_examples: Sequence[Mapping[str, Any]],
    *,
    teacher_macro: float,
    eligible_capabilities: Sequence[str],
) -> List[Dict[str, Any]]:
    """Exact DP frontier for one distance threshold per predicted capability."""

    true_counts: Dict[str, int] = {}
    predicted_groups: Dict[str, List[Mapping[str, Any]]] = {}
    for example in calibration_examples:
        true_counts[str(example["capability"])] = (
            true_counts.get(str(example["capability"]), 0) + 1
        )
        predicted_groups.setdefault(str(example["predicted_capability"]), []).append(
            example
        )
    capability_count = len(true_counts)

    options: Dict[str, List[Tuple[int, float, float]]] = {}
    for capability, group in sorted(predicted_groups.items()):
        ordered = sorted(group, key=lambda example: float(example["min_distance"]))
        minimum = float(ordered[0]["min_distance"])
        capability_options = [
            (0, minimum - max(1.0, abs(minimum)) * 1.0e-6, 0.0)
        ]
        if capability not in eligible_capabilities:
            options[capability] = capability_options
            continue
        cumulative_gap = 0.0
        for position, example in enumerate(ordered, start=1):
            truth = str(example["capability"])
            cumulative_gap += (
                float(example["local_score"]) - float(example["teacher_score"])
            ) / (capability_count * true_counts[truth])
            next_distance = (
                float(ordered[position]["min_distance"])
                if position < len(ordered)
                else None
            )
            distance = float(example["min_distance"])
            if next_distance is None or next_distance > distance:
                capability_options.append((position, distance, cumulative_gap))
        options[capability] = capability_options

    # accepted_count -> (best macro gap, thresholds)
    states: Dict[int, Tuple[float, Dict[str, float]]] = {0: (0.0, {})}
    for capability, capability_options in sorted(options.items()):
        next_states: Dict[int, Tuple[float, Dict[str, float]]] = {}
        for previous_count, (previous_gap, previous_thresholds) in states.items():
            for count, threshold, gap in capability_options:
                total_count = previous_count + count
                total_gap = previous_gap + gap
                existing = next_states.get(total_count)
                if existing is None or total_gap > existing[0]:
                    thresholds = dict(previous_thresholds)
                    thresholds[capability] = threshold
                    next_states[total_count] = (total_gap, thresholds)
        states = next_states

    total = len(calibration_examples)
    candidates = []
    for accepted, (gap, thresholds) in states.items():
        policy = GatePolicy(
            name="per_capability_distance",
            max_distance=min(thresholds.values()),
            allowed_capabilities=tuple(sorted(eligible_capabilities)),
            max_distance_by_capability=tuple(sorted(thresholds.items())),
        )
        route_correct = sum(
            int(
                policy.accepts(example)
                and str(example["predicted_capability"])
                == str(example["capability"])
            )
            for example in calibration_examples
        )
        candidates.append(
            {
                "policy": policy.to_dict(),
                "coverage": accepted / total,
                "macro_hybrid_score": teacher_macro + gap,
                "macro_teacher_score": teacher_macro,
                "macro_gap_to_teacher": gap,
                "accepted_route_accuracy": route_correct / accepted if accepted else None,
            }
        )
    return candidates


def bootstrap_macro_gap(
    examples: Sequence[Mapping[str, Any]],
    policy: GatePolicy,
    indices: Sequence[int],
    *,
    samples: int = 2000,
    seed: int = 13,
) -> Dict[str, float]:
    """Stratified paired bootstrap for hybrid minus teacher macro score."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for index in indices:
        example = examples[index]
        groups.setdefault(str(example["capability"]), []).append(example)
    rng = random.Random(seed)
    gaps = []
    for _ in range(samples):
        capability_gaps = []
        for group in groups.values():
            sampled = [group[rng.randrange(len(group))] for _ in range(len(group))]
            differences = [
                (
                    float(example["local_score"])
                    if policy.accepts(example)
                    else float(example["teacher_score"])
                )
                - float(example["teacher_score"])
                for example in sampled
            ]
            capability_gaps.append(sum(differences) / len(differences))
        gaps.append(sum(capability_gaps) / len(capability_gaps))
    gaps.sort()
    return {
        "samples": int(samples),
        "mean": sum(gaps) / len(gaps),
        "ci95_low": _percentile(gaps, 2.5),
        "ci95_high": _percentile(gaps, 97.5),
    }


def _quantile_grid(
    values: Iterable[float],
    size: int,
    *,
    include_infinity: bool = False,
) -> List[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot construct a grid from empty values")
    grid = [_percentile(ordered, 100.0 * index / (size - 1)) for index in range(size)]
    if include_infinity:
        grid.append(math.inf)
    return sorted(set(grid))


def _percentile(ordered: Sequence[float], percentile: float) -> float:
    if not ordered:
        raise ValueError("percentile needs at least one value")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
