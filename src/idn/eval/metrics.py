"""Core metric helpers used by evaluation and frontier reporting."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional


def accuracy_score(preds: List[int], refs: List[int]) -> float:
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    if not refs:
        return 0.0
    correct = sum(int(pred == ref) for pred, ref in zip(preds, refs))
    return correct / len(refs)


def matthews_corrcoef(preds: List[int], refs: List[int]) -> float:
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    tp = tn = fp = fn = 0
    for pred, ref in zip(preds, refs):
        if pred == 1 and ref == 1:
            tp += 1
        elif pred == 0 and ref == 0:
            tn += 1
        elif pred == 1 and ref == 0:
            fp += 1
        elif pred == 0 and ref == 1:
            fn += 1
        else:
            raise ValueError("matthews_corrcoef expects binary labels 0/1")

    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / denominator


def risk_gap(teacher_metric: float, student_metric: float) -> float:
    return teacher_metric - student_metric


def param_efficiency(metric_value: float, added_params: int) -> float:
    if added_params <= 1:
        return metric_value
    return metric_value / math.log10(added_params)


def coverage(accepted: Iterable[bool]) -> float:
    values = list(accepted)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def fallback_ratio(accepted: Iterable[bool]) -> float:
    return 1.0 - coverage(accepted)


def selective_metric(
    preds: List[int],
    refs: List[int],
    accepted: List[bool],
    metric_fn=accuracy_score,
) -> Optional[float]:
    if not (len(preds) == len(refs) == len(accepted)):
        raise ValueError("preds, refs, and accepted must have the same length")
    selected_preds = [pred for pred, keep in zip(preds, accepted) if keep]
    selected_refs = [ref for ref, keep in zip(refs, accepted) if keep]
    if not selected_refs:
        return None
    return metric_fn(selected_preds, selected_refs)


def pareto_frontier(
    rows: Iterable[Mapping[str, float]],
    metric_key: str = "metric_value",
    params_key: str = "added_params",
) -> List[Dict[str, float]]:
    """Return rows not dominated by another row with >= metric and <= params."""

    normalized_rows = [dict(row) for row in rows]
    frontier = []
    for row in normalized_rows:
        dominated = False
        for other in normalized_rows:
            if other is row:
                continue
            at_least_as_good = other[metric_key] >= row[metric_key]
            no_more_params = other[params_key] <= row[params_key]
            strictly_better = (
                other[metric_key] > row[metric_key]
                or other[params_key] < row[params_key]
            )
            if at_least_as_good and no_more_params and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda item: (item[params_key], -item[metric_key]))

