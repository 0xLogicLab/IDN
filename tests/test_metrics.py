from idn.eval.metrics import (
    accuracy_score,
    coverage,
    fallback_ratio,
    matthews_corrcoef,
    param_efficiency,
    pareto_frontier,
    risk_gap,
    selective_metric,
)


def test_core_metric_math():
    assert accuracy_score([1, 0, 1], [1, 1, 1]) == 2 / 3
    assert matthews_corrcoef([1, 0, 1, 0], [1, 0, 0, 1]) == 0.0
    assert risk_gap(0.91, 0.88) == 0.030000000000000027
    assert round(param_efficiency(0.8, 100), 3) == 0.4


def test_selective_and_fallback_metrics():
    accepted = [True, False, True, False]
    assert coverage(accepted) == 0.5
    assert fallback_ratio(accepted) == 0.5
    assert selective_metric([1, 0, 1, 0], [1, 1, 0, 0], accepted) == 0.5
    assert selective_metric([1], [0], [False]) is None


def test_pareto_frontier_keeps_only_non_dominated_rows():
    rows = [
        {"name": "tiny", "metric_value": 0.80, "added_params": 10},
        {"name": "bad", "metric_value": 0.70, "added_params": 20},
        {"name": "good", "metric_value": 0.85, "added_params": 30},
        {"name": "best", "metric_value": 0.86, "added_params": 25},
    ]

    frontier = pareto_frontier(rows)

    assert [row["name"] for row in frontier] == ["tiny", "best"]

