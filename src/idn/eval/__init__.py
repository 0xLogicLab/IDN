"""Evaluation modules."""

from idn.eval.runtime_benchmark import run_runtime_benchmark, summarize_runtime_benchmark
from idn.eval.dataset import BenchmarkDataset
from idn.eval.registry import (
    DEFAULT_METRICS,
    MetricRegistry,
    MetricSpec,
    MetricValue,
    register_metric,
)

__all__ = [
    "BenchmarkDataset",
    "run_runtime_benchmark",
    "summarize_runtime_benchmark",
    "DEFAULT_METRICS",
    "MetricRegistry",
    "MetricSpec",
    "MetricValue",
    "register_metric",
]

