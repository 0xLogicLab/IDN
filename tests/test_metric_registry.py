from types import SimpleNamespace

import pytest

from idn.config import ConfigError, _optional_metrics, load_resolved_config
from idn.eval.registry import MetricRegistry, metric_specs_from_config


def test_registry_registers_and_dispatches_calculator_with_options():
    registry = MetricRegistry()
    seen = {}

    @registry.register("example.multi@v1")
    def calculate(*, predictions, references, examples, options):
        seen["options"] = options
        return {"first": len(predictions), "second": len(references)}

    result = registry.compute(
        metric_specs_from_config(
            SimpleNamespace(
                metrics=[
                    {
                        "id": "example.multi@v1",
                        "alias": "sample",
                        "options": {"threshold": 0.7},
                    }
                ]
            )
        )[0],
        predictions=[1, 2],
        references=[1, 2, 3],
        examples=[],
    )

    assert seen["options"] == {"threshold": 0.7}
    assert result["sample.first"]["value"] == 2.0
    assert result["sample.first"]["count"] == 2
    assert result["sample.first"]["metric_id"] == "example.multi@v1"
    assert result["sample.second"]["value"] == 3.0


def test_registry_rejects_duplicate_and_unknown_metric_ids():
    registry = MetricRegistry()
    registry.register("example.metric@v1", lambda **_kwargs: {"score": 1})

    with pytest.raises(ValueError, match="already registered"):
        registry.register("example.metric@v1", lambda **_kwargs: {"score": 2})
    with pytest.raises(KeyError, match="unknown metric"):
        registry.get("missing.metric@v1")


def test_capability_metrics_config_accepts_strings_and_options():
    assert _optional_metrics(
        {
            "metrics": [
                "classification.accuracy@v1",
                {
                    "id": "classification.f1@v1",
                    "alias": "macro_f1",
                    "options": {"average": "macro"},
                },
            ]
        }
    ) == [
        {"id": "classification.accuracy@v1"},
        {
            "id": "classification.f1@v1",
            "options": {"average": "macro"},
            "alias": "macro_f1",
        },
    ]


def test_capability_metrics_config_rejects_invalid_options():
    with pytest.raises(ConfigError, match="options must be a mapping"):
        _optional_metrics({"metrics": [{"id": "metric@v1", "options": []}]})


def test_legacy_metric_fields_normalize_for_transition():
    specs = metric_specs_from_config(
        SimpleNamespace(
            metrics=[], metric_name="accuracy", secondary_metrics=["mcc"]
        )
    )

    assert [spec.metric_id for spec in specs] == ["accuracy", "mcc"]


def test_capability_yaml_can_define_metrics_without_legacy_metric_name(tmp_path):
    capability_path = tmp_path / "capability.yaml"
    capability_path.write_text(
        """capability:
  id: new_task
  name: New task
  task_type: classification
  metrics:
    - id: classification.accuracy@v1
      options:
        normalize: true
  higher_is_better: true
  dataset_path: new_task
  train_split: train
  eval_split: test
  max_new_tokens: 32
  target_source: gold
""",
        encoding="utf-8",
    )
    machine_path = tmp_path / "machine.yaml"
    machine_path.write_text(
        """machine:
  name: test
  model_dir: /models
  data_dir: /datasets
  device: cpu
  train_batch_size: 1
  eval_batch_size: 1
  local_files_only: true
""",
        encoding="utf-8",
    )

    resolved = load_resolved_config(capability_path, machine_path)

    assert resolved.capability.metric_name is None
    assert resolved.capability.task_type == "classification"
    assert resolved.capability.metrics == [
        {
            "id": "classification.accuracy@v1",
            "options": {"normalize": True},
        }
    ]
