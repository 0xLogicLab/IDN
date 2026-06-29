from pathlib import Path

import pytest

from idn.cli import main
from idn.config import ConfigError, deep_merge, load_resolved_config
from idn.registry import load_registry


ROOT = Path(__file__).resolve().parents[1]


def test_deep_merge_recurses_without_mutating_inputs():
    base = {"capability": {"id": "sst2", "eval_limit": 1000}, "x": 1}
    override = {"capability": {"eval_limit": 20}, "machine": {"name": "local"}}

    merged = deep_merge(base, override)

    assert merged["capability"]["id"] == "sst2"
    assert merged["capability"]["eval_limit"] == 20
    assert merged["machine"]["name"] == "local"
    assert base["capability"]["eval_limit"] == 1000


def test_resolved_config_rejects_fp16(tmp_path):
    machine = tmp_path / "machine.yaml"
    machine.write_text(
        """
machine:
  name: test
  teacher_model_path: /teacher
  student_model_path: /student
  dtype: fp16
  device: cuda
  train_batch_size: 1
  gradient_accumulation_steps: 1
  eval_batch_size: 1
  artifact_dir: artifacts
  eval_limit: 1
  local_files_only: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="fp16 is forbidden"):
        load_resolved_config(ROOT / "configs/capabilities/sst2.yaml", machine)


def test_registry_loads_all_phase0_capabilities():
    registry = load_registry(
        ROOT / "configs/capabilities", ROOT / "configs/machine/local_3090.yaml"
    )

    assert set(registry) == {
        "cnn_dailymail",
        "cola",
        "gsm8k",
        "mbpp",
        "squad",
        "sst2",
    }
    assert registry["sst2"].config.dataset_path == "/data/huggingface/dataset/glue_sst2"
    assert registry["sst2"].spec is not None
    assert registry["cola"].spec is not None
    assert registry["gsm8k"].spec is None


def test_cli_list_capabilities_prints_resolved_paths(capsys):
    exit_code = main(
        [
            "--machine",
            str(ROOT / "configs/machine/local_3090.yaml"),
            "list-capabilities",
            "--config-dir",
            str(ROOT / "configs/capabilities"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "sst2" in captured.out
    assert "/data/huggingface/model/Qwen2.5-1.5B-Instruct" in captured.out
    assert "/data/huggingface/model/Qwen2.5-7B-Instruct" in captured.out
