from pathlib import Path

from idn.cli import default_eval_output_path, main, safe_slug


ROOT = Path(__file__).resolve().parents[1]


def test_safe_slug_keeps_paths_filename_friendly():
    assert safe_slug("Qwen2.5-1.5B-Instruct") == "Qwen2.5-1.5B-Instruct"
    assert safe_slug("/data/model/name") == "data-model-name"
    assert safe_slug("   ") == "unknown"


def test_default_eval_output_path_includes_eval_identity():
    output = default_eval_output_path(
        artifact_dir=Path("artifacts"),
        capability_id="sst2",
        split="validation",
        model_path="/data/huggingface/model/Qwen2.5-1.5B-Instruct",
        adapter_path=None,
        limit=200,
        dtype="bfloat16",
    )

    assert output == Path(
        "artifacts/eval/sst2_validation_Qwen2.5-1.5B-Instruct_base_limit200_bfloat16.json"
    )


def test_cli_eval_refuses_existing_output_without_override(tmp_path, capsys):
    output = tmp_path / "eval.json"
    output.write_text("{}", encoding="utf-8")

    exit_code = main(
        [
            "--machine",
            str(ROOT / "configs/machine/local_3090.yaml"),
            "eval",
            "--config",
            str(ROOT / "configs/capabilities/sst2.yaml"),
            "--limit",
            "1",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Re-run with --override" in captured.err

