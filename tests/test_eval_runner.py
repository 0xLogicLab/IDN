import json

from idn.capabilities.base import Example
from idn.eval.runner import run_eval


class TinySpec:
    id = "tiny"
    metric_name = "accuracy"
    higher_is_better = True

    def load_split(self, split, limit=None):
        examples = [
            Example(id="a", prompt_inputs={"text": "good"}, reference=1),
            Example(id="b", prompt_inputs={"text": "bad"}, reference=0),
        ]
        return examples[:limit] if limit is not None else examples

    def format_prompt(self, ex):
        return [{"role": "user", "content": ex.prompt_inputs["text"]}]

    def parse_prediction(self, raw_text):
        return 1 if raw_text == "positive" else 0

    def target_text(self, ex, teacher_text=None):
        return "positive" if ex.reference == 1 else "negative"

    def metric(self, preds, refs):
        correct = sum(int(pred == ref) for pred, ref in zip(preds, refs))
        return {"accuracy": correct / len(refs)}

    def verify(self, raw_text, parsed):
        return parsed in (0, 1)


class TinyModel:
    def generate(self, messages, max_new_tokens):
        text = messages[0]["content"]
        return "positive" if text == "good" else "negative"


def test_run_eval_writes_json_artifact(tmp_path):
    output = tmp_path / "eval.json"

    result = run_eval(
        spec=TinySpec(),
        split="validation",
        limit=2,
        base_model_path="/model",
        adapter_path=None,
        output_path=output,
        dtype="bfloat16",
        device="cpu",
        max_new_tokens=8,
        local_files_only=True,
        model=TinyModel(),
    )

    assert result["metrics"] == {"accuracy": 1.0}
    assert result["num_examples"] == 2

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["predictions"][0]["raw_prediction"] == "positive"
    assert saved["predictions"][0]["prompt_inputs"] == {"text": "good"}
    assert saved["model"]["base_model_path"] == "/model"


def test_run_eval_prints_header_and_verbose_records(tmp_path, capsys):
    output = tmp_path / "eval.json"

    run_eval(
        spec=TinySpec(),
        split="validation",
        limit=1,
        base_model_path="/model",
        adapter_path="/adapter",
        output_path=output,
        dtype="bfloat16",
        device="cpu",
        max_new_tokens=8,
        local_files_only=True,
        model=TinyModel(),
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "Eval configuration" in captured.out
    assert "model: /model" in captured.out
    assert "adapter: /adapter" in captured.out
    assert "samples to evaluate: 1" in captured.out
    assert "metrics: accuracy" in captured.out
    assert "input:" in captured.out
    assert "text: good" in captured.out
    assert "label: 1" in captured.out
    assert "raw prediction: positive" in captured.out
    assert "parsed prediction: 1" in captured.out
    assert "Eval result" in captured.out
