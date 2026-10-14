import json

from idn.capabilities.jsonl_chat import JsonlChatDataset
from idn.capabilities.base import Example
from idn.eval.runtime_benchmark import run_runtime_benchmark, summarize_runtime_benchmark
from idn.eval.registry import MetricRegistry, MetricSpec
from idn.framework.types import InferenceResponse


class TinyDataset:
    id = "wildchat-test"

    def load_split(self, split, limit=None):
        rows = [
            Example(
                id="ex-1",
                prompt_inputs={"messages": [{"role": "user", "content": "hello"}]},
                reference={"capability": "C0", "text": "hello back"},
                gold_target="hello back",
            ),
            Example(
                id="ex-2",
                prompt_inputs={"messages": [{"role": "user", "content": "solve"}]},
                reference={"capability": "C1", "text": "42"},
                gold_target="42",
            ),
        ]
        return rows[:limit] if limit is not None else rows

    def format_prompt(self, example):
        return example.prompt_inputs["messages"]

    def target_text(self, example, teacher_text=None):
        return example.gold_target

    def capability_label(self, example):
        return example.reference["capability"]

    def parse_prediction(self, raw_text):
        return raw_text.strip()

    def verify(self, raw_text, parsed):
        return bool(parsed)


class TinyRuntime:
    def __init__(self):
        self.calls = []

    def infer(self, request, budget):
        self.calls.append(request.latest_user_text)
        capability = "C0" if request.latest_user_text == "hello" else "C2"
        return InferenceResponse(
            text="reply to " + request.latest_user_text,
            capability=capability,
            package_id="tiny-7b",
            route_confidence=0.9,
            used_fallback=request.latest_user_text == "solve",
            cache_hit=request.latest_user_text == "hello" and len(self.calls) > 1,
        )


def test_runtime_benchmark_records_end_to_end_metrics_and_request_metadata(tmp_path):
    runtime = TinyRuntime()
    result = run_runtime_benchmark(
        dataset=TinyDataset(),
        runtime=runtime,
        split="test",
        output_dir=tmp_path,
        warmup_examples=0,
        quality_scorer=lambda example, text: {"rubric": 4.0},
        progress_every=0,
    )

    assert runtime.calls == ["hello", "solve"]
    assert result["overall"]["requests"] == 2
    assert result["overall"]["route_capability_accuracy"] == 0.5
    assert result["overall"]["fallback_rate"] == 0.5
    assert result["overall"]["cache_hit_rate"] == 0.0
    assert result["overall"]["verification_pass_rate"] == 1.0
    assert result["overall"]["quality_scores"]["rubric"] == {"mean": 4.0, "n": 2}
    assert result["by_capability"]["C1"]["route_capability_accuracy"] == 0.0

    records = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert records[0]["package_id"] == "tiny-7b"
    assert records[0]["reference"] == "hello back"
    assert (tmp_path / "run_config.json").is_file()
    assert (tmp_path / "metrics.json").is_file()


def test_jsonl_adapter_leaves_missing_capability_unlabeled(tmp_path):
    path = tmp_path / "unlabeled.jsonl"
    path.write_text(json.dumps({
        "id": "conversation-1",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    }) + "\n", encoding="utf-8")
    dataset = JsonlChatDataset(split_files={"test": path}, dataset_id="my-dataset")

    example = dataset.load_split("test")[0]

    assert dataset.capability_label(example) is None


def test_summary_handles_missing_gold_capability():
    records = [{
        "gold_capability": None,
        "routed_capability": "unknown",
        "response_nonempty": True,
        "used_fallback": False,
        "cache_hit": False,
        "latency_ms": 10.0,
    }]
    summary = summarize_runtime_benchmark(records, elapsed_total_s=1.0)
    assert summary["overall"]["route_capability_accuracy"] is None
    assert summary["by_capability"]["unknown"]["requests"] == 1


def test_runtime_benchmark_runs_configured_metric_with_options_and_alias(tmp_path):
    runtime = TinyRuntime()
    registry = MetricRegistry()
    calls = {}

    @registry.register("test.reply_stats@v1")
    def reply_stats(*, predictions, references, examples, options):
        calls["options"] = options
        calls["sizes"] = (len(predictions), len(references), len(examples))
        return {"matched": 0.5, "nonempty": 1.0}

    result = run_runtime_benchmark(
        dataset=TinyDataset(),
        runtime=runtime,
        split="test",
        output_dir=tmp_path,
        warmup_examples=0,
        metric_specs=[
            MetricSpec(
                "test.reply_stats@v1",
                options={"mode": "strict"},
                alias="reply",
            )
        ],
        metric_registry=registry,
        progress_every=0,
    )

    assert calls == {"options": {"mode": "strict"}, "sizes": (2, 2, 2)}
    assert result["task_metrics"]["reply.matched"] == {
        "value": 0.5,
        "count": 2,
        "status": "ok",
        "higher_is_better": None,
        "metric_id": "test.reply_stats@v1",
        "options": {"mode": "strict"},
    }


def test_runtime_benchmark_uses_legacy_capability_metric_when_no_metric_specs(tmp_path):
    class DatasetWithMetric(TinyDataset):
        def metric(self, predictions, references):
            return {"legacy_score": 0.75}

    result = run_runtime_benchmark(
        dataset=DatasetWithMetric(),
        runtime=TinyRuntime(),
        split="test",
        output_dir=tmp_path,
        warmup_examples=0,
        progress_every=0,
    )

    assert result["task_metrics"]["legacy_score"]["value"] == 0.75


def test_runtime_benchmark_reads_metric_declarations_from_dataset_config(tmp_path):
    class ConfiguredDataset(TinyDataset):
        config = type(
            "Config",
            (),
            {
                "metrics": [
                    {
                        "id": "test.count@v1",
                        "options": {"offset": 2},
                    }
                ],
                "metric_modules": [],
            },
        )()

    registry = MetricRegistry()
    registry.register(
        "test.count@v1",
        lambda *, predictions, references, examples, options: {
            "count": len(predictions) + options["offset"]
        },
    )
    result = run_runtime_benchmark(
        dataset=ConfiguredDataset(),
        runtime=TinyRuntime(),
        split="test",
        output_dir=tmp_path,
        warmup_examples=0,
        metric_registry=registry,
        progress_every=0,
    )

    assert result["task_metrics"]["count"]["value"] == 4.0
