import json
from types import SimpleNamespace

import pytest

from idn.capabilities.jsonl_chat import JsonlChatDataset
from idn.framework import (
    DatasetRegistry,
    IDNRuntime,
    InMemoryResponseCache,
    InferenceRequest,
    ModelPackage,
    ModelPackageRegistry,
    ModelProviderRegistry,
    OpenAICompatibleProvider,
    ResourceBudget,
    RuntimePolicy,
    build_default_dataset_registry,
)
from idn.framework.runtime import FrameworkError
from idn.models.loader import ChatModel


class FakeRouter:
    def __init__(self, capability="math", confidence=0.9, in_scope=True):
        self.capability = capability
        self.confidence = confidence
        self.in_scope = in_scope
        self.calls = 0

    def route(self, text):
        self.calls += 1
        return SimpleNamespace(
            capability=self.capability,
            confidence=self.confidence,
            in_scope=self.in_scope,
        )


class FakeProvider:
    def __init__(self):
        self.calls = []

    def generate(self, package, request):
        self.calls.append((package.package_id, request.latest_user_text))
        return "answer from " + package.package_id


def package(
    package_id,
    *,
    capability="math",
    size=1.5,
    memory=8.0,
    quality=0.7,
    certified=True,
    fallback=False,
):
    return ModelPackage(
        package_id=package_id,
        capability=capability,
        model_path="/models/" + package_id,
        model_size_b=size,
        estimated_memory_gib=memory,
        estimated_latency_ms=100.0,
        quality_lower_bound=quality,
        certified=certified,
        is_fallback=fallback,
    )


def make_request(text="solve 2+2", tenant_id="lab"):
    return InferenceRequest.from_messages(
        [{"role": "user", "content": text}],
        max_new_tokens=32,
        tenant_id=tenant_id,
    )


def test_runtime_uses_exact_cache_before_routing_and_generation():
    router = FakeRouter()
    provider = FakeProvider()
    runtime = IDNRuntime(
        router=router,
        packages=ModelPackageRegistry([package("math-1b")]),
        provider=provider,
        cache=InMemoryResponseCache(),
        deployment_version="test-v1",
    )

    first = runtime.infer(make_request())
    second = runtime.infer(make_request())

    assert first.text == "answer from math-1b"
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert router.calls == 1
    assert len(provider.calls) == 1


def test_resource_budget_selects_smallest_certified_model_meeting_quality():
    router = FakeRouter()
    provider = FakeProvider()
    packages = ModelPackageRegistry(
        [
            package("math-1b", size=1.5, memory=8.0, quality=0.72),
            package("math-7b", size=7.0, memory=20.0, quality=0.88),
        ]
    )
    runtime = IDNRuntime(router=router, packages=packages, provider=provider)

    small = runtime.infer(
        make_request("small-budget"),
        ResourceBudget(max_memory_gib=10.0, min_quality_lower_bound=0.7),
    )
    larger = runtime.infer(
        make_request("higher-quality"),
        ResourceBudget(max_memory_gib=24.0, min_quality_lower_bound=0.85),
    )

    assert small.package_id == "math-1b"
    assert larger.package_id == "math-7b"


def test_known_quality_ranks_before_unknown_when_resources_tie():
    registry = ModelPackageRegistry(
        [
            package("unknown-quality", memory=8.0, quality=None),
            package("measured-quality", memory=8.0, quality=0.65),
        ]
    )

    selected = registry.select("math", ResourceBudget())

    assert selected is not None
    assert selected.package_id == "measured-quality"


def test_out_of_scope_request_uses_configured_fallback():
    router = FakeRouter(capability="math", in_scope=False)
    provider = FakeProvider()
    runtime = IDNRuntime(
        router=router,
        packages=ModelPackageRegistry(
            [
                package("math-1b"),
                package(
                    "general-fallback",
                    capability=None,
                    size=72.0,
                    memory=80.0,
                    quality=0.95,
                    fallback=True,
                ),
            ]
        ),
        provider=provider,
    )

    result = runtime.infer(make_request())

    assert result.package_id == "general-fallback"
    assert result.used_fallback is True
    assert result.fallback_reason == "request_out_of_scope"


def test_fallback_must_fit_the_memory_budget():
    runtime = IDNRuntime(
        router=FakeRouter(in_scope=False),
        packages=ModelPackageRegistry(
            [
                package(
                    "large-fallback",
                    capability=None,
                    size=72.0,
                    memory=80.0,
                    quality=0.95,
                    fallback=True,
                )
            ]
        ),
        provider=FakeProvider(),
    )

    with pytest.raises(FrameworkError, match="no fallback package"):
        runtime.infer(make_request(), ResourceBudget(max_memory_gib=32.0))


def test_remote_package_is_not_filtered_by_callers_local_memory_budget():
    registry = ModelPackageRegistry(
        [
            ModelPackage(
                package_id="remote-math",
                capability="math",
                backend_model_id="qwen25-7b",
                provider_id="vllm",
                certified=True,
                is_remote=True,
            )
        ]
    )

    selected = registry.select("math", ResourceBudget(max_memory_gib=4.0))

    assert selected is not None
    assert selected.package_id == "remote-math"


def test_uncertified_package_is_not_used_without_fallback():
    runtime = IDNRuntime(
        router=FakeRouter(),
        packages=ModelPackageRegistry([package("uncertified", certified=False)]),
        provider=FakeProvider(),
    )

    with pytest.raises(FrameworkError, match="no fallback package"):
        runtime.infer(make_request())


def test_cache_isolated_by_tenant_and_deployment_version():
    provider = FakeProvider()
    runtime = IDNRuntime(
        router=FakeRouter(),
        packages=ModelPackageRegistry([package("math-1b")]),
        provider=provider,
        cache=InMemoryResponseCache(),
        deployment_version="model-bundle-a",
    )

    runtime.infer(make_request(tenant_id="group-a"))
    runtime.infer(make_request(tenant_id="group-b"))

    assert len(provider.calls) == 2


def test_jsonl_chat_adapter_expands_supervised_assistant_turns(tmp_path):
    path = tmp_path / "train.jsonl"
    row = {
        "id": "conversation-7",
        "capability": 2,
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Earlier answer", "loss": False},
            {"role": "user", "content": "Second question", "loss": False},
            {"role": "assistant", "content": "Target answer", "loss": True},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = JsonlChatDataset(
        dataset_id="wildchat",
        split_files={"train": path},
        capability_map={"2": "C2"},
    )

    examples = dataset.load_split("train")
    prompt = dataset.format_prompt(examples[0])

    assert len(examples) == 1
    assert examples[0].id == "conversation-7-turn-0"
    assert examples[0].prompt_inputs["capability"] == "C2"
    assert prompt[-1] == {"role": "user", "content": "Second question"}
    assert dataset.target_text(examples[0]) == "Target answer"


def test_default_dataset_registry_keeps_six_tasks_and_adds_jsonl_chat():
    names = set(build_default_dataset_registry().names())

    assert {"sst2", "cola", "gsm8k", "squad", "mbpp", "cnn_dailymail"} <= names
    assert "jsonl_chat" in names


def test_provider_registry_dispatches_by_package_provider_id():
    router = FakeRouter()
    provider_registry = ModelProviderRegistry()
    fake_provider = FakeProvider()
    provider_registry.register("local-test", fake_provider)
    package_with_provider = ModelPackage(
        package_id="math-7b",
        capability="math",
        model_path="/models/math-7b",
        certified=True,
        provider_id="local-test",
    )
    runtime = IDNRuntime(
        router=router,
        packages=ModelPackageRegistry([package_with_provider]),
        provider=provider_registry,
    )

    result = runtime.infer(make_request())

    assert result.package_id == "math-7b"
    assert fake_provider.calls == [("math-7b", "solve 2+2")]


def test_openai_compatible_provider_uses_backend_model_id_and_chat_api():
    calls = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"endpoint answer"}}]}'

    def fake_opener(request, timeout):
        calls["url"] = request.full_url
        calls["body"] = json.loads(request.data.decode("utf-8"))
        calls["timeout"] = timeout
        return FakeResponse()

    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:8000/v1",
        timeout_seconds=9,
        opener=fake_opener,
    )
    remote_package = ModelPackage(
        package_id="remote-qwen",
        capability="math",
        backend_model_id="qwen25-7b",
        provider_id="vllm",
        is_remote=True,
    )

    answer = provider.generate(remote_package, make_request())

    assert answer == "endpoint answer"
    assert calls["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert calls["body"]["model"] == "qwen25-7b"
    assert calls["body"]["messages"] == [{"role": "user", "content": "solve 2+2"}]


def test_qwen2_family_uses_editable_backend_for_all_checkpoint_sizes(monkeypatch):
    calls = []

    def fake_qwen2_loader(path, **kwargs):
        calls.append((path, kwargs))
        return "qwen2-model"

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            raise AssertionError("qwen2 must use the editable local backend")

    monkeypatch.setattr(
        "idn.models.loader.load_local_qwen2_for_causal_lm",
        fake_qwen2_loader,
    )
    for size in ("1.5", "7", "14", "72"):
        model = ChatModel("/models/Qwen2.5-{}B-Instruct".format(size))
        loaded = model._load_causal_lm(
            FakeAutoModel,
            "qwen2",  # Qwen2.5 config.json uses this architecture id.
            {"torch_dtype": "bf16"},
        )
        assert loaded == "qwen2-model"

    assert len(calls) == 4
    assert [call[0] for call in calls] == [
        "/models/Qwen2.5-1.5B-Instruct",
        "/models/Qwen2.5-7B-Instruct",
        "/models/Qwen2.5-14B-Instruct",
        "/models/Qwen2.5-72B-Instruct",
    ]


def test_other_text_architectures_use_transformers_auto_mapping():
    calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return "generic-causal-lm"

    model = ChatModel("/models/Llama-3.1-8B-Instruct")
    loaded = model._load_causal_lm(
        FakeAutoModel,
        "llama",
        {"torch_dtype": "bf16"},
    )

    assert loaded == "generic-causal-lm"
    assert calls == [("/models/Llama-3.1-8B-Instruct", {"torch_dtype": "bf16"})]
