"""Runnable wiring example without downloading or loading a model."""

from types import SimpleNamespace

from idn.framework import (
    IDNRuntime,
    InMemoryResponseCache,
    InferenceRequest,
    ModelPackage,
    ModelPackageRegistry,
    ModelProviderRegistry,
    ResourceBudget,
    RuntimePolicy,
)


CAPABILITY_NAMES = {
    "C0": "Creative / Narrative",
    "C1": "General / Informational",
    "C2": "Technical / Formal",
    "C3": "Professional / Academic",
}

# Synthetic values only, to demonstrate budget-based selection. Replace them
# with measured memory, latency, and quality for the target hardware/protocol.
QWEN25_SIZE_PROFILES = {
    1.5: {"memory_gib": 5.0, "latency_ms": 80.0, "quality": 0.62},
    7.0: {"memory_gib": 20.0, "latency_ms": 160.0, "quality": 0.78},
    14.0: {"memory_gib": 40.0, "latency_ms": 280.0, "quality": 0.84},
    72.0: {"memory_gib": 180.0, "latency_ms": 800.0, "quality": 0.90},
}


class DemoRouter:
    def route(self, text):
        # 演示代码用显式 [C0]...[C3] 标记模拟路由，不是训练好的分类器。
        capability = next(
            (name for name in CAPABILITY_NAMES if text.startswith("[{}]".format(name))),
            None,
        )
        return SimpleNamespace(
            capability=capability or "unknown",
            confidence=0.91 if capability else 0.0,
            in_scope=capability is not None,
        )


class DemoProvider:
    def generate(self, package, request):
        return "[{}] demo response to: {}".format(
            package.package_id, request.latest_user_text
        )


def make_wildchat_package(capability, size_b):
    """Build one illustrative model/adapter package for a WildChat capability."""
    size_label = str(int(size_b)) if float(size_b).is_integer() else str(size_b)
    size_name = "{}b".format(size_label)
    profile = QWEN25_SIZE_PROFILES[size_b]
    return ModelPackage(
        package_id="wildchat-{}-qwen-{}".format(capability, size_name),
        capability=capability,
        model_path="/models/Qwen2.5-{}B-Instruct".format(size_label),
        adapter_path="artifacts/adapters/wildchat/{}/qwen-{}".format(
            capability, size_name
        ),
        model_size_b=size_b,
        estimated_memory_gib=profile["memory_gib"],
        estimated_latency_ms=profile["latency_ms"],
        quality_lower_bound=profile["quality"],
        certified=False,
        provider_id="demo",
    )


packages = ModelPackageRegistry(
    [
        make_wildchat_package(capability, size_b)
        for capability in CAPABILITY_NAMES
        for size_b in QWEN25_SIZE_PROFILES
    ]
)

providers = ModelProviderRegistry()
providers.register("demo", DemoProvider())
runtime = IDNRuntime(
    router=DemoRouter(),
    packages=packages,
    provider=providers,
    cache=InMemoryResponseCache(capacity=128, ttl_seconds=120),
    deployment_version="demo-bundle-v1",
    # 示例用假模型和合成占位分数，因此关闭认证门槛；真实部署应使用默认认证策略。
    policy=RuntimePolicy(require_certification=False),
)
request = InferenceRequest.from_messages(
    [{"role": "user", "content": "[C0] Write a short story about a robot."}],
    max_new_tokens=64,
)
small = runtime.infer(
    request,
    ResourceBudget(max_memory_gib=8.0, min_quality_lower_bound=0.60),
)
larger = runtime.infer(
    request,
    ResourceBudget(max_memory_gib=24.0, min_quality_lower_bound=0.75),
)
print(small.package_id, small.text)
print(larger.package_id, larger.text)
