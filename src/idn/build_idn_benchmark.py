from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    task_type: str
    capability: str
    difficulty: str
    source: str
    prompt: str
    answer: dict[str, Any]
    metadata: dict[str, Any]


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


CAPABILITIES = [
    {
        "capability": "translation",
        "task": "Translate short text between English and Chinese while preserving meaning.",
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "latency_class": "interactive",
        "privacy_class": "tenant",
        "quality_tier": "standard",
        "resource_requirement": "edge_1_5b_or_regional_7b",
    },
    {
        "capability": "summarization",
        "task": "Summarize long text into concise bullet points.",
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "latency_class": "normal",
        "privacy_class": "tenant",
        "quality_tier": "standard",
        "resource_requirement": "regional_7b",
    },
    {
        "capability": "json_extraction",
        "task": "Extract structured fields into a strict JSON object.",
        "input_modalities": ["text"],
        "output_modalities": ["json"],
        "latency_class": "interactive",
        "privacy_class": "private",
        "quality_tier": "high",
        "resource_requirement": "regional_7b_with_schema_verifier",
    },
    {
        "capability": "code_assistance",
        "task": "Generate or debug small code snippets with executable constraints.",
        "input_modalities": ["text", "code"],
        "output_modalities": ["code", "text"],
        "latency_class": "normal",
        "privacy_class": "tenant",
        "quality_tier": "high",
        "resource_requirement": "cloud_teacher_or_regional_7b",
    },
    {
        "capability": "math_reasoning",
        "task": "Solve multi-step arithmetic or symbolic reasoning problems.",
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "latency_class": "normal",
        "privacy_class": "public",
        "quality_tier": "high",
        "resource_requirement": "cloud_teacher",
    },
    {
        "capability": "policy_sensitive_qa",
        "task": "Answer private or regulated-domain questions under locality and audit constraints.",
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "latency_class": "normal",
        "privacy_class": "regulated",
        "quality_tier": "high",
        "resource_requirement": "trusted_local_or_cloud_with_attestation",
    },
    {
        "capability": "vision_detection",
        "task": "Detect objects or events from image or video input.",
        "input_modalities": ["image", "video"],
        "output_modalities": ["json", "text"],
        "latency_class": "realtime",
        "privacy_class": "private",
        "quality_tier": "standard",
        "resource_requirement": "edge_gpu_vision_runtime",
    },
    {
        "capability": "rag_qa",
        "task": "Answer questions using retrieved context and cite supporting evidence.",
        "input_modalities": ["text", "retrieved_context"],
        "output_modalities": ["text", "citations"],
        "latency_class": "normal",
        "privacy_class": "tenant",
        "quality_tier": "high",
        "resource_requirement": "regional_7b_with_retrieval_cache",
    },
]

RESOURCES = [
    {
        "node_id": "local-phone-npu",
        "tier": "local",
        "accelerator": "mobile_npu",
        "vram_gb": 4,
        "trust_level": "device_bound",
        "privacy_scope": ["private", "tenant", "public"],
        "models": ["qwen2.5-1.5b-int4"],
        "load": "low",
        "latency_to_user_ms": 8,
    },
    {
        "node_id": "campus-edge-3090",
        "tier": "edge",
        "accelerator": "rtx3090",
        "vram_gb": 24,
        "trust_level": "site_attested",
        "privacy_scope": ["private", "tenant", "public"],
        "models": ["qwen2.5-1.5b", "qwen2.5-7b", "vision-small"],
        "load": "medium",
        "latency_to_user_ms": 18,
    },
    {
        "node_id": "regional-pop-a100",
        "tier": "regional",
        "accelerator": "a100",
        "vram_gb": 80,
        "trust_level": "provider_attested",
        "privacy_scope": ["tenant", "public"],
        "models": ["qwen2.5-7b", "qwen2.5-14b", "rag-runtime"],
        "load": "low",
        "latency_to_user_ms": 42,
    },
    {
        "node_id": "cloud-teacher-h100",
        "tier": "cloud",
        "accelerator": "h100_cluster",
        "vram_gb": 640,
        "trust_level": "provider_attested",
        "privacy_scope": ["regulated", "tenant", "public"],
        "models": ["teacher-large", "qwen2.5-72b"],
        "load": "medium",
        "latency_to_user_ms": 120,
    },
]

REQUESTS = [
    {
        "request": "Translate a two-sentence customer support message from English to Chinese under 150 ms.",
        "capability": "translation",
        "quality_target": "standard",
        "latency_ms": 150,
        "privacy": "tenant",
        "expected_route": "edge",
        "fallback": False,
        "reason": "interactive latency and tenant privacy fit an attested edge node with a compact language model",
    },
    {
        "request": "Summarize a 20-page internal incident report with names and customer IDs.",
        "capability": "summarization",
        "quality_target": "high",
        "latency_ms": 3000,
        "privacy": "private",
        "expected_route": "edge",
        "fallback": True,
        "reason": "private data should stay local or edge; high quality requires verification and fallback on low confidence",
    },
    {
        "request": "Extract task, latency target, privacy label, and fallback policy from a capability description as strict JSON.",
        "capability": "json_extraction",
        "quality_target": "high",
        "latency_ms": 800,
        "privacy": "tenant",
        "expected_route": "regional",
        "fallback": True,
        "reason": "schema validity is critical, so route to a stronger model with a schema verifier",
    },
    {
        "request": "Solve a multi-step math word problem where the final numeric answer must be exact.",
        "capability": "math_reasoning",
        "quality_target": "high",
        "latency_ms": 5000,
        "privacy": "public",
        "expected_route": "cloud",
        "fallback": True,
        "reason": "high-accuracy multi-step reasoning benefits from the teacher model",
    },
    {
        "request": "Answer an enterprise RAG question using cached policy documents and cite two evidence spans.",
        "capability": "rag_qa",
        "quality_target": "high",
        "latency_ms": 2500,
        "privacy": "tenant",
        "expected_route": "regional",
        "fallback": True,
        "reason": "retrieval cache and citation verification are available at the regional node",
    },
    {
        "request": "Detect whether a factory camera frame contains a safety helmet violation within 80 ms.",
        "capability": "vision_detection",
        "quality_target": "standard",
        "latency_ms": 80,
        "privacy": "private",
        "expected_route": "edge",
        "fallback": False,
        "reason": "video privacy and real-time latency require edge execution",
    },
    {
        "request": "Debug a Python function that fails a provided unit test.",
        "capability": "code_assistance",
        "quality_target": "high",
        "latency_ms": 4000,
        "privacy": "tenant",
        "expected_route": "cloud",
        "fallback": True,
        "reason": "code correctness should be verified by tests and escalated when local confidence is low",
    },
    {
        "request": "Answer a regulated medical triage question containing private patient symptoms.",
        "capability": "policy_sensitive_qa",
        "quality_target": "high",
        "latency_ms": 5000,
        "privacy": "regulated",
        "expected_route": "cloud",
        "fallback": True,
        "reason": "regulated content needs trusted execution, auditability, and high-quality reasoning",
    },
]


def capability_descriptor_examples(start: int) -> list[BenchmarkExample]:
    out = []
    for i, cap in enumerate(CAPABILITIES, start=start):
        prompt = (
            "Extract an IDN capability descriptor from the service description. "
            "Return JSON with fields capability, task, input_modalities, output_modalities, "
            "quality_tier, latency_class, privacy_class, resource_requirement.\n\n"
            f"Service description: {cap['task']} Privacy class is {cap['privacy_class']}. "
            f"Expected latency is {cap['latency_class']}. Required quality tier is {cap['quality_tier']}."
        )
        answer = {k: cap[k] for k in ["capability", "task", "input_modalities", "output_modalities", "quality_tier", "latency_class", "privacy_class", "resource_requirement"]}
        out.append(BenchmarkExample(f"idn-desc-{i:04d}", "capability_descriptor_extraction", cap["capability"], "easy", "idn-paper.tex:261-268; capability_extraction_implementation_plan.md:368-399", prompt, answer, {"schema": "capability_descriptor_v1"}))
    return out


def route_examples(start: int) -> list[BenchmarkExample]:
    out = []
    for i, req in enumerate(REQUESTS, start=start):
        prompt = (
            "You are the IDN service router. Choose one route among local, edge, regional, cloud. "
            "Return JSON with route, fallback_required, and reason.\n\n"
            f"Request: {req['request']}\n"
            f"Capability: {req['capability']}\nQuality target: {req['quality_target']}\n"
            f"Latency target ms: {req['latency_ms']}\nPrivacy: {req['privacy']}\n"
            f"Available resources: {json.dumps(RESOURCES, ensure_ascii=False)}"
        )
        answer = {"route": req["expected_route"], "fallback_required": req["fallback"], "reason": req["reason"]}
        diff = "medium" if req["fallback"] else "easy"
        out.append(BenchmarkExample(f"idn-route-{i:04d}", "route_decision", req["capability"], diff, "idn-paper.tex:314-343; idn-paper.tex:392-395", prompt, answer, {"latency_target_ms": req["latency_ms"], "privacy": req["privacy"]}))
    return out


def fallback_examples(start: int) -> list[BenchmarkExample]:
    scenarios = [
        ("routing_score=0.93, verifier_pass=true, schema_valid=true, privacy=public", False, "high routing score and verifier pass allow local serving"),
        ("routing_score=0.41, verifier_pass=true, schema_valid=true, privacy=tenant", True, "routing confidence is below the acceptance gate"),
        ("routing_score=0.88, verifier_pass=false, schema_valid=false, privacy=tenant", True, "verifier failure and invalid schema require teacher fallback"),
        ("routing_score=0.76, verifier_pass=true, schema_valid=true, privacy=regulated, node_trust=unattested", True, "regulated requests require trusted execution evidence"),
        ("routing_score=0.82, verifier_pass=true, schema_valid=true, queue_delay_ms=900, latency_target_ms=300", True, "queue delay violates latency target, so route to a compatible fallback"),
        ("routing_score=0.79, verifier_pass=true, schema_valid=true, cache_hit=true, state_compatible=true", False, "compatible state reuse supports local completion"),
        ("routing_score=0.91, verifier_pass=true, schema_valid=true, state_compatible=false, task=long_context", True, "state incompatibility can dominate long-context latency and correctness"),
        ("routing_score=0.68, verifier_pass=true, schema_valid=true, task=math_reasoning, quality=high", True, "high-quality reasoning should escalate when confidence is moderate"),
    ]
    out = []
    for i, (signal, fallback, reason) in enumerate(scenarios, start=start):
        prompt = (
            "Decide whether an IDN edge response should be accepted or fall back to the teacher. "
            "Return JSON with accept_local, fallback_required, target, and reason.\n\n"
            f"Signals: {signal}"
        )
        answer = {"accept_local": not fallback, "fallback_required": fallback, "target": "teacher" if fallback else "local", "reason": reason}
        out.append(BenchmarkExample(f"idn-fallback-{i:04d}", "fallback_decision", "service_routing", "medium", "capability_extraction_implementation_plan.md:404-436", prompt, answer, {"signals": signal}))
    return out


def resource_fit_examples(start: int) -> list[BenchmarkExample]:
    pairs = [
        ("translation", "local-phone-npu", True, "compact translation can run on a local quantized model"),
        ("math_reasoning", "local-phone-npu", False, "multi-step high-quality reasoning exceeds the local model capability"),
        ("vision_detection", "campus-edge-3090", True, "edge GPU hosts the vision runtime and meets privacy constraints"),
        ("policy_sensitive_qa", "regional-pop-a100", False, "regional node privacy scope excludes regulated requests"),
        ("rag_qa", "regional-pop-a100", True, "regional node has retrieval cache and enough accelerator memory"),
        ("code_assistance", "campus-edge-3090", False, "high-quality code repair may require teacher fallback and test execution"),
        ("json_extraction", "regional-pop-a100", True, "regional model plus schema verifier fits structured extraction"),
        ("summarization", "cloud-teacher-h100", True, "cloud teacher can serve high-quality summarization within relaxed latency"),
    ]
    cap_by_name = {c["capability"]: c for c in CAPABILITIES}
    res_by_id = {r["node_id"]: r for r in RESOURCES}
    out = []
    for i, (capability, node_id, feasible, reason) in enumerate(pairs, start=start):
        prompt = (
            "Judge whether the resource profile can serve the requested capability. "
            "Return JSON with feasible, route_tier, and reason.\n\n"
            f"Capability descriptor: {json.dumps(cap_by_name[capability], ensure_ascii=False)}\n"
            f"Resource profile: {json.dumps(res_by_id[node_id], ensure_ascii=False)}"
        )
        answer = {"feasible": feasible, "route_tier": res_by_id[node_id]["tier"] if feasible else "fallback", "reason": reason}
        out.append(BenchmarkExample(f"idn-resource-{i:04d}", "resource_fit", capability, "medium", "idn-paper.tex:283-289; idn-paper.tex:235-238", prompt, answer, {"node_id": node_id}))
    return out


def cache_examples(start: int) -> list[BenchmarkExample]:
    scenarios = [
        ("popular system prompt prefix, public scope, 70% predicted reuse, 4 KB", "cache", "prefix cache has high reuse and low privacy risk"),
        ("session-private KV cache, same user returns within 30 seconds, compatible model hash", "reuse_local", "session affinity and compatibility make KV reuse valuable"),
        ("tenant document retrieval result, tenant-shared scope, matching access policy", "cache", "retrieval result can be reused within tenant policy"),
        ("private medical context, public cache scope requested", "reject_cache", "privacy label conflicts with sharing scope"),
        ("large KV cache stored on another edge, transfer cost exceeds recompute cost", "recompute", "state movement cost exceeds expected latency benefit"),
        ("adapter weights for high-demand translation capability at regional node", "cache", "artifact cache reduces load and startup overhead"),
        ("result cache for nondeterministic creative generation", "reject_cache", "nondeterministic output weakens safe result reuse"),
        ("signed prompt template with compatible hash and public policy", "reuse_local", "signature and compatibility allow safe reuse"),
    ]
    out = []
    for i, (state, action, reason) in enumerate(scenarios, start=start):
        prompt = (
            "Choose an IDN cache action among cache, reuse_local, recompute, reject_cache. "
            "Return JSON with action, cache_type, and reason.\n\n"
            f"State candidate: {state}"
        )
        cache_type = "kv_cache" if "KV" in state else "artifact" if "adapter" in state else "result" if "result" in state else "prefix_or_context"
        answer = {"action": action, "cache_type": cache_type, "reason": reason}
        out.append(BenchmarkExample(f"idn-cache-{i:04d}", "cache_decision", "state_aware_caching", "medium", "idn-paper.tex:364-380", prompt, answer, {"state_candidate": state}))
    return out


def trust_examples(start: int) -> list[BenchmarkExample]:
    scenarios = [
        ("capability descriptor signed by provider root and deployed on site-attested edge", True, "signature chain and node attestation satisfy trust policy"),
        ("cached state has compatible hash but missing signature metadata", False, "state integrity cannot be verified"),
        ("regional controller delegates signing authority to site agent for prefix cache objects", True, "delegated authority matches object class"),
        ("private context object stored unencrypted in tenant-shared cache", False, "private state requires encryption and restricted scope"),
        ("execution receipt records capability version, node attestation, policy verdict, and accounting record", True, "receipt supports audit and compliance"),
        ("adapter package from unknown source with missing lineage", False, "model supply-chain provenance is missing"),
        ("public translation capability with signed lineage and public privacy label", True, "public capability has sufficient provenance"),
        ("regulated request routed to unattested local node", False, "regulated policy requires trusted execution evidence"),
    ]
    out = []
    for i, (case, allowed, reason) in enumerate(scenarios, start=start):
        prompt = (
            "Evaluate an IDN trust-policy case. Return JSON with allowed, required_evidence, and reason.\n\n"
            f"Case: {case}"
        )
        required = ["signature", "lineage", "attestation"] if allowed else ["signature", "encryption", "attestation", "lineage"]
        answer = {"allowed": allowed, "required_evidence": required, "reason": reason}
        out.append(BenchmarkExample(f"idn-trust-{i:04d}", "trust_policy", "trust_management", "hard", "idn-paper.tex:382-395", prompt, answer, {"case": case}))
    return out


PROMPT_STYLE_VARIANTS = [
    ("Return JSON", "Return JSON"),
    ("Return JSON", "Produce a valid JSON object"),
    ("Return JSON", "Respond with one valid JSON object"),
    ("Return JSON", "Output a machine-readable JSON object"),
]


def make_dataset(repeats: int, seed: int) -> list[BenchmarkExample]:
    random.seed(seed)
    base: list[BenchmarkExample] = []
    start = 0
    builders = [capability_descriptor_examples, route_examples, fallback_examples, resource_fit_examples, cache_examples, trust_examples]
    for builder in builders:
        chunk = builder(start)
        base.extend(chunk)
        start += len(chunk)

    out = []
    for r in range(repeats):
        old_phrase, new_phrase = PROMPT_STYLE_VARIANTS[r % len(PROMPT_STYLE_VARIANTS)]
        for ex in base:
            payload = asdict(ex)
            payload["id"] = f"{ex.id}-v{r+1:02d}"
            if r > 0:
                payload["prompt"] = payload["prompt"].replace(old_phrase, new_phrase)
                payload["metadata"] = {**payload["metadata"], "construction_variant": r, "prompt_style": new_phrase}
            else:
                payload["metadata"] = {**payload["metadata"], "construction_variant": r, "prompt_style": "Return JSON"}
            out.append(BenchmarkExample(**payload))
    random.shuffle(out)
    return out


def write_jsonl(path: Path, examples: list[BenchmarkExample]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an IDN routing/capability benchmark dataset.")
    parser.add_argument("--output", default="data/benchmarks/idn_benchmark_v1.jsonl")
    parser.add_argument("--train-output", default="data/benchmarks/idn_benchmark_v1_train.jsonl")
    parser.add_argument("--eval-output", default="data/benchmarks/idn_benchmark_v1_eval.jsonl")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    examples = make_dataset(args.repeats, args.seed)
    split = int(len(examples) * 0.75)
    write_jsonl(Path(args.output), examples)
    write_jsonl(Path(args.train_output), examples[:split])
    write_jsonl(Path(args.eval_output), examples[split:])

    stats: dict[str, int] = {}
    for ex in examples:
        stats[ex.task_type] = stats.get(ex.task_type, 0) + 1
    print(json.dumps({"total": len(examples), "train": split, "eval": len(examples) - split, "task_counts": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
