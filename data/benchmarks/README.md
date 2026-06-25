# IDN Benchmark V1

Generated: 2026-06-25

This dataset is a small IDN-specific benchmark for comparing small and large models before routing or capability-distribution decisions. It is grounded in the local IDN paper and implementation plan:

- `idn-paper.tex`
- `capability_extraction_implementation_plan.md`
- `局部能力提取方案_PPT大纲.md`

## Files

- `idn_benchmark_v1.jsonl` — full dataset, 192 examples
- `idn_benchmark_v1_train.jsonl` — train/dev split, 144 examples
- `idn_benchmark_v1_eval.jsonl` — held-out eval split, 48 examples

## Schema

Each line is a JSON object:

```json
{
  "id": "idn-route-0012-v03",
  "task_type": "route_decision",
  "capability": "json_extraction",
  "difficulty": "medium",
  "source": "idn-paper.tex:314-343",
  "prompt": "...",
  "answer": { "route": "regional", "fallback_required": true, "reason": "..." },
  "metadata": { "latency_target_ms": 800, "privacy": "tenant" }
}
```

## Task types

The dataset has 6 task types, each with 32 examples:

1. `capability_descriptor_extraction`
   - Extract deployable capability descriptors.
   - Tests schema following and IDN abstraction understanding.

2. `route_decision`
   - Choose `local`, `edge`, `regional`, or `cloud` for a request.
   - Tests latency/privacy/quality/resource tradeoff reasoning.

3. `fallback_decision`
   - Decide whether to accept local output or fall back to teacher.
   - Tests confidence gate, verifier, schema validity, trust, and state compatibility reasoning.

4. `resource_fit`
   - Judge whether a resource profile can serve a capability.
   - Tests matching capability descriptors to node resources.

5. `cache_decision`
   - Choose cache action among `cache`, `reuse_local`, `recompute`, `reject_cache`.
   - Tests state-aware caching and privacy constraints.

6. `trust_policy`
   - Decide whether a capability/state/execution case satisfies trust policy.
   - Tests signature, lineage, attestation, encryption, and audit evidence reasoning.

## Why this dataset fits model distribution

IDN routing cares about whether a small edge model can make reliable decisions under explicit constraints. Generic datasets such as CoLA or GSM8K test isolated language or math skills. This dataset directly tests the control-plane decisions IDN needs:

- What capability is requested?
- Which node tier should serve it?
- Is fallback required?
- Does the resource profile fit?
- Should state be cached or reused?
- Is the trust evidence sufficient?

## Current size

```json
{
  "total": 192,
  "train": 144,
  "eval": 48,
  "task_counts": {
    "capability_descriptor_extraction": 32,
    "route_decision": 32,
    "fallback_decision": 32,
    "resource_fit": 32,
    "cache_decision": 32,
    "trust_policy": 32
  }
}
```

## Regenerate

```bash
cd /home/sig/files/IDN
/home/sig/anaconda3/bin/python src/idn/build_idn_benchmark.py \
  --repeats 4 \
  --seed 7 \
  --output data/benchmarks/idn_benchmark_v1.jsonl \
  --train-output data/benchmarks/idn_benchmark_v1_train.jsonl \
  --eval-output data/benchmarks/idn_benchmark_v1_eval.jsonl
```

## Evaluation recommendation

For small-vs-large model comparison, score at two levels:

- Strict JSON validity: output must parse as JSON.
- Field accuracy: compare key fields such as `route`, `fallback_required`, `feasible`, `action`, `allowed`.

The `reason` field should be evaluated separately with semantic similarity or LLM judge because many valid explanations can differ in wording.
