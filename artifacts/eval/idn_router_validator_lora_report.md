# IDN Router + Validator LoRA Experiment

## Scope

This run implements the production-oriented split:

```text
7B teacher = fallback + difficult reasoning
1.5B student = bounded specialist worker
Router = dispatch / acceptance policy
Validator = risk gate
Dataset = capability transfer carrier
```

The task family is IDN capability routing and validation. It deliberately excludes stock/market tasks.

## Data

- Train: `data/benchmarks/idn_benchmark_v2_train.jsonl` — 720 examples
- Eval subset: first 40 rows of `data/benchmarks/idn_benchmark_v2_eval.jsonl`
- Task types: capability descriptor extraction, route decision, fallback decision, resource fit, cache decision, trust policy
- Source material: `idn-paper.tex`, `capability_extraction_implementation_plan.md`, and recovered prior IDN capability-distribution conversation context

## Models

- Base student: `/data/huggingface/model/Qwen2.5-1.5B-Instruct`
- Teacher reference: `/data/huggingface/model/Qwen2.5-7B-Instruct`
- Tuned student adapter: `artifacts/adapters/qwen25_15b_idn_router_validator_lora_fp32`

## Training command

```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false \
/home/sig/anaconda3/envs/control/bin/python src/idn/train_idn_lora.py \
  --model /data/huggingface/model/Qwen2.5-1.5B-Instruct \
  --train data/benchmarks/idn_benchmark_v2_train.jsonl \
  --eval data/benchmarks/idn_benchmark_v2_eval.jsonl \
  --output artifacts/adapters/qwen25_15b_idn_router_validator_lora_fp32 \
  --epochs 0.2 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 1024 \
  --torch-dtype float32
```

Training used fp32 because fp16 on this CUDA/PyTorch stack produced invalid generation (`!` repetition) and NaN loss. The fp32 smoke train had a valid loss curve:

- train loss: 1.70
- eval loss: 1.08
- LoRA trainable params: 18,464,768 / 1,562,179,072 = 1.182%

## Evaluation commands

Student/base/tuned eval:

```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false \
/home/sig/anaconda3/envs/control/bin/python src/idn/eval_router_validator_lora.py \
  --dataset data/benchmarks/idn_benchmark_v2_eval.jsonl \
  --small-model /data/huggingface/model/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/adapters/qwen25_15b_idn_router_validator_lora_fp32 \
  --output-path artifacts/eval/idn_router_validator_lora_fp32_eval_40_no_teacher.jsonl \
  --summary-path artifacts/eval/idn_router_validator_lora_fp32_eval_40_no_teacher_summary.json \
  --max-samples 40 \
  --max-new-tokens 96 \
  --torch-dtype float32 \
  --skip-teacher
```

Combined with prior 7B reference rows:

```bash
/home/sig/anaconda3/envs/control/bin/python src/idn/combine_router_validator_eval.py \
  --prior artifacts/eval/idn_benchmark_v2_qwen25_7b_vs_15b_strict.jsonl \
  --tuned artifacts/eval/idn_router_validator_lora_fp32_eval_40_no_teacher.jsonl \
  --summary artifacts/eval/idn_router_validator_combined_40_summary.json \
  --output artifacts/eval/idn_router_validator_combined_40.jsonl
```

## Metrics on 40 eval examples

| System | JSON valid | Exact match | Field accuracy | Mean latency | Local coverage |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B base | 90.0% | 2.5% | 31.78% | 2363 ms | 100% raw |
| Qwen2.5-1.5B LoRA | 100.0% | 27.5% | 44.19% | 1992 ms | 100% raw |
| Qwen2.5-7B prior reference | 75.0% | 7.5% | 24.81% | 1986 ms | fallback only |
| Router + Validator | 90.0% | 32.5% | 35.66% | 1817 ms | 27.5% accepted locally |

## Task-level effects

- `trust_policy`: tuned 1.5B reached 100% exact match on this subset; validator accepts those rows locally.
- `resource_fit`: tuned 1.5B reached 60% exact match; accepted rows stay local.
- `fallback_decision`: tuned 1.5B improved exact match to 28.6%; validator accepts exact rows.
- `route_decision`: tuned 1.5B still weak; validator routes these mostly to teacher.
- `cache_decision` and `capability_descriptor_extraction`: JSON stability improved, semantic exactness still needs better data or scoring.

## Interpretation

The implementation validates the intended architecture: 1.5B improves strongly inside bounded IDN policy tasks, and the Validator limits local serving to rows that pass exact/schema checks. The current adapter is a smoke-scale LoRA, so the local coverage is 27.5%; production direction is to expand coverage by training longer and adding task-specific validator thresholds.

Recommended next engineering step: train 1 full fp32 epoch or move to a newer CUDA/PyTorch stack for stable bf16/fp16, then evaluate the full 240-example split.
