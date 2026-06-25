# IDN Capability Extraction Prototype

This repo implements the first slice of IDN capability extraction: deploy a cloud teacher model and an edge/base model, run both on a dataset, and collect latency/quality-gap records for later region discovery and LoRA extraction.

## First test pair

- Teacher: `Qwen/Qwen2.5-7B-Instruct`
- Edge/base: `Qwen/Qwen2.5-1.5B-Instruct`

## Local smoke test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
idn-cap eval-models --config configs/qwen25_eval.yaml --num-samples 2
```

## A100 test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[gpu]'
nvidia-smi
idn-cap eval-models --config configs/qwen25_eval.yaml --num-samples 20
```

Outputs:

- `artifacts/eval/qwen25_7b_teacher_vs_15b_base_alpaca20.jsonl`
- `artifacts/eval/qwen25_7b_teacher_vs_15b_base_alpaca20_summary.json`
