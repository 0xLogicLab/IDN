#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=src
PYTHON=python
MACHINE=h100
MODEL=Qwen2.5-1.5B-Instruct

mkdir -p artifacts/eval/lora_all_limit1000

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/cnn_dailymail.yaml \
    --model $MODEL --adapter artifacts/adapters/cnn_dailymail/r16 \
    --output artifacts/eval/lora_all_limit1000/cnn_dailymail.json --limit 1000 --verbose

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/cola.yaml \
    --model $MODEL --adapter artifacts/adapters/cola/r16 \
    --output artifacts/eval/lora_all_limit1000/cola.json --limit 1000 --verbose

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/gsm8k.yaml \
    --model $MODEL --adapter artifacts/adapters/gsm8k/r16 \
    --output artifacts/eval/lora_all_limit1000/gsm8k.json --limit 1000 --verbose

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/mbpp.yaml \
    --model $MODEL --adapter artifacts/adapters/mbpp/r16 \
    --output artifacts/eval/lora_all_limit1000/mbpp.json --limit 1000 --verbose

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/squad.yaml \
    --model $MODEL --adapter artifacts/adapters/squad/r16 \
    --output artifacts/eval/lora_all_limit1000/squad.json --limit 1000 --verbose

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/sst2.yaml \
    --model $MODEL --adapter artifacts/adapters/sst2/r16 \
    --output artifacts/eval/lora_all_limit1000/sst2.json --limit 1000 --verbose