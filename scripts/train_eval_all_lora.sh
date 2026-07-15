#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=src
PYTHON=python
MACHINE=h100
MODEL=Qwen2.5-1.5B-Instruct

$PYTHON -m idn.cli --machine $MACHINE train-lora \
    --config configs/capabilities/cnn_dailymail.yaml \
    --model $MODEL --rank 16 \
    --output artifacts/adapters/cnn_dailymail/r16

$PYTHON -m idn.cli --machine $MACHINE train-lora \
    --config configs/capabilities/cola.yaml \
    --model $MODEL --rank 16 \
    --output artifacts/adapters/cola/r16

$PYTHON -m idn.cli --machine $MACHINE train-lora \
    --config configs/capabilities/gsm8k.yaml \
    --model $MODEL --rank 16 \
    --output artifacts/adapters/gsm8k/r16

$PYTHON -m idn.cli --machine $MACHINE train-lora \
    --config configs/capabilities/mbpp.yaml \
    --model $MODEL --rank 16 \
    --output artifacts/adapters/mbpp/r16

$PYTHON -m idn.cli --machine $MACHINE train-lora \
    --config configs/capabilities/squad.yaml \
    --model $MODEL --rank 16 \
    --output artifacts/adapters/squad/r16

$PYTHON -m idn.cli --machine $MACHINE train-lora \
    --config configs/capabilities/sst2.yaml \
    --model $MODEL --rank 16 \
    --output artifacts/adapters/sst2/r16

mkdir -p artifacts/eval/lora_all

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/cnn_dailymail.yaml \
    --model $MODEL --adapter artifacts/adapters/cnn_dailymail/r16 \
    --output artifacts/eval/lora_all/cnn_dailymail.json

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/cola.yaml \
    --model $MODEL --adapter artifacts/adapters/cola/r16 \
    --output artifacts/eval/lora_all/cola.json

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/gsm8k.yaml \
    --model $MODEL --adapter artifacts/adapters/gsm8k/r16 \
    --output artifacts/eval/lora_all/gsm8k.json

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/mbpp.yaml \
    --model $MODEL --adapter artifacts/adapters/mbpp/r16 \
    --output artifacts/eval/lora_all/mbpp.json

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/squad.yaml \
    --model $MODEL --adapter artifacts/adapters/squad/r16 \
    --output artifacts/eval/lora_all/squad.json

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/sst2.yaml \
    --model $MODEL --adapter artifacts/adapters/sst2/r16 \
    --output artifacts/eval/lora_all/sst2.json
