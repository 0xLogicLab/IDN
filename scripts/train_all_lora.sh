#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=src
PYTHON=python
MACHINE=h100
MODEL=Qwen2.5-1.5B-Instruct

$PYTHON -m idn.cli --machine $MACHINE train-lora \
    --config configs/capabilities/cnn_dailymail.yaml \
    --model $MODEL --rank 16 \
    --output artifacts/adapters/cnn_dailymail/r16 \
    --batch-size 16 --limit 10000

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
    --output artifacts/adapters/squad/r16 --limit 10000

$PYTHON -m idn.cli --machine $MACHINE train-lora \
    --config configs/capabilities/sst2.yaml \
    --model $MODEL --rank 16 \
    --output artifacts/adapters/sst2/r16
