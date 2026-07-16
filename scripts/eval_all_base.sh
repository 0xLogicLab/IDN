#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=src
PYTHON=python
MACHINE=h100
MODEL=Qwen2.5-7B-Instruct
dir=base_all_limit1000_7B

mkdir -p artifacts/eval/$dir

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/cnn_dailymail.yaml \
    --model $MODEL \
    --output artifacts/eval/$dir/cnn_dailymail.json --limit 1000 --verbose &

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/cola.yaml \
    --model $MODEL \
    --output artifacts/eval/$dir/cola.json --limit 1000 --verbose &

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/gsm8k.yaml \
    --model $MODEL \
    --output artifacts/eval/$dir/gsm8k.json --limit 1000 --verbose &

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/mbpp.yaml \
    --model $MODEL \
    --output artifacts/eval/$dir/mbpp.json --limit 1000 --verbose &

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/squad.yaml \
    --model $MODEL \
    --output artifacts/eval/$dir/squad.json --limit 1000 --verbose &

$PYTHON -m idn.cli --machine $MACHINE eval \
    --config configs/capabilities/sst2.yaml \
    --model $MODEL \
    --output artifacts/eval/$dir/sst2.json --limit 1000 --verbose &