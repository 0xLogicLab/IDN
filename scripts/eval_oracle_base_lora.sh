#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=src
PYTHON=python
MACHINE=3090
DATASET=artifacts/eval/idn_mixed_1000.jsonl

$PYTHON scripts/eval_mixed_oracle.py \
    --mode base --machine $MACHINE --dataset $DATASET \
    --output artifacts/eval/idn_mixed_oracle/base.json

$PYTHON scripts/eval_mixed_oracle.py \
    --mode lora --machine $MACHINE --dataset $DATASET \
    --adapter-root artifacts/adapters --rank 16 \
    --output artifacts/eval/idn_mixed_oracle/oracle_lora.json
