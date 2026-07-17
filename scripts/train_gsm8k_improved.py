#!/usr/bin/env python3
"""Train a conservative, base-preserving GSM8K LoRA specialist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idn.config import load_machine_config, resolve_relative_dir
from idn.registry import load_registry
from idn.training.train_lora import TrainLoRAError, train_lora


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a conservative GSM8K LoRA with base-logit preservation."
    )
    parser.add_argument("--machine", default="3090")
    parser.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output", default="artifacts/adapters/gsm8k/r16_preserved")
    parser.add_argument("--answer-loss-weight", type=float, default=1.0)
    parser.add_argument("--base-kl-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--override", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    machine = load_machine_config(args.machine)
    model_path = Path(
        resolve_relative_dir(
            base_dir=args.model_dir or machine.model_dir,
            relative_dir=args.model,
            field="--model",
        )
    )
    output = Path(args.output)
    if output.exists() and not args.override:
        print("GSM8K improved adapter already exists; skipping training.")
        print("adapter: {0}".format(output))
        return 0
    registry = load_registry(
        config_dir=Path("configs/capabilities"), machine_path=args.machine
    )
    registration = registry["gsm8k"]
    if registration.spec is None:
        raise SystemExit("GSM8K capability is not implemented")
    batch_size = args.batch_size or machine.train_batch_size
    try:
        result = train_lora(
            base_model_path=model_path,
            spec=registration.spec,
            split="train",
            output_dir=output,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            learning_rate=args.learning_rate,
            scheduler="cosine",
            warmup_ratio=0.03,
            epochs=args.epochs,
            max_length=args.max_length,
            batch_size=batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=1.0,
            dtype=args.dtype,
            device=machine.device,
            local_files_only=machine.local_files_only,
            seed=args.seed,
            log_steps=args.log_steps,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            answer_loss_weight=args.answer_loss_weight,
            base_kl_weight=args.base_kl_weight,
            override=args.override,
            progress=print,
        )
    except TrainLoRAError as exc:
        raise SystemExit("GSM8K training error: {0}".format(exc)) from exc
    print("GSM8K improved adapter complete.")
    print("adapter: {0}".format(result.adapter_dir))
    print("train_meta: {0}".format(output / "train_meta.json"))
    print("training_log: {0}".format(output / "training_log.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
