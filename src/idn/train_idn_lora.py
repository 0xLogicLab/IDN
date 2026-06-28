from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_messages(tokenizer: Any, prompt: str, answer: dict[str, Any]) -> str:
    answer_text = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    wrapped = (
        "You are evaluating an IDN model capability benchmark.\n"
        "Return exactly one valid JSON object.\n"
        "Use double-quoted JSON keys and values.\n"
        "Do not use markdown fences, comments, explanations, or trailing commas.\n\n"
        f"{prompt}\n\n"
        "Final answer: one valid JSON object only."
    )
    messages = [
        {"role": "system", "content": "You are a strict JSON API. Output only valid JSON and no prose."},
        {"role": "user", "content": wrapped},
        {"role": "assistant", "content": answer_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


class JsonSftDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.examples = []
        for row in rows:
            text = format_messages(tokenizer, row["prompt"], row["answer"])
            encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False)
            encoded["labels"] = list(encoded["input_ids"])
            self.examples.append(encoded)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {key: list(value) for key, value in self.examples[idx].items()}


class CausalCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padded = label + [-100] * (max_len - len(label))
            padded_labels.append(padded)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-1.5B on IDN Router/Validator JSON tasks.")
    parser.add_argument("--model", default="/data/huggingface/model/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train", default="data/benchmarks/idn_benchmark_v2_train.jsonl")
    parser.add_argument("--eval", default="data/benchmarks/idn_benchmark_v2_eval.jsonl")
    parser.add_argument("--output", default="artifacts/adapters/qwen25_15b_idn_router_validator_lora")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--torch-dtype", default="float32")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(args.train)
    eval_rows = read_jsonl(args.eval)
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]

    train_dataset = JsonSftDataset(train_rows, tokenizer, args.max_length)
    eval_dataset = JsonSftDataset(eval_rows[: min(64, len(eval_rows))], tokenizer, args.max_length)

    dtype_map = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map.get(args.torch_dtype.lower(), torch.float32),
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        fp16=args.torch_dtype.lower() in {"float16", "fp16"},
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    metadata = {
        "base_model": args.model,
        "train_dataset": args.train,
        "eval_dataset": args.eval,
        "train_examples": len(train_rows),
        "eval_examples_for_loss": len(eval_dataset),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with Path(args.output, "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
