"""Gold-supervised LoRA specialization for capability packages."""

from __future__ import annotations

import importlib.metadata
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from idn.capabilities.base import CapabilitySpec, Example
from idn.config import ConfigError, load_yaml
from idn.extraction.carve import hash_model_directory
from idn.extraction.heal import render_prompt, write_json_atomic
from idn.models.loader import torch_dtype
from idn.models.qwen2_local.loader import load_qwen2_for_causal_lm


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
TARGET_MODULE_ALIASES = {
    "qkvo": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "all": DEFAULT_TARGET_MODULES,
    "all_linear": DEFAULT_TARGET_MODULES,
}


class TrainLoRAError(RuntimeError):
    """Raised when specialist LoRA training cannot proceed."""


@dataclass(frozen=True)
class LoRATrainingDefaults:
    dtype: str = "bfloat16"
    rank: int = 16
    alpha: Optional[int] = None
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = DEFAULT_TARGET_MODULES
    learning_rate: float = 2.0e-4
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    epochs: int = 2
    max_length: int = 1024
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    seed: int = 13
    log_steps: int = 10
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class TrainLoRAResult:
    adapter_dir: Path
    metadata: Mapping[str, Any]
    training_log: Sequence[Mapping[str, Any]]


def load_lora_training_defaults(path: Path) -> LoRATrainingDefaults:
    """Load and validate the ``training`` section of a LoRA YAML file."""

    raw = load_yaml(path)
    training = raw.get("training")
    if not isinstance(training, Mapping):
        raise ConfigError("Missing mapping 'training' in {0}".format(path))
    lora = training.get("lora", {})
    if not isinstance(lora, Mapping):
        raise ConfigError("training.lora must be a mapping in {0}".format(path))
    defaults = LoRATrainingDefaults(
        dtype=str(training.get("dtype", "bfloat16")),
        rank=int(lora.get("r", 16)),
        alpha=_optional_int(lora.get("alpha"), "training.lora.alpha"),
        dropout=float(lora.get("dropout", 0.05)),
        target_modules=normalize_target_modules(
            lora.get("target_modules", DEFAULT_TARGET_MODULES)
        ),
        learning_rate=float(training.get("lr", 2.0e-4)),
        scheduler=str(training.get("scheduler", "cosine")),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        epochs=int(training.get("epochs", 2)),
        max_length=int(training.get("max_length", 1024)),
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 4)
        ),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        seed=int(training.get("seed", 13)),
        log_steps=int(training.get("log_steps", 10)),
        gradient_checkpointing=bool(
            training.get("gradient_checkpointing", True)
        ),
    )
    _validate_training_values(
        rank=defaults.rank,
        dropout=defaults.dropout,
        learning_rate=defaults.learning_rate,
        epochs=defaults.epochs,
        max_length=defaults.max_length,
        batch_size=1,
        gradient_accumulation_steps=defaults.gradient_accumulation_steps,
        warmup_ratio=defaults.warmup_ratio,
        max_grad_norm=defaults.max_grad_norm,
        log_steps=defaults.log_steps,
        scheduler=defaults.scheduler,
        target_modules=defaults.target_modules,
        max_steps=None,
    )
    return defaults


def normalize_target_modules(value: Any) -> Tuple[str, ...]:
    """Normalize YAML lists, CSV strings, and legacy target-module aliases."""

    if isinstance(value, str):
        alias = TARGET_MODULE_ALIASES.get(value.strip().lower())
        if alias is not None:
            return tuple(alias)
        values = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        values = [str(item).strip() for item in value]
    else:
        raise ConfigError("LoRA target_modules must be a list or comma-separated string")
    modules = tuple(item for item in values if item)
    if not modules:
        raise ConfigError("LoRA target_modules must contain at least one module")
    return modules


def encode_training_example(
    *,
    tokenizer: Any,
    spec: CapabilitySpec,
    example: Example,
    max_length: int,
    answer_loss_weight: float = 1.0,
) -> Mapping[str, List[Any]]:
    """Encode one chat example while masking prompt tokens from SFT loss."""

    prompt = render_prompt(tokenizer, spec.format_prompt(example))
    target = spec.target_text(example)
    if not target or not target.strip():
        raise TrainLoRAError("example {0} has an empty target".format(example.id))
    prompt_ids = list(
        tokenizer(prompt, add_special_tokens=False)["input_ids"]
    )
    target_text = target.strip()
    target_ids = list(
        tokenizer(target_text, add_special_tokens=False)["input_ids"]
    )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and (
        not target_ids or target_ids[-1] != int(eos_token_id)
    ):
        target_ids.append(int(eos_token_id))
    if not target_ids:
        raise TrainLoRAError("example {0} has no target tokens".format(example.id))

    if answer_loss_weight <= 0.0:
        raise TrainLoRAError("answer_loss_weight must be positive")
    target_weights = [1.0] * len(target_ids)
    answer_marker = target_text.rfind("####")
    if answer_loss_weight != 1.0 and answer_marker >= 0:
        # Tokenize the prefix to locate the marker in the full target sequence.
        # Include the token immediately before the marker because BPE tokenization
        # can merge whitespace with the marker itself.
        prefix_length = len(
            tokenizer(target_text[:answer_marker], add_special_tokens=False)[
                "input_ids"
            ]
        )
        answer_start = max(0, prefix_length - 1)
        target_weights[answer_start:] = [answer_loss_weight] * (
            len(target_weights) - answer_start
        )

    if len(target_ids) >= max_length:
        target_ids = target_ids[-max_length:]
        target_weights = target_weights[-max_length:]
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[-(max_length - len(target_ids)) :]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    loss_weights = [0.0] * len(prompt_ids) + target_weights
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "loss_weights": loss_weights,
    }


def collate_training_examples(
    records: Sequence[Mapping[str, Sequence[Any]]],
    pad_token_id: int,
) -> Mapping[str, Any]:
    """Right-pad variable-length causal-LM records into PyTorch tensors."""

    if not records:
        raise TrainLoRAError("cannot collate an empty training batch")
    import torch

    width = max(len(record["input_ids"]) for record in records)
    input_rows = []
    mask_rows = []
    label_rows = []
    weight_rows = []
    for record in records:
        padding = width - len(record["input_ids"])
        input_rows.append(list(record["input_ids"]) + [pad_token_id] * padding)
        mask_rows.append(list(record["attention_mask"]) + [0] * padding)
        label_rows.append(list(record["labels"]) + [-100] * padding)
        weights = list(record.get("loss_weights", []))
        if not weights:
            weights = [1.0 if label != -100 else 0.0 for label in record["labels"]]
        weight_rows.append(weights + [0.0] * padding)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "loss_weights": torch.tensor(weight_rows, dtype=torch.float32),
    }


def weighted_causal_lm_loss(
    *,
    logits: Any,
    labels: Any,
    loss_weights: Any,
) -> Any:
    """Compute mean next-token CE with per-label weights.

    Labels equal to -100 remain ignored. The weights are aligned with labels,
    including the causal one-token shift.
    """

    import torch.nn.functional as functional

    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = loss_weights[..., 1:].to(dtype=shift_logits.dtype).contiguous()
    token_loss = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    valid_weights = shift_weights.view(-1) * (shift_labels.view(-1) != -100)
    denominator = valid_weights.sum().clamp_min(1.0)
    return (token_loss * valid_weights).sum() / denominator


def base_preservation_kl_loss(
    *,
    student_logits: Any,
    base_logits: Any,
    labels: Any,
    temperature: float = 1.0,
) -> Any:
    """KL regularization on target positions against the adapter-disabled base."""

    import torch.nn.functional as functional

    student = student_logits[..., :-1, :].float() / temperature
    teacher = base_logits[..., :-1, :].float() / temperature
    valid = (labels[..., 1:] != -100).to(dtype=student.dtype)
    token_kl = functional.kl_div(
        functional.log_softmax(student, dim=-1),
        functional.softmax(teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return (token_kl * valid).sum() * (temperature**2) / valid.sum().clamp_min(1.0)


def train_lora(
    *,
    base_model_path: Path,
    spec: CapabilitySpec,
    split: str,
    output_dir: Path,
    limit: Optional[int] = None,
    rank: int = 16,
    alpha: Optional[int] = None,
    dropout: float = 0.05,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    learning_rate: float = 2.0e-4,
    scheduler: str = "cosine",
    warmup_ratio: float = 0.03,
    epochs: int = 2,
    max_steps: Optional[int] = None,
    max_length: int = 1024,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    dtype: str = "bfloat16",
    device: str = "cuda",
    local_files_only: bool = True,
    seed: int = 13,
    log_steps: int = 10,
    gradient_checkpointing: bool = True,
    answer_loss_weight: float = 1.0,
    base_kl_weight: float = 0.0,
    override: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> TrainLoRAResult:
    """Fine-tune a LoRA specialist on gold capability targets."""

    target_modules = tuple(target_modules)
    if limit is not None and limit <= 0:
        raise TrainLoRAError("limit must be positive or None")
    _validate_training_values(
        rank=rank,
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=epochs,
        max_length=max_length,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        log_steps=log_steps,
        scheduler=scheduler,
        target_modules=target_modules,
        max_steps=max_steps,
    )
    if answer_loss_weight <= 0.0:
        raise TrainLoRAError("answer_loss_weight must be positive")
    if base_kl_weight < 0.0:
        raise TrainLoRAError("base_kl_weight must be non-negative")
    base_model_path = Path(base_model_path)
    output_dir = Path(output_dir)
    _validate_model_dir(base_model_path)
    _prepare_output_dir(output_dir, override=override)

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TrainLoRAError(
            "torch, transformers, and peft are required for LoRA training"
        ) from exc

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_device = _resolve_device(torch, device)
    resolved_dtype = torch_dtype(dtype)
    if target_device.type == "cpu" and resolved_dtype == torch.bfloat16:
        resolved_dtype = torch.float32

    _report(progress, "hashing base model")
    base_model_hash = hash_model_directory(base_model_path)
    if tokenizer is None:
        _report(progress, "loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_model_path),
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is None:
            raise TrainLoRAError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if model is None:
        _report(progress, "loading frozen base model")
        model = load_qwen2_for_causal_lm(
            str(base_model_path),
            torch_dtype=resolved_dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
            device_map={"": str(target_device)},
        )
    else:
        model = model.to(target_device)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    resolved_alpha = int(alpha if alpha is not None else 2 * rank)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=resolved_alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    try:
        model = get_peft_model(model, lora_config)
    except (KeyError, ValueError) as exc:
        raise TrainLoRAError(
            "could not attach LoRA to target modules {0}: {1}".format(
                ",".join(target_modules), exc
            )
        ) from exc
    model.train()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    if not trainable_parameters:
        raise TrainLoRAError("LoRA did not create any trainable parameters")
    _report(
        progress,
        "trainable parameters: {0}/{1}".format(trainable_count, total_count),
    )

    _report(progress, "loading {0} examples".format(split))
    examples = spec.load_split(split=split, limit=limit)
    if not examples:
        raise TrainLoRAError("no examples loaded for split: {0}".format(split))
    batches_per_epoch = int(math.ceil(len(examples) / float(batch_size)))
    steps_per_epoch = int(
        math.ceil(batches_per_epoch / float(gradient_accumulation_steps))
    )
    planned_steps = epochs * steps_per_epoch
    total_steps = min(planned_steps, max_steps) if max_steps is not None else planned_steps
    if total_steps <= 0:
        raise TrainLoRAError("training resolved to zero optimizer steps")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_cosine_schedule(total_steps, warmup_ratio),
    )
    started_at = _utc_now()
    metadata = _build_metadata(
        status="running",
        spec=spec,
        split=split,
        examples=len(examples),
        base_model_path=base_model_path,
        base_model_hash=base_model_hash,
        rank=rank,
        alpha=resolved_alpha,
        dropout=dropout,
        target_modules=target_modules,
        learning_rate=learning_rate,
        scheduler=scheduler,
        warmup_ratio=warmup_ratio,
        epochs=epochs,
        planned_steps=total_steps,
        completed_steps=0,
        max_length=max_length,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        dtype=dtype,
        device=str(target_device),
        seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        trainable_count=trainable_count,
        total_count=total_count,
        started_at=started_at,
        final_loss=None,
        answer_loss_weight=answer_loss_weight,
        base_kl_weight=base_kl_weight,
    )
    write_json_atomic(output_dir / "train_meta.json", metadata)
    write_json_atomic(output_dir / "training_log.json", {"steps": []})

    training_log: List[Mapping[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    completed_steps = 0
    pad_token_id = int(tokenizer.pad_token_id)
    for epoch in range(1, epochs + 1):
        order = list(range(len(examples)))
        random.Random(seed + epoch - 1).shuffle(order)
        cursor = 0
        while cursor < len(order) and completed_steps < total_steps:
            micro_batches = []
            for _ in range(gradient_accumulation_steps):
                indices = order[cursor : cursor + batch_size]
                cursor += len(indices)
                if not indices:
                    break
                micro_batches.append(indices)
            accumulated_loss = 0.0
            accumulated_sft_loss = 0.0
            accumulated_kl_loss = 0.0
            for indices in micro_batches:
                records = [
                    encode_training_example(
                        tokenizer=tokenizer,
                        spec=spec,
                        example=examples[index],
                        max_length=max_length,
                        answer_loss_weight=answer_loss_weight,
                    )
                    for index in indices
                ]
                batch = collate_training_examples(records, pad_token_id=pad_token_id)
                batch = {
                    key: value.to(target_device) for key, value in batch.items()
                }
                with _autocast(torch, target_device, resolved_dtype):
                    loss_weights = batch.pop("loss_weights")
                    base_logits = None
                    if base_kl_weight > 0.0:
                        with torch.no_grad(), model.disable_adapter():
                            base_output = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                use_cache=False,
                            )
                            base_logits = base_output.logits
                    output = model(**batch, use_cache=False)
                    if output.loss is None or not torch.isfinite(output.loss):
                        raise TrainLoRAError("training produced a non-finite loss")
                    if answer_loss_weight == 1.0:
                        sft_loss = output.loss
                    else:
                        sft_loss = weighted_causal_lm_loss(
                            logits=output.logits,
                            labels=batch["labels"],
                            loss_weights=loss_weights,
                        )
                        if not torch.isfinite(sft_loss):
                            raise TrainLoRAError("weighted training produced a non-finite loss")
                    if base_logits is None:
                        kl_loss = sft_loss.new_zeros(())
                    else:
                        kl_loss = base_preservation_kl_loss(
                            student_logits=output.logits,
                            base_logits=base_logits,
                            labels=batch["labels"],
                        )
                    loss = sft_loss + base_kl_weight * kl_loss
                    loss = loss / len(micro_batches)
                loss.backward()
                accumulated_loss += float(loss.detach().cpu()) * len(micro_batches)
                accumulated_sft_loss += float(sft_loss.detach().cpu())
                accumulated_kl_loss += float(kl_loss.detach().cpu())

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, max_grad_norm
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
            entry = {
                "step": completed_steps,
                "epoch": epoch,
                "loss": accumulated_loss / len(micro_batches),
                "sft_loss": accumulated_sft_loss / len(micro_batches),
                "kl_loss": accumulated_kl_loss / len(micro_batches),
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rate": float(lr_scheduler.get_last_lr()[0]),
            }
            training_log.append(entry)
            if completed_steps == 1 or completed_steps % log_steps == 0:
                _report(
                    progress,
                    "step {0}/{1} epoch={2} loss={3:.6f}".format(
                        completed_steps, total_steps, epoch, entry["loss"]
                    ),
                )
                write_json_atomic(
                    output_dir / "training_log.json", {"steps": training_log}
                )
        if completed_steps >= total_steps:
            break

    _report(progress, "saving LoRA adapter: {0}".format(output_dir))
    model.save_pretrained(str(output_dir), safe_serialization=True)
    final_loss = float(training_log[-1]["loss"])
    metadata = _build_metadata(
        status="complete",
        spec=spec,
        split=split,
        examples=len(examples),
        base_model_path=base_model_path,
        base_model_hash=base_model_hash,
        rank=rank,
        alpha=resolved_alpha,
        dropout=dropout,
        target_modules=target_modules,
        learning_rate=learning_rate,
        scheduler=scheduler,
        warmup_ratio=warmup_ratio,
        epochs=epochs,
        planned_steps=total_steps,
        completed_steps=completed_steps,
        max_length=max_length,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        dtype=dtype,
        device=str(target_device),
        seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        trainable_count=trainable_count,
        total_count=total_count,
        started_at=started_at,
        final_loss=final_loss,
        answer_loss_weight=answer_loss_weight,
        base_kl_weight=base_kl_weight,
    )
    write_json_atomic(output_dir / "training_log.json", {"steps": training_log})
    write_json_atomic(output_dir / "train_meta.json", metadata)
    return TrainLoRAResult(
        adapter_dir=output_dir,
        metadata=metadata,
        training_log=tuple(training_log),
    )


def _build_metadata(
    *,
    status: str,
    spec: CapabilitySpec,
    split: str,
    examples: int,
    base_model_path: Path,
    base_model_hash: str,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
    learning_rate: float,
    scheduler: str,
    warmup_ratio: float,
    epochs: int,
    planned_steps: int,
    completed_steps: int,
    max_length: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    dtype: str,
    device: str,
    seed: int,
    gradient_checkpointing: bool,
    trainable_count: int,
    total_count: int,
    started_at: str,
    final_loss: Optional[float],
    answer_loss_weight: float = 1.0,
    base_kl_weight: float = 0.0,
) -> Mapping[str, Any]:
    return {
        "artifact_version": 1,
        "status": status,
        "method": "gold_supervised_lora_sft",
        "capability": spec.id,
        "data": {
            "split": split,
            "examples": examples,
            "target_source": "gold",
        },
        "lineage": {
            "base_model_path": str(base_model_path),
            "base_model_hash": base_model_hash,
            "started_at": started_at,
            "updated_at": _utc_now(),
        },
        "lora": {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": list(target_modules),
        },
        "optimization": {
            "learning_rate": learning_rate,
            "scheduler": scheduler,
            "warmup_ratio": warmup_ratio,
            "epochs": epochs,
            "planned_steps": planned_steps,
            "completed_steps": completed_steps,
            "final_loss": final_loss,
            "answer_loss_weight": answer_loss_weight,
            "base_kl_weight": base_kl_weight,
            "max_length": max_length,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_grad_norm": max_grad_norm,
            "gradient_checkpointing": gradient_checkpointing,
        },
        "parameters": {
            "trainable": trainable_count,
            "total": total_count,
            "trainable_fraction": trainable_count / total_count,
        },
        "runtime": {
            "dtype": dtype,
            "device": device,
            "seed": seed,
            "versions": {
                package: _package_version(package)
                for package in ("torch", "transformers", "peft")
            },
        },
    }


def _validate_training_values(**values: Any) -> None:
    for name in (
        "rank",
        "epochs",
        "max_length",
        "batch_size",
        "gradient_accumulation_steps",
        "log_steps",
    ):
        if int(values[name]) <= 0:
            raise TrainLoRAError("{0} must be positive".format(name))
    if values["max_steps"] is not None and int(values["max_steps"]) <= 0:
        raise TrainLoRAError("max_steps must be positive")
    for name in ("learning_rate", "max_grad_norm"):
        if float(values[name]) <= 0.0:
            raise TrainLoRAError("{0} must be positive".format(name))
    dropout = float(values["dropout"])
    if dropout < 0.0 or dropout >= 1.0:
        raise TrainLoRAError("dropout must be in [0, 1)")
    warmup_ratio = float(values["warmup_ratio"])
    if warmup_ratio < 0.0 or warmup_ratio > 1.0:
        raise TrainLoRAError("warmup_ratio must be in [0, 1]")
    if values["scheduler"] != "cosine":
        raise TrainLoRAError("only the cosine scheduler is currently supported")
    if not values["target_modules"]:
        raise TrainLoRAError("target_modules must not be empty")


def _validate_model_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise TrainLoRAError("base model directory does not exist: {0}".format(path))
    if not (path / "config.json").exists():
        raise TrainLoRAError("base model is missing config.json: {0}".format(path))


def _prepare_output_dir(path: Path, override: bool) -> None:
    if path.exists():
        if not override:
            raise TrainLoRAError("output directory already exists: {0}".format(path))
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _resolve_device(torch: Any, device: str) -> Any:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise TrainLoRAError("CUDA was requested but is unavailable")
        return torch.device(device)
    if device == "cpu":
        return torch.device("cpu")
    raise TrainLoRAError("unsupported training device: {0}".format(device))


def _cosine_schedule(steps: int, warmup_ratio: float) -> Callable[[int], float]:
    warmup_steps = int(steps * warmup_ratio)

    def scale(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        remaining = max(1, steps - warmup_steps)
        progress = min(
            1.0,
            max(0.0, (current_step - warmup_steps) / float(remaining)),
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return scale


def _autocast(torch: Any, device: Any, dtype: Any) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type="cpu", enabled=False)


def _optional_int(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("{0} must be an integer or null".format(label))
    return value


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _report(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)
