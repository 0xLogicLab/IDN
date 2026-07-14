"""Teacher-guided LoRA healing for carved Qwen2 models."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from idn.capabilities.base import CapabilitySpec, Example
from idn.models.loader import torch_dtype
from idn.models.qwen2_local.loader import load_qwen2_for_causal_lm


class HealError(RuntimeError):
    """Raised when a carved model cannot be healed."""


@dataclass(frozen=True)
class DistillationRecord:
    example_id: str
    input_ids: Any
    prompt_length: int
    teacher_text: str


@dataclass(frozen=True)
class HealResult:
    adapter_dir: Path
    report: Mapping[str, Any]
    training_log: Sequence[Mapping[str, Any]]


def heal_qwen2_lora(
    *,
    carved_model_path: Path,
    teacher_model_path: Path,
    spec: CapabilitySpec,
    split: str,
    num_examples: int,
    output_dir: Path,
    steps: int = 500,
    learning_rate: float = 2.0e-4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 512,
    max_new_tokens: int = 8,
    temperature: float = 2.0,
    kl_weight: float = 1.0,
    hidden_weight: float = 0.1,
    hidden_layer_stride: int = 4,
    lora_rank: int = 16,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.05,
    target_modules: Sequence[str] = ("gate_proj", "up_proj", "down_proj"),
    warmup_ratio: float = 0.03,
    max_grad_norm: float = 1.0,
    dtype: str = "bfloat16",
    device: str = "cuda",
    local_files_only: bool = True,
    seed: int = 13,
    gradient_checkpointing: bool = True,
    require_valid_teacher: bool = True,
    override: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> HealResult:
    """Heal a carved Qwen2 model using unlabeled prompts and teacher outputs."""

    _validate_heal_args(
        num_examples=num_examples,
        steps=steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        kl_weight=kl_weight,
        hidden_weight=hidden_weight,
        hidden_layer_stride=hidden_layer_stride,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
    )
    carved_model_path = Path(carved_model_path)
    teacher_model_path = Path(teacher_model_path)
    output_dir = Path(output_dir)
    _validate_model_dir(carved_model_path, "carved model")
    _validate_model_dir(teacher_model_path, "teacher model")
    _prepare_output_dir(output_dir, override=override)

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise HealError("torch, transformers, and peft are required for healing") from exc

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = _resolve_device(torch, device)
    resolved_dtype = torch_dtype(dtype)
    if target_device.type == "cpu" and resolved_dtype == torch.bfloat16:
        resolved_dtype = torch.float32
    load_kwargs = {
        "torch_dtype": resolved_dtype,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        "device_map": {"": str(target_device)},
    }

    _report(progress, "loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        str(teacher_model_path),
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _report(progress, "loading frozen teacher")
    teacher = load_qwen2_for_causal_lm(str(teacher_model_path), **load_kwargs)
    teacher.eval()
    teacher.requires_grad_(False)

    _report(progress, "loading carved student")
    student = load_qwen2_for_causal_lm(str(carved_model_path), **load_kwargs)
    student.config.use_cache = False
    if gradient_checkpointing:
        student.gradient_checkpointing_enable()
        student.enable_input_require_grads()

    alpha = int(lora_alpha if lora_alpha is not None else 2 * lora_rank)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    student = get_peft_model(student, lora_config)
    student.train()
    trainable_parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in student.parameters())
    if not trainable_parameters:
        raise HealError("LoRA did not create any trainable parameters")
    _report(
        progress,
        "trainable parameters: {0}/{1}".format(trainable_count, total_count),
    )

    examples = spec.load_split(split=split, limit=num_examples)
    if not examples:
        raise HealError("no examples loaded for split: {0}".format(split))
    _report(progress, "generating teacher answers")
    records, skipped = build_distillation_records(
        teacher=teacher,
        tokenizer=tokenizer,
        spec=spec,
        examples=examples,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        device=target_device,
        require_valid_teacher=require_valid_teacher,
        progress=progress,
    )
    if not records:
        raise HealError("teacher produced no usable distillation records")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_cosine_schedule(steps=steps, warmup_ratio=warmup_ratio),
    )
    hidden_indices = hidden_state_indices(
        num_hidden_layers=int(student.base_model.model.config.num_hidden_layers),
        stride=hidden_layer_stride,
    )
    started_at = _utc_now()
    training_log: List[Mapping[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    record_order = list(range(len(records)))
    record_cursor = 0

    initial_report = _build_report(
        status="running",
        carved_model_path=carved_model_path,
        teacher_model_path=teacher_model_path,
        spec=spec,
        split=split,
        requested_examples=num_examples,
        usable_examples=len(records),
        skipped_examples=skipped,
        steps=steps,
        completed_steps=0,
        dtype=dtype,
        device=str(target_device),
        trainable_parameters=trainable_count,
        total_parameters=total_count,
        lora_rank=lora_rank,
        lora_alpha=alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        temperature=temperature,
        kl_weight=kl_weight,
        hidden_weight=hidden_weight,
        hidden_layers=hidden_indices,
        seed=seed,
        started_at=started_at,
        final_losses=None,
    )
    write_json_atomic(output_dir / "report.json", initial_report)
    write_json_atomic(output_dir / "training_log.json", {"steps": []})

    for step in range(1, steps + 1):
        totals = {"loss": 0.0, "kl_loss": 0.0, "hidden_loss": 0.0}
        for _micro_step in range(gradient_accumulation_steps):
            if record_cursor >= len(record_order):
                random.shuffle(record_order)
                record_cursor = 0
            record = records[record_order[record_cursor]]
            record_cursor += 1
            input_ids = record.input_ids.to(target_device)
            attention_mask = torch.ones_like(input_ids)
            output_hidden_states = hidden_weight > 0.0

            with torch.no_grad():
                teacher_output = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=output_hidden_states,
                )
            with _autocast(torch, target_device, resolved_dtype):
                student_output = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=output_hidden_states,
                )
                losses = distillation_losses(
                    student_output=student_output,
                    teacher_output=teacher_output,
                    prompt_length=record.prompt_length,
                    temperature=temperature,
                    kl_weight=kl_weight,
                    hidden_weight=hidden_weight,
                    hidden_indices=hidden_indices,
                )
                scaled_loss = losses["loss"] / gradient_accumulation_steps
            scaled_loss.backward()
            for name in totals:
                totals[name] += float(losses[name].detach().cpu())

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        entry = {
            "step": step,
            "loss": totals["loss"] / gradient_accumulation_steps,
            "kl_loss": totals["kl_loss"] / gradient_accumulation_steps,
            "hidden_loss": totals["hidden_loss"] / gradient_accumulation_steps,
            "grad_norm": float(grad_norm.detach().cpu()),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        training_log.append(entry)
        _report(
            progress,
            "step {0}/{1} loss={2:.6f} kl={3:.6f} hidden={4:.6f}".format(
                step,
                steps,
                entry["loss"],
                entry["kl_loss"],
                entry["hidden_loss"],
            ),
        )
        write_json_atomic(output_dir / "training_log.json", {"steps": training_log})

    adapter_dir = output_dir / "adapter"
    _report(progress, "saving LoRA adapter: {0}".format(adapter_dir))
    student.save_pretrained(str(adapter_dir))
    final_losses = dict(training_log[-1])
    report = _build_report(
        status="complete",
        carved_model_path=carved_model_path,
        teacher_model_path=teacher_model_path,
        spec=spec,
        split=split,
        requested_examples=num_examples,
        usable_examples=len(records),
        skipped_examples=skipped,
        steps=steps,
        completed_steps=steps,
        dtype=dtype,
        device=str(target_device),
        trainable_parameters=trainable_count,
        total_parameters=total_count,
        lora_rank=lora_rank,
        lora_alpha=alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        temperature=temperature,
        kl_weight=kl_weight,
        hidden_weight=hidden_weight,
        hidden_layers=hidden_indices,
        seed=seed,
        started_at=started_at,
        final_losses=final_losses,
    )
    write_json_atomic(output_dir / "report.json", report)
    return HealResult(
        adapter_dir=adapter_dir,
        report=report,
        training_log=tuple(training_log),
    )


def build_distillation_records(
    *,
    teacher: Any,
    tokenizer: Any,
    spec: CapabilitySpec,
    examples: Sequence[Example],
    max_length: int,
    max_new_tokens: int,
    device: Any,
    require_valid_teacher: bool,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[List[DistillationRecord], int]:
    import torch

    records = []
    skipped = 0
    prompt_budget = max_length - max_new_tokens
    if prompt_budget <= 0:
        raise HealError("max_length must be greater than max_new_tokens")
    for index, example in enumerate(examples, start=1):
        _report(progress, "teacher example {0}/{1}".format(index, len(examples)))
        prompt = render_prompt(tokenizer, spec.format_prompt(example))
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_ids = list(prompt_ids)[-prompt_budget:]
        if not prompt_ids:
            skipped += 1
            continue
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output_ids = teacher.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0, input_ids.shape[-1] :]
        teacher_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        parsed = spec.parse_prediction(teacher_text)
        if not teacher_text or (
            require_valid_teacher and not spec.verify(teacher_text, parsed)
        ):
            skipped += 1
            continue
        records.append(
            DistillationRecord(
                example_id=example.id,
                input_ids=output_ids.detach().cpu(),
                prompt_length=int(input_ids.shape[-1]),
                teacher_text=teacher_text,
            )
        )
    return records, skipped


def distillation_losses(
    *,
    student_output: Any,
    teacher_output: Any,
    prompt_length: int,
    temperature: float,
    kl_weight: float,
    hidden_weight: float,
    hidden_indices: Sequence[int],
) -> Mapping[str, Any]:
    import torch
    import torch.nn.functional as functional

    prediction_start = prompt_length - 1
    prediction_stop = student_output.logits.shape[1] - 1
    if prediction_start < 0 or prediction_stop <= prediction_start:
        raise HealError("distillation record has no teacher-output tokens")
    student_logits = student_output.logits[:, prediction_start:prediction_stop, :]
    teacher_logits = teacher_output.logits[:, prediction_start:prediction_stop, :]
    teacher_probabilities = functional.softmax(
        teacher_logits / temperature,
        dim=-1,
    )
    student_log_probabilities = functional.log_softmax(
        student_logits / temperature,
        dim=-1,
    )
    kl_loss = functional.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="none",
    ).sum(dim=-1).mean() * (temperature ** 2)

    hidden_loss = student_logits.new_zeros(())
    if hidden_weight > 0.0:
        if student_output.hidden_states is None or teacher_output.hidden_states is None:
            raise HealError("hidden states were not returned for activation matching")
        layer_losses = []
        for index in hidden_indices:
            student_hidden = student_output.hidden_states[index][
                :, prediction_start:prediction_stop, :
            ]
            teacher_hidden = teacher_output.hidden_states[index][
                :, prediction_start:prediction_stop, :
            ]
            if student_hidden.shape != teacher_hidden.shape:
                raise HealError(
                    "student/teacher hidden shape mismatch at index {0}: {1} vs {2}".format(
                        index,
                        tuple(student_hidden.shape),
                        tuple(teacher_hidden.shape),
                    )
                )
            layer_losses.append(
                functional.mse_loss(student_hidden.float(), teacher_hidden.float())
            )
        if layer_losses:
            hidden_loss = torch.stack(layer_losses).mean()
    total = kl_weight * kl_loss + hidden_weight * hidden_loss
    return {"loss": total, "kl_loss": kl_loss, "hidden_loss": hidden_loss}


def hidden_state_indices(num_hidden_layers: int, stride: int) -> Tuple[int, ...]:
    if num_hidden_layers <= 0:
        raise HealError("num_hidden_layers must be positive")
    if stride <= 0:
        raise HealError("hidden layer stride must be positive")
    indices = list(range(stride, num_hidden_layers + 1, stride))
    if not indices or indices[-1] != num_hidden_layers:
        indices.append(num_hidden_layers)
    return tuple(indices)


def render_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(
        "{0}: {1}".format(message["role"], message["content"])
        for message in messages
    ) + "\nassistant:"


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{0}.tmp".format(path.name))
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _build_report(
    *,
    status: str,
    carved_model_path: Path,
    teacher_model_path: Path,
    spec: CapabilitySpec,
    split: str,
    requested_examples: int,
    usable_examples: int,
    skipped_examples: int,
    steps: int,
    completed_steps: int,
    dtype: str,
    device: str,
    trainable_parameters: int,
    total_parameters: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str],
    learning_rate: float,
    gradient_accumulation_steps: int,
    temperature: float,
    kl_weight: float,
    hidden_weight: float,
    hidden_layers: Sequence[int],
    seed: int,
    started_at: str,
    final_losses: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "artifact_version": 1,
        "status": status,
        "method": "teacher_generated_lora_distillation",
        "capability": spec.id,
        "split": split,
        "carved_model_path": str(carved_model_path),
        "teacher_model_path": str(teacher_model_path),
        "uses_gold_labels": False,
        "requested_examples": requested_examples,
        "usable_examples": usable_examples,
        "skipped_examples": skipped_examples,
        "steps": steps,
        "completed_steps": completed_steps,
        "dtype": dtype,
        "device": device,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "lora": {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": list(target_modules),
        },
        "optimization": {
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "temperature": temperature,
            "kl_weight": kl_weight,
            "hidden_weight": hidden_weight,
            "hidden_state_indices": list(hidden_layers),
        },
        "seed": seed,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "final_losses": dict(final_losses) if final_losses is not None else None,
    }


def _cosine_schedule(steps: int, warmup_ratio: float) -> Callable[[int], float]:
    warmup_steps = int(steps * warmup_ratio)

    def scale(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        remaining = max(1, steps - warmup_steps)
        progress = min(1.0, max(0.0, (current_step - warmup_steps) / remaining))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return scale


def _autocast(torch: Any, device: Any, dtype: Any) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type="cpu", enabled=False)


def _resolve_device(torch: Any, device: str) -> Any:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise HealError("CUDA was requested but is unavailable")
        return torch.device(device)
    if device == "cpu":
        return torch.device("cpu")
    raise HealError("unsupported heal device: {0}".format(device))


def _prepare_output_dir(output_dir: Path, override: bool) -> None:
    if output_dir.exists():
        if not override:
            raise HealError("output directory already exists: {0}".format(output_dir))
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _validate_model_dir(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise HealError("{0} directory does not exist: {1}".format(label, path))
    if not (path / "config.json").exists():
        raise HealError("{0} is missing config.json: {1}".format(label, path))


def _validate_heal_args(**values: Any) -> None:
    positive_ints = (
        "num_examples",
        "steps",
        "gradient_accumulation_steps",
        "max_length",
        "max_new_tokens",
        "hidden_layer_stride",
        "lora_rank",
    )
    for name in positive_ints:
        if int(values[name]) <= 0:
            raise HealError("{0} must be positive".format(name))
    positive_floats = ("learning_rate", "temperature", "max_grad_norm")
    for name in positive_floats:
        if float(values[name]) <= 0.0:
            raise HealError("{0} must be positive".format(name))
    for name in ("kl_weight", "hidden_weight", "lora_dropout", "warmup_ratio"):
        if float(values[name]) < 0.0:
            raise HealError("{0} must be non-negative".format(name))
    if float(values["lora_dropout"]) >= 1.0:
        raise HealError("lora_dropout must be less than 1")
    if float(values["warmup_ratio"]) > 1.0:
        raise HealError("warmup_ratio must be at most 1")
    if float(values["kl_weight"]) == 0.0 and float(values["hidden_weight"]) == 0.0:
        raise HealError("at least one heal loss weight must be positive")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _report(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)
